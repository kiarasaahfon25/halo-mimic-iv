'''
Standalone benchmark of HALO synthetic-data generation speed.

Runs HALO's reference sampler (HALOModel.sample in model.py, unchanged) on the
trained checkpoint at several sample batch sizes and reports, for each one:
  - seconds per generated patient (and the projected time for a full run)
  - visits generated per patient, codes per visit, and how many records ended
  - fine-head steps per visit (the number of passes through the code-level head)
  - where the time goes: coarse transformer, fine head, rest of the sampling loop
It also checks that the model is really on the GPU, and times the pieces of a
single fine-head step (mask multiply, matrix multiply, Bernoulli sampling) with
and without the mask applied once in advance and with TF32.

Nothing in the repo is modified: model.py and test_model.py are not changed, and
the only file written is a JSON summary in ./results/. Timing is added by wrapping
the model's methods from the outside, so the generated records are the same as
the reference code would produce.

Usage:
  python benchmark_sampling.py --target mortality
  python benchmark_sampling.py --target mortality --batch_sizes 1,4,16,64,256 --num_patients 256 --budget 900
  python benchmark_sampling.py --smoke_test     # tiny random model, just checks the script runs
'''
import os
import json
import time
import pickle
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from model import HALOModel
from config import HALOConfig, TARGET_SIZES
from halo_utils import setup, paths


# Load checkpoint weights into the model. Works with both checkpoint formats:
# the older one (full state_dict including the autoregressive masks, plus the
# optimizer) and the newer weights-only one without the masks. Kept here instead
# of importing from halo_utils so the benchmark runs with either version of it.
def load_checkpoint_weights(model, weights):
  missing, unexpected = model.load_state_dict(weights, strict=False)
  if unexpected or any(not k.endswith('.mask') for k in missing):
    raise RuntimeError(f'Checkpoint does not match the model: missing {missing}, unexpected {unexpected}')


def parse_benchmark_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--target', choices=list(TARGET_SIZES), default='labels',
                      help='which trained HALO model to benchmark')
  parser.add_argument('--batch_sizes', default='1,4,16,64,256',
                      help='comma-separated sample batch sizes to try')
  parser.add_argument('--num_patients', type=int, default=256,
                      help='patients to generate at each batch size')
  parser.add_argument('--budget', type=float, default=900,
                      help='maximum seconds spent at each batch size; a batch still running at the limit is stopped')
  parser.add_argument('--micro_reps', type=int, default=10,
                      help='repetitions for each timing of a single fine-head step')
  parser.add_argument('--skip_micro', action='store_true',
                      help='skip the single-step timings')
  parser.add_argument('--smoke_test', action='store_true',
                      help='use a tiny randomly initialised model instead of the checkpoint (checks the script runs)')
  return parser.parse_args()


args = parse_benchmark_args()
device = setup(seed=4)
on_gpu = device.type == 'cuda'

if args.smoke_test:
  config = HALOConfig(target=args.target, code_vocab_size=200, n_embd=48, n_layer=2, n_head=12)
else:
  config = HALOConfig(target=args.target)
output_paths = paths(args.target)
END_IDX = config.code_vocab_size + config.label_vocab_size + 1
START_IDX = config.code_vocab_size + config.label_vocab_size


def sync():
  if on_gpu:
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# 1. Environment and GPU check
# ---------------------------------------------------------------------------
print('=' * 70)
print('ENVIRONMENT')
print('=' * 70)
print(f'torch {torch.__version__}, built for CUDA {torch.version.cuda}')
print(f'torch.cuda.is_available(): {torch.cuda.is_available()}')
print(f'device used: {device}')
env_info = {'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
            'cuda_available': torch.cuda.is_available(), 'device': str(device)}
if on_gpu:
  props = torch.cuda.get_device_properties(0)
  print(f'GPU: {props.name}, {props.total_memory / 1e9:.1f} GB')
  print(f'TF32 for matmul enabled: {torch.backends.cuda.matmul.allow_tf32}')
  env_info.update({'gpu': props.name, 'gpu_memory_gb': props.total_memory / 1e9,
                   'tf32_matmul_default': torch.backends.cuda.matmul.allow_tf32})
else:
  print('\n' + '!' * 70)
  print('WARNING: PyTorch cannot see a GPU, so everything below runs on the CPU.')
  print('Generation in test_model.py would also be on the CPU (setup() falls back')
  print('silently). Check that this environment has a CUDA build of PyTorch that')
  print("matches the node's driver before trusting any speed numbers.")
  print('!' * 70 + '\n')

# Load the model the same way test_model.py does
model = HALOModel(config).to(device)
if not args.smoke_test:
  checkpoint = torch.load(output_paths['model'], map_location='cpu')
  load_checkpoint_weights(model, checkpoint['model'])
  del checkpoint
model.eval()

param_device = next(model.parameters()).device
print(f'model parameters are on: {param_device}')
print(f'vocabulary: {config.code_vocab_size} codes, {config.total_vocab_size} total tokens; '
      f'fine-head width {config.n_embd + config.total_vocab_size}')
if on_gpu:
  print(f'GPU memory used by the model: {torch.cuda.memory_allocated() / 1e9:.1f} GB')
  env_info['model_gpu_memory_gb'] = torch.cuda.memory_allocated() / 1e9
env_info['model_device'] = str(param_device)

full_run_size = None
if not args.smoke_test and os.path.exists('./data/trainDataset.pkl'):
  full_run_size = len(pickle.load(open('./data/trainDataset.pkl', 'rb')))
  print(f'full synthetic dataset size (= training set): {full_run_size} patients')


# ---------------------------------------------------------------------------
# 2. Reference sampling loop, copied unchanged from test_model.py
# ---------------------------------------------------------------------------
def sample_sequence(model, length, context, batch_size, device='cuda', sample=True):
  empty = torch.zeros((1,1,config.total_vocab_size), device=device, dtype=torch.float32).repeat(batch_size, 1, 1)
  context = torch.tensor(context, device=device, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1)
  prev = context.unsqueeze(1)
  context = None
  with torch.no_grad():
    for _ in range(length-1):
      prev = model.sample(torch.cat((prev,empty), dim=1), sample)
      if torch.sum(torch.sum(prev[:,:,config.code_vocab_size+config.label_vocab_size+1], dim=1).bool().int(), dim=0).item() == batch_size:
        break
  ehr = prev.cpu().detach().numpy()
  prev = None
  empty = None
  return ehr


# ---------------------------------------------------------------------------
# 3. Instrumentation: wrap the model's methods to count and time them.
#    The wrappers call the original methods with the same arguments, so the
#    sampling itself is unchanged. The reference loop already waits for the GPU
#    at every step (torch.nonzero), so the extra synchronisation for timing
#    adds very little.
# ---------------------------------------------------------------------------
class BudgetExceeded(Exception):
  pass


class Stats:
  def reset(self, budget):
    self.budget = budget
    self.start = time.perf_counter()
    self.visit_calls = 0          # calls to model.sample (one per generated visit position)
    self.steps = 0                # calls to the fine head (one per code committed)
    self.steps_this_call = 0
    self.steps_per_call = []
    self.t_transformer = 0.0
    self.t_head = 0.0

  def elapsed(self):
    return time.perf_counter() - self.start


stats = Stats()
stats.reset(float('inf'))

orig_transformer_forward = model.transformer.forward
orig_head_sample = model.ehr_head.sample
orig_model_sample = model.sample


def timed_transformer_forward(*a, **k):
  sync()
  t = time.perf_counter()
  out = orig_transformer_forward(*a, **k)
  sync()
  stats.t_transformer += time.perf_counter() - t
  return out


def timed_head_sample(*a, **k):
  if stats.elapsed() > stats.budget:
    raise BudgetExceeded()
  sync()
  t = time.perf_counter()
  out = orig_head_sample(*a, **k)
  sync()
  stats.t_head += time.perf_counter() - t
  stats.steps += 1
  stats.steps_this_call += 1
  return out


def counted_model_sample(*a, **k):
  stats.steps_this_call = 0
  out = orig_model_sample(*a, **k)
  stats.visit_calls += 1
  stats.steps_per_call.append(stats.steps_this_call)
  return out


model.transformer.forward = timed_transformer_forward
model.ehr_head.sample = timed_head_sample
model.sample = counted_model_sample


# Summarise generated records the same way convert_ehr in test_model.py reads them
def summarize_records(ehr):
  out = []
  for p in range(len(ehr)):
    n_visits, codes_per_visit, ended = 0, [], False
    for j in range(2, len(ehr[p])):
      indices = np.nonzero(ehr[p][j])[0]
      n_codes = int(np.sum(indices < config.code_vocab_size))
      if n_codes > 0:
        n_visits += 1
        codes_per_visit.append(n_codes)
      if END_IDX in indices:
        ended = True
        break
    out.append({'visits': n_visits, 'codes_per_visit': codes_per_visit, 'ended': ended})
  return out


def fmt_time(seconds):
  if seconds is None:
    return 'n/a'
  if seconds < 120:
    return f'{seconds:.0f} s'
  if seconds < 7200:
    return f'{seconds / 60:.1f} min'
  if seconds < 172800:
    return f'{seconds / 3600:.1f} h'
  return f'{seconds / 86400:.1f} days'


# ---------------------------------------------------------------------------
# 4. Generation benchmark at each batch size
# ---------------------------------------------------------------------------
stoken = np.zeros(config.total_vocab_size)
stoken[START_IDX] = 1
batch_sizes = [int(b) for b in args.batch_sizes.split(',')]

print('\n' + '=' * 70)
print(f'GENERATION BENCHMARK: {args.num_patients} patients per batch size, '
      f'at most {fmt_time(args.budget)} each')
print('=' * 70)

results = []
for bs in batch_sizes:
  print(f'\n--- sample_batch_size = {bs} ---', flush=True)
  if on_gpu:
    torch.cuda.reset_peak_memory_stats()
  stats.reset(args.budget)
  records = []
  stopped_mid_batch = False
  stopped_between_batches = False
  for i in range(0, args.num_patients, bs):
    if stats.elapsed() > args.budget:
      stopped_between_batches = True
      break
    n = min(bs, args.num_patients - i)
    torch.manual_seed(4 + i)  # same per-batch seeding as test_model.py
    try:
      ehr = sample_sequence(model, config.n_ctx, stoken, batch_size=n, device=device, sample=True)
    except BudgetExceeded:
      stopped_mid_batch = True
      break
    records += summarize_records(ehr)
    t_completed = stats.elapsed()   # time up to the end of the last finished batch
    print(f'  {len(records)}/{args.num_patients} patients, {fmt_time(t_completed)} elapsed', flush=True)
  elapsed = stats.elapsed()

  n_done = len(records)
  # Steps of a visit that was cut off by the budget are not a complete visit
  complete_calls = stats.steps_per_call
  r = {
    'batch_size': bs,
    'patients_completed': n_done,
    'stopped_by_budget': stopped_mid_batch or stopped_between_batches,
    'stopped_mid_batch': stopped_mid_batch,
    'elapsed_s': elapsed,
    'visit_calls': stats.visit_calls,
    'fine_head_steps': stats.steps,
    'steps_per_visit_call_mean': float(np.mean(complete_calls)) if complete_calls else None,
    'steps_per_visit_call_max': int(np.max(complete_calls)) if complete_calls else None,
    'ms_per_fine_head_step': 1000 * stats.t_head / stats.steps if stats.steps else None,
    'time_transformer_s': stats.t_transformer,
    'time_fine_head_s': stats.t_head,
    'time_rest_of_loop_s': elapsed - stats.t_transformer - stats.t_head,
    'peak_gpu_memory_gb': torch.cuda.max_memory_allocated() / 1e9 if on_gpu else None,
  }
  if n_done > 0:
    # Time per patient only counts batches that finished
    r['s_per_patient'] = t_completed / n_done
    r['visits_per_patient_mean'] = float(np.mean([x['visits'] for x in records]))
    all_codes = [c for x in records for c in x['codes_per_visit']]
    r['codes_per_visit_mean'] = float(np.mean(all_codes)) if all_codes else 0.0
    r['codes_per_visit_max'] = int(np.max(all_codes)) if all_codes else 0
    r['fraction_records_ended'] = float(np.mean([x['ended'] for x in records]))
  if stopped_mid_batch and n_done == 0:
    # Not even one batch finished: the elapsed time is a lower bound for one batch
    r['s_per_patient_lower_bound'] = elapsed / bs
  if full_run_size and r.get('s_per_patient'):
    r['projected_full_run_s'] = r['s_per_patient'] * full_run_size
  results.append(r)

  print(f'  patients completed: {n_done}' + (' (stopped by time budget)' if r['stopped_by_budget'] else ''))
  print(f'  visit positions generated: {stats.visit_calls}, fine-head steps: {stats.steps}')
  if complete_calls:
    print(f'  fine-head steps per visit position: mean {r["steps_per_visit_call_mean"]:.1f}, max {r["steps_per_visit_call_max"]}')
  if stats.steps:
    print(f'  time per fine-head step: {r["ms_per_fine_head_step"]:.2f} ms')
  print(f'  time split: transformer {fmt_time(stats.t_transformer)}, fine head {fmt_time(stats.t_head)}, '
        f'rest of loop {fmt_time(r["time_rest_of_loop_s"])}')
  if n_done > 0:
    print(f'  generated visits per patient: {r["visits_per_patient_mean"]:.2f}, codes per visit: '
          f'mean {r["codes_per_visit_mean"]:.1f} (max {r["codes_per_visit_max"]}), '
          f'records with an END token: {100 * r["fraction_records_ended"]:.0f}%')
  if r.get('s_per_patient'):
    print(f'  seconds per patient: {r["s_per_patient"]:.3f}')
  elif r.get('s_per_patient_lower_bound'):
    print(f'  seconds per patient: more than {r["s_per_patient_lower_bound"]:.1f} (first batch did not finish)')
  if r.get('projected_full_run_s'):
    print(f'  projected full run ({full_run_size} patients, one job): {fmt_time(r["projected_full_run_s"])}')

# Put the model's own methods back before the single-step timings
model.transformer.forward = orig_transformer_forward
model.ehr_head.sample = orig_head_sample
model.sample = orig_model_sample


# ---------------------------------------------------------------------------
# 5. Single fine-head step timings
#    Uses random inputs of the right shape; the trained weights are only read.
# ---------------------------------------------------------------------------
def time_ms(fn, reps):
  fn()
  sync()
  ts = []
  for _ in range(reps):
    sync()
    t = time.perf_counter()
    fn()
    sync()
    ts.append(time.perf_counter() - t)
  return 1000 * float(np.median(ts))


micro = []
if not args.skip_micro:
  print('\n' + '=' * 70)
  print('SINGLE FINE-HEAD STEP TIMINGS (median of %d runs, milliseconds)' % args.micro_reps)
  print('=' * 70)
  layer = model.ehr_head.auto1
  width = layer.weight.shape[1]
  tf32_default = torch.backends.cuda.matmul.allow_tf32
  with torch.no_grad():
    premasked = layer.mask * layer.weight   # the mask applied once, as option 1 would do
    t_mask = time_ms(lambda: layer.mask * layer.weight, args.micro_reps)
    print(f'mask multiply for one layer (same for every batch size): {t_mask:.2f}')
    print(f'{"batch":>6} {"layer now":>10} {"premasked":>10} {"pre+TF32":>10} {"head step":>10} '
          f'{"sampling":>10} {"max |diff|":>11}')
    for bs in batch_sizes:
      x = torch.randn(bs, 1, width, device=device)
      t_layer = time_ms(lambda: layer(x), args.micro_reps)
      t_pre = time_ms(lambda: F.linear(x, premasked, layer.bias), args.micro_reps)
      diff = (layer(x) - F.linear(x, premasked, layer.bias)).abs().max().item()
      t_tf32 = None
      if on_gpu:
        torch.backends.cuda.matmul.allow_tf32 = True
        t_tf32 = time_ms(lambda: F.linear(x, premasked, layer.bias), args.micro_reps)
        torch.backends.cuda.matmul.allow_tf32 = tf32_default
      history = torch.randn(bs, 2, config.n_embd, device=device)
      visits = torch.zeros(bs, 2, config.total_vocab_size, device=device)
      t_head = time_ms(lambda: model.ehr_head.sample(history, visits), args.micro_reps)
      probs = torch.rand(bs, 1, config.total_vocab_size, device=device)
      t_samp = time_ms(lambda: torch.nonzero(torch.bernoulli(probs)[:, 0, :], as_tuple=True)[1].min(), args.micro_reps)
      micro.append({'batch_size': bs, 'mask_multiply_ms': t_mask, 'layer_now_ms': t_layer,
                    'layer_premasked_ms': t_pre, 'layer_premasked_tf32_ms': t_tf32,
                    'full_head_step_ms': t_head, 'bernoulli_nonzero_ms': t_samp,
                    'premasked_max_abs_diff': diff})
      tf32_str = f'{t_tf32:10.2f}' if t_tf32 is not None else f'{"n/a":>10}'
      print(f'{bs:>6} {t_layer:10.2f} {t_pre:10.2f} {tf32_str} {t_head:10.2f} {t_samp:10.2f} {diff:11.2e}')
    del premasked
  print('"layer now" is one fine-head layer as the code runs it (mask multiply + matrix multiply);')
  print('"head step" is both layers. A max |diff| of 0 means applying the mask once gives identical output.')


# ---------------------------------------------------------------------------
# 6. Summary
# ---------------------------------------------------------------------------
print('\n' + '=' * 70)
print('SUMMARY')
print('=' * 70)
print(f'{"batch":>6} {"patients":>9} {"s/patient":>10} {"steps/visit":>12} {"ms/step":>8} '
      f'{"visits/pt":>10} {"ended":>6} {"full run":>10}')
for r in results:
  spp = f'{r["s_per_patient"]:.3f}' if r.get('s_per_patient') else (
        f'>{r["s_per_patient_lower_bound"]:.1f}' if r.get('s_per_patient_lower_bound') else 'n/a')
  spv = f'{r["steps_per_visit_call_mean"]:.1f}' if r['steps_per_visit_call_mean'] is not None else 'n/a'
  mps = f'{r["ms_per_fine_head_step"]:.1f}' if r['ms_per_fine_head_step'] else 'n/a'
  vpp = f'{r["visits_per_patient_mean"]:.2f}' if r.get('visits_per_patient_mean') is not None else 'n/a'
  end = f'{100 * r["fraction_records_ended"]:.0f}%' if r.get('fraction_records_ended') is not None else 'n/a'
  print(f'{r["batch_size"]:>6} {r["patients_completed"]:>9} {spp:>10} {spv:>12} {mps:>8} {vpp:>10} '
        f'{end:>6} {fmt_time(r.get("projected_full_run_s")):>10}')
if not on_gpu:
  print('\nReminder: these numbers are from the CPU, not the GPU.')

out_path = f'./results/benchmark_sampling_{args.target}{"_smoke" if args.smoke_test else ""}.json'
with open(out_path, 'w') as f:
  json.dump({'args': vars(args), 'environment': env_info, 'full_run_size': full_run_size,
             'generation': results, 'single_step': micro}, f, indent=2)
print(f'\nSaved to {out_path}')
