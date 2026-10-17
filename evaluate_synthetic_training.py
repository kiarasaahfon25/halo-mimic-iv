import torch
import pickle
import numpy as np
import torch.nn as nn
from sklearn import metrics
from config import HALOConfig
from halo_utils import parse_args, setup, target_vector, paths
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Usage: python evaluate_synthetic_training.py --target {labels,mortality,readmission} [--subgroups]
#   labels:      one classifier per CCS phenotype (diagnosis prediction)
#   mortality:   in-hospital mortality
#   readmission: 30-day readmission
# --subgroups also reports F1 by age group and sex
args = parse_args(subgroups=True)
config = HALOConfig(target=args.target)
output_paths = paths(args.target)
device = setup(seed=4)

LR = 0.001
EPOCHS = 25
BATCH_SIZE = 512
LSTM_HIDDEN_DIM = 32
EMBEDDING_DIM = 64
NUM_TRAIN_EXAMPLES = 5000
NUM_TEST_EXAMPLES = 1000
NUM_VAL_EXAMPLES = 500

# One task per label in the target (25 CCS phenotypes, or a single outcome)
if args.target == 'labels':
    task_names = pickle.load(open("./data/idToLabel.pkl", "rb"))
else:
    task_names = [args.target]
LABEL_IDX_LIST = list(range(config.label_vocab_size))

train_ehr_dataset = pickle.load(open('./data/trainDataset.pkl', 'rb'))
val_ehr_dataset = pickle.load(open('./data/valDataset.pkl', 'rb'))
test_ehr_dataset = pickle.load(open('./data/testDataset.pkl', 'rb'))
halo_ehr_dataset = pickle.load(open(output_paths['synthetic'], 'rb'))
halo_ehr_dataset = [
    p for p in halo_ehr_dataset
    if len(p['visits']) > 0
]


class DiagnosisModel(nn.Module):
    def __init__(self, config):
        super(DiagnosisModel, self).__init__()
        self.embedding = nn.Linear(config.code_vocab_size, EMBEDDING_DIM, bias=False)
        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(input_size=EMBEDDING_DIM,
                            hidden_size=LSTM_HIDDEN_DIM,
                            num_layers=2,
                            dropout=0.5,
                            batch_first=True,
                            bidirectional=True)
        self.fc = nn.Linear(2*LSTM_HIDDEN_DIM, 1)

    def forward(self, input_visits, lengths):
        visit_emb = self.embedding(input_visits)
        visit_emb = self.dropout(visit_emb)
        packed_input = pack_padded_sequence(visit_emb, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out_forward = output[range(len(output)), lengths - 1, :LSTM_HIDDEN_DIM]
        out_reverse = output[:, 0, LSTM_HIDDEN_DIM:]
        out_combined = torch.cat((out_forward, out_reverse), 1)

        patient_embedding = self.fc(out_combined)
        patient_embedding = torch.squeeze(patient_embedding, 1)
        prob = torch.sigmoid(patient_embedding)

        return prob


def get_label(p, label_idx):
    return target_vector(p, args.target)[label_idx]


def get_batch(ehr_dataset, loc, batch_size, label_idx):
    ehr = ehr_dataset[loc:loc+batch_size]
    batch_ehr = np.zeros((len(ehr), config.n_ctx, config.code_vocab_size))
    batch_labels = np.array([get_label(p, label_idx) for p in ehr])
    batch_lens = np.zeros(len(ehr))

    for i, p in enumerate(ehr):
        visits = p['visits']
        batch_lens[i] = len(visits)

        for j, v in enumerate(visits):
            batch_ehr[i,j][v] = 1

    return batch_ehr, batch_labels, batch_lens


# Sample a balanced dataset: half positives, half negatives
# (positives are sampled with replacement only when there are too few)
# Returns None when the dataset has no positive examples
def sample_balanced(dataset, label_idx, num_examples):
    pos = [p for p in dataset if get_label(p, label_idx) == 1]
    neg = [p for p in dataset if get_label(p, label_idx) == 0]
    half = int(num_examples/2)
    if len(pos) == 0:
        return None
    return list(
        np.random.choice(pos, half, replace=len(pos) < half)
    ) + list(
        np.random.choice(neg, half, replace=False)
    )


def train_model(model, train_dataset, val_dataset, save_name, label_idx):
    global_loss = 1e10
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    bce = nn.BCELoss()

    for e in range(EPOCHS):
        np.random.shuffle(train_dataset)
        train_losses = []

        for i in range(0, len(train_dataset), BATCH_SIZE):
            model.train()

            batch_ehr, batch_labels, batch_lens = get_batch(train_dataset, i, BATCH_SIZE, label_idx)
            batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
            batch_labels = torch.tensor(batch_labels, dtype=torch.float32).to(device)

            optimizer.zero_grad()
            prob = model(batch_ehr, batch_lens)
            loss = bce(prob, batch_labels)
            train_losses.append(loss.cpu().detach().numpy())
            loss.backward()
            optimizer.step()

        cur_train_loss = np.mean(train_losses)
        print("Epoch %d Training Loss:%.5f" % (e, cur_train_loss))

        model.eval()
        with torch.no_grad():
            val_losses = []
            for v_i in range(0, len(val_dataset), BATCH_SIZE):
                batch_ehr, batch_labels, batch_lens = get_batch(val_dataset, v_i, BATCH_SIZE, label_idx)
                batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
                batch_labels = torch.tensor(batch_labels, dtype=torch.float32).to(device)

                prob = model(batch_ehr, batch_lens)
                val_loss = bce(prob, batch_labels)
                val_losses.append(val_loss.cpu().detach().numpy())

            cur_val_loss = np.mean(val_losses)
            print("Epoch %d Validation Loss:%.5f" % (e, cur_val_loss))

            if cur_val_loss < global_loss:
                global_loss = cur_val_loss
                state = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                torch.save(state, f'./save/{save_name}')
                print('------------ Save best model ------------')

    model.load_state_dict(state['model'])


def get_age_bin(age):
    if 0 <= age <= 14:
        return "0-14"
    elif 15 <= age <= 24:
        return "15-24"
    elif 25 <= age <= 44:
        return "25-44"
    elif 45 <= age <= 64:
        return "45-64"
    elif 65 <= age <= 74:
        return "65-74"
    elif age >= 75:
        return "75+"


# F1 score for each subgroup (e.g. age bin or sex)
def subgroup_f1(name, group_names, group_list, true_list, round_list):
    groups = {g: [] for g in group_names}
    for group, true, pred in zip(group_list, true_list, round_list):
        if group in groups:
            groups[group].append((true, pred))

    group_f1 = {}
    for group, patients in groups.items():
        if len(patients) == 0:
            group_f1[group] = None
            continue

        group_true = [x[0] for x in patients]
        group_pred = [x[1] for x in patients]
        group_f1[group] = metrics.f1_score(group_true, group_pred, zero_division=0)

        print(
            f"{name} {group} - N: {len(patients)}, "
            f"Actual positives: {sum(group_true)}, "
            f"Predicted positives: {sum(group_pred)}, "
            f"F1: {group_f1[group]:.4f}"
        )
    return group_f1


def test_model(model, test_dataset, label_idx):
    loss_list = []
    pred_list = []
    true_list = []
    bce = nn.BCELoss()

    model.eval()
    with torch.no_grad():
        for i in range(0, len(test_dataset), BATCH_SIZE):
            batch_ehr, batch_labels, batch_lens = get_batch(test_dataset, i, BATCH_SIZE, label_idx)
            batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
            batch_labels = torch.tensor(batch_labels, dtype=torch.float32).to(device)

            prob = model(batch_ehr, batch_lens)
            val_loss = bce(prob, batch_labels)
            loss_list.append(val_loss.cpu().detach().numpy())
            pred_list += list(prob.cpu().detach().numpy())
            true_list += list(batch_labels.cpu().detach().numpy())

    round_list = np.around(pred_list)

    # Extract, save, and display test metrics
    avg_loss = np.mean(loss_list)
    cmatrix = metrics.confusion_matrix(true_list, round_list)
    acc = metrics.accuracy_score(true_list, round_list)
    prc = metrics.precision_score(true_list, round_list)
    rec = metrics.recall_score(true_list, round_list)
    f1 = metrics.f1_score(true_list, round_list)
    auroc = metrics.roc_auc_score(true_list, pred_list)
    (precisions, recalls, _) = metrics.precision_recall_curve(true_list, pred_list)
    auprc = metrics.auc(recalls, precisions)

    metrics_dict = {}
    metrics_dict['Test Loss'] = avg_loss
    metrics_dict['Confusion Matrix'] = cmatrix
    metrics_dict['Accuracy'] = acc
    metrics_dict['Precision'] = prc
    metrics_dict['Recall'] = rec
    metrics_dict['F1 Score'] = f1
    metrics_dict['AUROC'] = auroc
    metrics_dict['AUPRC'] = auprc

    # F1 by age group and sex (test_dataset holds the real patients)
    if args.subgroups:
        age_list = [get_age_bin(p['last_age']) for p in test_dataset]
        gender_list = [p['gender'] for p in test_dataset]
        metrics_dict['Age F1 Score'] = subgroup_f1(
            "Age", ["0-14", "15-24", "25-44", "45-64", "65-74", "75+"],
            age_list, true_list, round_list)
        metrics_dict['Gender F1 Score'] = subgroup_f1(
            "Gender", ["F", "M"], gender_list, true_list, round_list)

    print('Test Loss: ', avg_loss)
    print('Confusion Matrix: ', cmatrix)
    print('Accuracy: ', acc)
    print('Precision: ', prc)
    print('Recall: ', rec)
    print('F1 Score: ', f1)
    print('AUROC: ', auroc)
    print('AUPRC: ', auprc)
    print("\n")

    return metrics_dict


results = {}

for i in LABEL_IDX_LIST:
    print(f"===== {task_names[i]} =====")
    label_results = {}

    # Prepare balanced datasets
    val_dataset = sample_balanced(val_ehr_dataset, i, NUM_VAL_EXAMPLES)
    test_dataset = sample_balanced(test_ehr_dataset, i, NUM_TEST_EXAMPLES)
    train_dataset_real = sample_balanced(train_ehr_dataset, i, NUM_TRAIN_EXAMPLES)
    train_dataset_halo = sample_balanced(halo_ehr_dataset, i, NUM_TRAIN_EXAMPLES)

    datasets = [val_dataset, test_dataset, train_dataset_real, train_dataset_halo]
    if any(d is None for d in datasets):
        print(f"Skipping {task_names[i]}: no positive examples in one of the datasets\n")
        results[task_names[i]] = None
        continue

    # Train on real and on synthetic data, test both on real data
    for name, train_dataset in [('Real', train_dataset_real), ('HALO', train_dataset_halo)]:
        save_name = f"syn_diag_{args.target}_{name}_{i}"
        model = DiagnosisModel(config).to(device)
        train_model(model, train_dataset, val_dataset, save_name, i)

        state = torch.load(f'./save/{save_name}', map_location=device)
        model.load_state_dict(state['model'])
        label_results[name] = test_model(model, test_dataset, i)

    results[task_names[i]] = label_results


if args.subgroups:
    results_path = output_paths['synthetic_training_subgroups']
else:
    results_path = output_paths['synthetic_training']
pickle.dump(results, open(results_path, "wb"))
