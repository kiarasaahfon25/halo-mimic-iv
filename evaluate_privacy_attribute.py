import pickle
import numpy as np
from tqdm import tqdm
from collections import Counter
from config import HALOConfig
from halo_utils import parse_args, setup, target_vector, paths

# Usage: python evaluate_privacy_attribute.py --target {labels,mortality,readmission}
# Attribute inference attack (from the HALO paper): an attacker who knows a training
# patient's target and their 100 most common codes finds the closest synthetic patient
# and guesses the patient's remaining (uncommon) codes from it. The baseline uses real
# test patients instead of synthetic ones; similar F1 scores mean the synthetic data
# leaks little beyond what any real population would.

K = 1 # Number of closest patients used to guess the codes
CHUNK_SIZE = 1000 # Patients compared at once (limits memory)


# Patient as a set of target indices and a set of codes across all visits
def flatten(p, target):
    return {'labels': target_vector(p, target), 'visits': set([c for v in p['visits'] for c in v])}


# Split into what the attacker knows ('labels': target indices plus common codes,
# offset past the target) and what they try to infer ('codes': the uncommon codes)
def split_known(p, common_codes, label_vocab_size):
    return {
        'labels': set([c for c in p['labels'].nonzero()[0].tolist()] + [c + label_vocab_size for c in p['visits'] if c in common_codes]),
        'codes': set([c for c in p['visits'] if c not in common_codes])
    }


# Binary matrix of each patient's known attributes, with columns remapped to 0..n_cols-1
def known_matrix(dataset, column):
    matrix = np.zeros((len(dataset), len(column)), dtype=np.float32)
    for i, p in enumerate(dataset):
        matrix[i, [column[c] for c in p['labels']]] = 1
    return matrix


# Same result as the original HALO loop (for each patient, sort every reference patient
# by the size of the symmetric difference of known attributes and keep the first k),
# computed as |A| + |B| - 2|A & B| with matrix products instead of Python set operations.
# A stable sort keeps ties in reference order, exactly like the original list.sort.
def find_closest_all(dataset, reference_dataset, k):
    column = {c: i for i, c in enumerate(sorted(set(c for p in list(dataset) + list(reference_dataset) for c in p['labels'])))}
    query = known_matrix(dataset, column)
    reference = known_matrix(reference_dataset, column)
    query_sizes = query.sum(1)
    reference_sizes = reference.sum(1)

    closest = []
    for start in tqdm(range(0, len(query), CHUNK_SIZE)):
        q = query[start:start+CHUNK_SIZE]
        dists = query_sizes[start:start+CHUNK_SIZE, None] + reference_sizes[None, :] - 2 * (q @ reference.T)
        closest += list(np.argsort(dists, axis=1, kind='stable')[:, :k])
    return closest


def calc_attribute_risk(train_dataset, reference_dataset, k):
    tp = 0
    fp = 0
    fn = 0
    closest = find_closest_all(train_dataset, reference_dataset, k)
    for p, closest_idx in zip(train_dataset, closest):
        closest_k = [reference_dataset[j]['codes'] for j in closest_idx]
        pred_codes = set([cd for cd, cnt in Counter([c for codes in closest_k for c in codes]).items() if cnt > k/2])
        true_pos = len(pred_codes.intersection(p['codes']))
        false_pos = len(pred_codes) - true_pos
        false_neg = len(p['codes']) - true_pos
        tp += true_pos
        fp += false_pos
        fn += false_neg

    f1 = tp / (tp + (0.5 * (fp + fn)))
    return f1


if __name__ == '__main__':
    args = parse_args()
    config = HALOConfig(target=args.target)
    output_paths = paths(args.target)
    setup(seed=4)

    test_ehr_dataset = pickle.load(open('./data/testDataset.pkl', 'rb'))
    test_ehr_dataset = [flatten(p, args.target) for p in test_ehr_dataset]
    train_ehr_dataset = pickle.load(open('./data/trainDataset.pkl', 'rb'))
    train_ehr_dataset = [flatten(p, args.target) for p in train_ehr_dataset]
    train_ehr_dataset = np.random.choice(train_ehr_dataset, len(test_ehr_dataset), replace=False)
    synthetic_ehr_dataset = pickle.load(open(output_paths['synthetic'], 'rb'))
    synthetic_ehr_dataset = [flatten(p, args.target) for p in synthetic_ehr_dataset if len(p['visits']) > 0]
    synthetic_ehr_dataset = np.random.choice(synthetic_ehr_dataset, len(test_ehr_dataset), replace=False)

    common_codes = set([cd for cd, _ in Counter([c for p in train_ehr_dataset for c in p['visits']]).most_common()[0:100]])

    test_ehr_dataset = [split_known(p, common_codes, config.label_vocab_size) for p in test_ehr_dataset]
    train_ehr_dataset = [split_known(p, common_codes, config.label_vocab_size) for p in train_ehr_dataset]
    synthetic_ehr_dataset = [split_known(p, common_codes, config.label_vocab_size) for p in synthetic_ehr_dataset]

    att_risk = calc_attribute_risk(train_ehr_dataset, synthetic_ehr_dataset, K)
    baseline_risk = calc_attribute_risk(train_ehr_dataset, test_ehr_dataset, K)
    results = {
        "Attribute Attack F1 Score": att_risk,
        "Baseline Attack F1 Score": baseline_risk
    }
    pickle.dump(results, open(output_paths['privacy_attribute'], "wb"))
    print(results)
