import pickle
import random
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from scipy import sparse
from config import HALOConfig
from halo_utils import parse_args, setup, target_vector, paths

# Usage: python evaluate_privacy_membership.py --target {labels,mortality,readmission}
# Membership inference attack (from the HALO paper): for an equal mix of real training
# patients (members) and real test patients (non-members), find the Hamming distance to
# the closest synthetic patient and guess "member" when it is below the median.
# An accuracy near 0.5 means the attacker can't tell who was in HALO's training data.

NUM_TEST_EXAMPLES = 7500 # Patients sampled from each of the training and test sets
CHUNK_SIZE = 256 # Attack patients compared at once (limits memory)


# Sparse matrix with one row per patient and a column per (visit position, code),
# so that a matrix product counts the codes two patients share at the same visit position
def visit_matrix(dataset, config):
    rows, cols = [], []
    for i, p in enumerate(dataset):
        for j, v in enumerate(p['visits']):
            for c in set(v):
                rows.append(i)
                cols.append(j * config.code_vocab_size + c)
    return sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(dataset), config.n_ctx * config.code_vocab_size))


# Same result as the original HALO find_hamming loop, for every attack patient at once:
#   1 if the number of visits differs
#   + number of differing target entries
#   + for each of the attack patient's visits i: |v_i| + |s_i| - 2|v_i & s_i|
#     (s_i is empty when the synthetic patient has fewer visits)
# minimised over the synthetic patients, using sparse matrix products instead of Python loops.
def find_hamming_all(attack_dataset, synthetic_dataset, config):
    attack_lens = np.array([len(p['visits']) for p in attack_dataset])
    synthetic_lens = np.array([len(p['visits']) for p in synthetic_dataset])

    # Codes per attack patient, and per synthetic patient within its first t visits
    attack_sizes = np.array([sum(len(v) for v in p['visits']) for p in attack_dataset], dtype=np.float32)
    synthetic_cum_sizes = np.zeros((len(synthetic_dataset), config.n_ctx + 1), dtype=np.float32)
    for i, p in enumerate(synthetic_dataset):
        sizes = [len(v) for v in p['visits']]
        synthetic_cum_sizes[i, 1:len(sizes)+1] = np.cumsum(sizes)
        synthetic_cum_sizes[i, len(sizes)+1:] = synthetic_cum_sizes[i, len(sizes)]

    attack_targets = np.array([target_vector(p, config.target) for p in attack_dataset], dtype=np.float32)
    synthetic_targets = np.array([target_vector(p, config.target) for p in synthetic_dataset], dtype=np.float32)

    attack_visits = visit_matrix(attack_dataset, config)
    synthetic_visits_T = visit_matrix(synthetic_dataset, config).T.tocsr()

    min_dists = []
    for start in tqdm(range(0, len(attack_dataset), CHUNK_SIZE)):
        end = start + CHUNK_SIZE
        lens = attack_lens[start:end]
        shared = (attack_visits[start:end] @ synthetic_visits_T).toarray()
        target_dists = (attack_targets[start:end].sum(1)[:, None] + synthetic_targets.sum(1)[None, :]
                        - 2 * (attack_targets[start:end] @ synthetic_targets.T))
        dists = ((lens[:, None] != synthetic_lens[None, :]).astype(np.float32)
                 + target_dists
                 + attack_sizes[start:end, None]
                 + synthetic_cum_sizes[:, lens].T
                 - 2 * shared)
        min_dists += list(dists.min(1))
    return np.array(min_dists)


if __name__ == '__main__':
    args = parse_args()
    config = HALOConfig(target=args.target)
    output_paths = paths(args.target)
    setup(seed=4)

    train_ehr_dataset = pickle.load(open('./data/trainDataset.pkl', 'rb'))
    test_ehr_dataset = pickle.load(open('./data/testDataset.pkl', 'rb'))
    synthetic_ehr_dataset = pickle.load(open(output_paths['synthetic'], 'rb'))
    synthetic_ehr_dataset = [p for p in synthetic_ehr_dataset if len(p['visits']) > 0]

    # Members (label 1) from the training set, non-members (label 0) from the test set
    attack_dataset = ([(p, 1) for p in random.sample(train_ehr_dataset, NUM_TEST_EXAMPLES)]
                      + [(p, 0) for p in random.sample(test_ehr_dataset, NUM_TEST_EXAMPLES)])
    np.random.shuffle(attack_dataset)

    # Perform the Hamming Distance experiment
    dists = find_hamming_all([p for (p, l) in attack_dataset], synthetic_ehr_dataset, config)
    median_dist = np.median(dists)
    preds = [1 if d < median_dist else 0 for d in dists]
    labels = [l for (p, l) in attack_dataset]
    results = {
        "Accuracy": metrics.accuracy_score(labels, preds),
        "Precision": metrics.precision_score(labels, preds),
        "Recall": metrics.recall_score(labels, preds),
        "F1": metrics.f1_score(labels, preds)
    }
    pickle.dump(results, open(output_paths['privacy_membership'], "wb"))
    print(results)
