'''
Code shared by train_model.py, test_model.py and evaluate_synthetic_training.py.

Each script is run once per prediction target, and each target gets its own HALO
model, synthetic dataset and results:
  - labels:      25 CCS phenotypes (diagnosis prediction), p['labels']
  - mortality:   in-hospital mortality, p['mortality']
  - readmission: 30-day readmission, p['readmission']
'''
import os
import random
import argparse
import torch
import numpy as np
from config import TARGET_SIZES


def parse_args(subgroups=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', choices=list(TARGET_SIZES), default='labels',
                        help='prediction target HALO is conditioned on')
    if subgroups:
        parser.add_argument('--subgroups', action='store_true',
                            help='also report F1 by age group and sex')
    return parser.parse_args()


# Set random seeds for reproducibility and return the device to run on
def setup(seed=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return device


# The patient's target as a vector (mortality and readmission are single values)
def target_vector(p, target):
    return np.atleast_1d(p[target])


# Output files for a target, creating the folders if needed
def paths(target):
    for folder in ['./save', './results/datasets', './results/testing_stats',
                   './results/synthetic_training_stats', './results/loss_curves']:
        os.makedirs(folder, exist_ok=True)
    return {
        'model': f'./save/halo_model_{target}',
        'losses': f'./save/losses_{target}.pkl',
        'loss_plot': f'./results/loss_curves/loss_{target}',
        'synthetic': f'./results/datasets/haloDataset_{target}.pkl',
        'test_intermediate': f'./results/testing_stats/HALO_intermediate_results_{target}.pkl',
        'test_metrics': f'./results/testing_stats/HALO_Metrics_{target}.pkl',
        'synthetic_training': f'./results/synthetic_training_stats/fully_synthetic_stats_{target}.pkl',
        'synthetic_training_subgroups': f'./results/synthetic_training_stats/fully_synthetic_stats_{target}_age.pkl',
    }


# Convert patient visits and the target into batched tensors for HALO
def get_batch(ehr_dataset, loc, batch_size, config):
    # EHR data saved as [P_1, P_2, ... , P_i]
    #   Where each patient P has visits [V_1, V_2, ... , V_j]
    #     Where each visit V is [C_1, C_2, ... , C_k]
    #   And a target (labels, mortality or readmission) that fills the label slot
    ehr = ehr_dataset[loc:loc+batch_size]

    label_start = config.code_vocab_size
    label_end = config.code_vocab_size + config.label_vocab_size
    batch_ehr = np.zeros((len(ehr), config.n_ctx, config.total_vocab_size))
    batch_mask = np.zeros((len(ehr), config.n_ctx, 1))
    for i, p in enumerate(ehr):
        visits = p['visits']
        for j, v in enumerate(visits):
            batch_ehr[i,j+2][v] = 1
            batch_mask[i,j+2] = 1
        batch_ehr[i,1,label_start:label_end] = target_vector(p, config.target) # Set the patient target
        batch_ehr[i,len(visits)+1,label_end+1] = 1 # Set the final visit to have the end token
        batch_ehr[i,len(visits)+2:,label_end+2] = 1 # Set the rest to the padded visit token

    batch_mask[:,1] = 1 # Set the mask to cover the labels
    batch_ehr[:,0,label_end] = 1 # Set the first visits to be the start token
    batch_mask = batch_mask[:,1:,:] # Shift the mask to match the shifted labels and predictions the model will return
    return batch_ehr, batch_mask
