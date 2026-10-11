import torch
import pickle
import random
import itertools
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from sklearn import metrics
import matplotlib.pyplot as plt
from config import HALOConfig
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

SEED = 4
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
LR = 0.001
EPOCHS = 25
LABEL_IDX_LIST = list(range(25))
BATCH_SIZE = 512
LSTM_HIDDEN_DIM = 32
EMBEDDING_DIM = 64
NUM_TRAIN_EXAMPLES = 5000
NUM_TEST_EXAMPLES = 1000
NUM_VAL_EXAMPLES = 500

local_rank = -1
fp16 = False
if local_rank == -1:
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  n_gpu = torch.cuda.device_count()
else:
  torch.cuda.set_device(local_rank)
  device = torch.device("cuda", local_rank)
  n_gpu = 1
  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
  torch.distributed.init_process_group(backend='nccl')
if torch.cuda.is_available():
  torch.cuda.manual_seed_all(SEED)

# Add the labels to the index_to_code mapping
index_to_code = pickle.load(open("./data/idToLabel.pkl", "rb"))

config = HALOConfig()
train_ehr_dataset = pickle.load(open('./data/trainDataset.pkl', 'rb'))
val_ehr_dataset = pickle.load(open('./data/valDataset.pkl', 'rb'))
test_ehr_dataset = pickle.load(open('./data/testDataset.pkl', 'rb'))
halo_ehr_dataset = pickle.load(open('./results/datasets/haloDataset.pkl', 'rb'))
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


def get_batch(ehr_dataset, loc, batch_size, label_idx):
    ehr = ehr_dataset[loc:loc+batch_size]
    batch_ehr = np.zeros((len(ehr), config.n_ctx, config.code_vocab_size))
    batch_labels = np.array([p['labels'][label_idx] for p in ehr])
    batch_lens = np.zeros(len(ehr))

    for i, p in enumerate(ehr):
        visits = p['visits']
        batch_lens[i] = len(visits)

        for j, v in enumerate(visits):
            batch_ehr[i,j][v] = 1

    return batch_ehr, batch_labels, batch_lens


def train_model(model, train_dataset, val_dataset, save_name, label_idx):
    global_loss = 1e10
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    bce = nn.BCELoss()

    for e in range(EPOCHS):
        np.random.shuffle(train_dataset)
        train_losses = []

        for i in range(0, len(train_dataset), BATCH_SIZE):
            model.train()

            batch_ehr, batch_labels, batch_lens = get_batch(
                train_dataset,
                i,
                BATCH_SIZE,
                label_idx
            )

            batch_ehr = torch.tensor(
                batch_ehr,
                dtype=torch.float32
            ).to(device)

            batch_labels = torch.tensor(
                batch_labels,
                dtype=torch.float32
            ).to(device)

            optimizer.zero_grad()

            prob = model(
                batch_ehr,
                batch_lens
            )

            loss = bce(
                prob,
                batch_labels
            )

            train_losses.append(
                loss.cpu().detach().numpy()
            )

            loss.backward()
            optimizer.step()

        cur_train_loss = np.mean(train_losses)

        print(
            "Epoch %d Training Loss:%.5f"
            % (e, cur_train_loss)
        )

        model.eval()

        with torch.no_grad():

            val_losses = []

            for v_i in range(
                0,
                len(val_dataset),
                BATCH_SIZE
            ):

                batch_ehr, batch_labels, batch_lens = get_batch(
                    val_dataset,
                    v_i,
                    BATCH_SIZE,
                    label_idx
                )

                batch_ehr = torch.tensor(
                    batch_ehr,
                    dtype=torch.float32
                ).to(device)

                batch_labels = torch.tensor(
                    batch_labels,
                    dtype=torch.float32
                ).to(device)

                prob = model(
                    batch_ehr,
                    batch_lens
                )

                val_loss = bce(
                    prob,
                    batch_labels
                )

                val_losses.append(
                    val_loss.cpu().detach().numpy()
                )

            cur_val_loss = np.mean(val_losses)

            print(
                "Epoch %d Validation Loss:%.5f"
                % (e, cur_val_loss)
            )

            if cur_val_loss < global_loss:

                global_loss = cur_val_loss

                state = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }

                torch.save(
                    state,
                    f'./save/{save_name}'
                )

                print(
                    '------------ Save best model ------------'
                )

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
    
def test_model(model, test_dataset, label_idx):

    loss_list = []
    pred_list = []
    true_list = []
    age_list = []
    gender_list = []

    bce = nn.BCELoss()

    model.eval()

    with torch.no_grad():

        for i in range(
            0,
            len(test_dataset),
            BATCH_SIZE
        ):

            batch_ehr, batch_labels, batch_lens = get_batch(
                test_dataset,
                i,
                BATCH_SIZE,
                label_idx
            )

            batch_ehr = torch.tensor(
                batch_ehr,
                dtype=torch.float32
            ).to(device)

            batch_labels = torch.tensor(
                batch_labels,
                dtype=torch.float32
            ).to(device)

            prob = model(
                batch_ehr,
                batch_lens
            )

            val_loss = bce(
                prob,
                batch_labels
            )

            loss_list.append(
                val_loss.cpu().detach().numpy()
            )

            pred_list += list(
                prob.cpu().detach().numpy()
            )

            true_list += list(
                batch_labels.cpu().detach().numpy()
            )
            age_list += [p['last_age'] for p in test_dataset[i:i+BATCH_SIZE]]
            gender_list += [p['gender'] for p in test_dataset[i:i+BATCH_SIZE]]
   


    round_list = np.around(pred_list)

    # Extract, save, and display test metrics
    avg_loss = np.mean(loss_list)

    cmatrix = metrics.confusion_matrix(
        true_list,
        round_list
    )

    acc = metrics.accuracy_score(
        true_list,
        round_list
    )

    prc = metrics.precision_score(
        true_list,
        round_list
    )

    rec = metrics.recall_score(
        true_list,
        round_list
    )

    f1 = metrics.f1_score(
        true_list,
        round_list
    )
    # Calculate F1 score for each age group
    age_bins = {
        "0-14": [],
        "15-24": [],
        "25-44": [],
        "45-64": [],
        "65-74": [],
        "75+": []
    }

    for age, true, pred in zip(age_list, true_list, round_list):
        age_bin = get_age_bin(age)

        if age_bin is not None:
            age_bins[age_bin].append((true, pred))

    age_f1 = {}

    for age_bin, patients in age_bins.items():

        if len(patients) == 0:
            age_f1[age_bin] = None
            continue

        age_true = [x[0] for x in patients]
        age_pred = [x[1] for x in patients]

        

        age_f1[age_bin] = metrics.f1_score(
            age_true,
            age_pred,
            zero_division=0
        )

        print(
            f"Age {age_bin} - N: {len(patients)}, "
            f"Actual positives: {sum(age_true)}, "
            f"Predicted positives: {sum(age_pred)}, "
            f"F1: {age_f1[age_bin]:.4f}"
        )
    gender_groups = {
    "F": [],
    "M": []
    }
    
    for gender, true, pred in zip(gender_list, true_list, round_list):
        if gender in gender_groups:
            gender_groups[gender].append((true, pred))

    gender_f1 = {}

    for gender, patients in gender_groups.items():
        if len(patients) == 0:
            gender_f1[gender] = None
            continue

        gender_true = [x[0] for x in patients]
        gender_pred = [x[1] for x in patients]

        gender_f1[gender] = metrics.f1_score(
            gender_true,
            gender_pred,
            zero_division=0
        )

        print(
            f"Gender {gender} - N: {len(patients)}, "
            f"Actual positives: {sum(gender_true)}, "
            f"Predicted positives: {sum(gender_pred)}, "
            f"F1: {gender_f1[gender]:.4f}"
        )      

    auroc = metrics.roc_auc_score(
        true_list,
        pred_list
    )

    (precisions, recalls, _) = metrics.precision_recall_curve(
        true_list,
        pred_list
    )

    auprc = metrics.auc(
        recalls,
        precisions
    )

    metrics_dict = {}

    metrics_dict['Test Loss'] = avg_loss
    metrics_dict['Confusion Matrix'] = cmatrix
    metrics_dict['Accuracy'] = acc
    metrics_dict['Precision'] = prc
    metrics_dict['Recall'] = rec
    metrics_dict['F1 Score'] = f1
    metrics_dict['AUROC'] = auroc
    metrics_dict['AUPRC'] = auprc
    metrics_dict['Age F1 Score'] = age_f1
    metrics_dict['Gender F1 Score'] = gender_f1



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

    label_results = {}

    # Prepare datasets
    halo_pos_label_dataset = [
        p for p in halo_ehr_dataset
        if p['labels'][i] == 1
    ]

    halo_neg_label_dataset = [
        p for p in halo_ehr_dataset
        if p['labels'][i] == 0
    ]

    train_pos_label_dataset = [
        p for p in train_ehr_dataset
        if p['labels'][i] == 1
    ]

    train_neg_label_dataset = [
        p for p in train_ehr_dataset
        if p['labels'][i] == 0
    ]

    val_pos_label_dataset = [
        p for p in val_ehr_dataset
        if p['labels'][i] == 1
    ]

    val_neg_label_dataset = [
        p for p in val_ehr_dataset
        if p['labels'][i] == 0
    ]

    test_pos_label_dataset = [
        p for p in test_ehr_dataset
        if p['labels'][i] == 1
    ]

    test_neg_label_dataset = [
        p for p in test_ehr_dataset
        if p['labels'][i] == 0
    ]

    val_dataset = list(
        np.random.choice(
            val_pos_label_dataset,
            int(NUM_VAL_EXAMPLES/2),
            replace=(
                False
                if len(val_pos_label_dataset) >= NUM_VAL_EXAMPLES
                else True
            )
        )
    ) + list(
        np.random.choice(
            val_neg_label_dataset,
            int(NUM_VAL_EXAMPLES/2),
            replace=False
        )
    )

    test_dataset = list(
        np.random.choice(
            test_pos_label_dataset,
            int(NUM_TEST_EXAMPLES/2),
            replace=(
                False
                if len(test_pos_label_dataset) >= NUM_TEST_EXAMPLES
                else True
            )
        )
    ) + list(
        np.random.choice(
            test_neg_label_dataset,
            int(NUM_TEST_EXAMPLES/2),
            replace=False
        )
    )

    train_dataset_real = list(
        np.random.choice(
            train_pos_label_dataset,
            int(NUM_TRAIN_EXAMPLES/2),
            replace=(
                False
                if len(test_pos_label_dataset) >= int(NUM_TRAIN_EXAMPLES/2)
                else True
            )
        )
    ) + list(
        np.random.choice(
            train_neg_label_dataset,
            int(NUM_TRAIN_EXAMPLES/2),
            replace=False
        )
    )

    train_dataset_halo = list(
        np.random.choice(
            halo_pos_label_dataset,
            int(NUM_TRAIN_EXAMPLES/2),
            replace=(
                False
                if len(halo_pos_label_dataset) >= int(NUM_TRAIN_EXAMPLES/2)
                else True
            )
        )
    ) + list(
        np.random.choice(
            halo_neg_label_dataset,
            int(NUM_TRAIN_EXAMPLES/2),
            replace=False
        )
    )

    # Perform the different experiments

    model_real = DiagnosisModel(config).to(device)

    train_model(
        model_real,
        train_dataset_real,
        val_dataset,
        f"syn_diag_Real_{i}",
        i
    )

    state = torch.load(
        f'./save/syn_diag_Real_{i}'
    )

    model_real.load_state_dict(
        state['model']
    )

    test_results_real = test_model(
        model_real,
        test_dataset,
        i
    )

    label_results[f'Real'] = test_results_real


    model_halo = DiagnosisModel(config).to(device)

    train_model(
        model_halo,
        train_dataset_halo,
        val_dataset,
        f"syn_diag_HALO_{i}",
        i
    )

    state = torch.load(
        f'./save/syn_diag_HALO_{i}'
    )

    model_halo.load_state_dict(
        state['model']
    )

    test_results_halo = test_model(
        model_halo,
        test_dataset,
        i
    )

    label_results[f'HALO'] = test_results_halo


    results[index_to_code[i]] = label_results


pickle.dump(
    results,
    open(
        f"results/synthetic_training_stats/fully_synthetic_stats_age.pkl",
        "wb"
    )
)