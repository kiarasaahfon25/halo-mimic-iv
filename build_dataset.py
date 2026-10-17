import yaml
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os

MAX_VISITS = 46 

# Check that required files and folders exist before building the dataset
required_files = [
    "hcup_ccs_2015_definitions_benchmark.yaml",
    "ccs_benchmark_icd10_codes.csv",
]

for file in required_files:
    if not os.path.isfile(file):
        raise FileNotFoundError(f"Required file not found: {file}")

# Check that the output directory exists
if not os.path.isdir("./data"):
    raise FileNotFoundError(
        "Output directory './data' does not exist. Create it before running."
    )

print("All required files and output directories exist.")


#retrieving mimic data
mimic_dir = "/fs02/datasets/MIMIC-IV/physionet.org/files/mimiciv/3.1/hosp"

admissionFile = mimic_dir + "/admissions.csv.gz"
diagnosisFile = mimic_dir + "/diagnoses_icd.csv.gz"
patientFile = mimic_dir + "/patients.csv.gz"


print("Loading CSVs Into Dataframes")
#admissions
admissionDf = pd.read_csv(admissionFile, dtype=str)
admissionDf['admittime'] = pd.to_datetime(admissionDf['admittime']) #covert column into python datetime
admissionDf['dischtime'] = pd.to_datetime(admissionDf['dischtime']) 
admissionDf = admissionDf.sort_values('admittime') #oldest to newest
admissionDf = admissionDf.reset_index(drop=True) 

#diagnosis 
diagnosisDf = pd.read_csv(diagnosisFile, dtype=str).set_index("hadm_id")
diagnosisDf = diagnosisDf.sort_index()

diagnosisDf = diagnosisDf[
   diagnosisDf['icd_code'].notnull() & 
   diagnosisDf['icd_version'].notnull()
   ].copy() #remove diagnoses without an ICD_Code

# Keep the original code without prefix
diagnosisDf['raw_icd_code'] = diagnosisDf['icd_code']

# Keep both ICD-9 and ICD-10 codes
diagnosisDf['icd_code'] = (
    'ICD' + diagnosisDf['icd_version'] + ':' + diagnosisDf['icd_code']
)

diagnosisDf = diagnosisDf[['raw_icd_code', 'icd_code', 'icd_version']]

#patients
patientDf = pd.read_csv(patientFile, dtype=str)
#patientDf['dob'] = pd.to_datetime(patientDf['dob']) #convert date of birth into python datetime
patientDf = patientDf.set_index('subject_id')

#calculate patient age
def calculate_age(anchor_age, anchor_year, admittime):
    age = int(anchor_age) + (admittime.year - int(anchor_year))
    return min(age,91)
'''
#limiting records for easier run on laptop
admissionDf = admissionDf.head()

diagnosisDf = diagnosisDf[
    diagnosisDf.index.isin(admissionDf['hadm_id'])
]'''

print("Building Dataset")
data = {}
#tqdm adds a progression bar
for row in tqdm(admissionDf.itertuples(), total=admissionDf.shape[0]):          
    #Extracting Admissions Table Info
    hadm_id = row.hadm_id
    subject_id = row.subject_id
            
    # Extracting the Diagnoses
    if hadm_id in diagnosisDf.index: 
        diagnoses = list(set(diagnosisDf.loc[[hadm_id]]["icd_code"])) #get all  diagnosis code for this hospital admission
    else:
        diagnoses = []
    
    # Building the hospital admission data point
    if subject_id not in data:
      data[subject_id] = {
         'visits': [diagnoses],#create initial visit (new patient)
         'admittimes': [row.admittime],
         'dischtimes': [row.dischtime],
         'hospital_expire_flags': [row.hospital_expire_flag]
      }
    else:
      data[subject_id]['visits'].append(diagnoses) #add subsequent visits (same patient)
      data[subject_id]['admittimes'].append(row.admittime)
      data[subject_id]['dischtimes'].append(row.dischtime)
      data[subject_id]['hospital_expire_flags'].append(row.hospital_expire_flag)

# Calculate age at the patient's last recorded admission, mortality and readmission
num_negative_gaps = 0 # overlapping admissions (next admittime before previous dischtime)
for subject_id, patient in data.items():
    anchor_age = patientDf.loc[subject_id, 'anchor_age']
    anchor_year = patientDf.loc[subject_id, 'anchor_year']

    last_admittime = patient['admittimes'][-1]

    patient['last_age'] = calculate_age(
        anchor_age, anchor_year, last_admittime
    )
    patient['gender'] = patientDf.loc[subject_id, 'gender']

    # In-hospital mortality: based on hospital_expire_flags
    patient['mortality'] = int(
    any(flag == '1' for flag in patient['hospital_expire_flags'])
    )

    # 30-day readmission: next admission less than 30 days after the previous discharge
    gaps = [
        patient['admittimes'][k + 1] - patient['dischtimes'][k]
        for k in range(len(patient['admittimes']) - 1)
    ]

    #double check if some hospital admissions for the same patient overlap!!
    patient['readmission'] = int(any(gap < pd.Timedelta(days=30) for gap in gaps))
    num_negative_gaps += sum(gap < pd.Timedelta(0) for gap in gaps)

    del patient['admittimes']
    del patient['dischtimes']
    del patient['hospital_expire_flags']


    if len(patient['visits']) > MAX_VISITS:
       patient['visits'] = patient['visits'][-MAX_VISITS:]

    
data = list(data.values()) #removes subject ids 

code_to_index = {} #save code as index

#all_codes = list(set([c for p in data.values() for v in p['visits'] for c in v])) commenting out for readability

all_codes = []

for patient in data:
    for visit in patient['visits']:
        for code in visit:
            all_codes.append(code)

all_codes = sorted(set(all_codes)) #remove duplicate codes and turn into a list

np.random.seed(4)
np.random.shuffle(all_codes)
for c in all_codes:
    code_to_index[c] = len(code_to_index) #give each code a number
print(f"VOCAB SIZE: {len(code_to_index)}")

#reverse dictionary to show index, codex
index_to_code = {v: k for k, v in code_to_index.items()}

print("Adding Labels")
with open("hcup_ccs_2015_definitions_benchmark.yaml") as definitions_file:
    definitions = yaml.full_load(definitions_file)

code_to_group = {} # Associates code to a group
for group in definitions:
  if definitions[group]['use_in_benchmark'] == False:
      continue
  codes = ['ICD9:' + code for code in definitions[group]['codes']]
  for code in codes:
      if code not in code_to_group:
        code_to_group[code] = group
      else:
        assert code_to_group[code] == group #checks for  conflicts

# ICD-10 codes for the benchmark groups (generated by convert_benchmark_to_icd10.py)
icd10_benchmark_df = pd.read_csv("ccs_benchmark_icd10_codes.csv", dtype=str)
for group, icd10_code in zip(icd10_benchmark_df['group'], icd10_benchmark_df['icd10_code']):
  code = 'ICD10:' + icd10_code
  if code not in code_to_group:
    code_to_group[code] = group
  else:
    assert code_to_group[code] == group #checks for  conflicts. If the code already exists verify that it belongs to the same group

#creates sorted list of bench mark groups, position in the list is their id
id_to_group = sorted([
  k for k in definitions.keys() 
  if definitions[k]['use_in_benchmark'] ==  True]
  )
 
#create dictionary with benchmark group: id 
group_to_id = dict((x, i) for (i, x) in enumerate(id_to_group))

# Add Labels
for p in data:
  label = np.zeros(len(group_to_id))
  for v in p['visits']:
    for c in v:
      # Codes are prefixed with their ICD version, e.g. "ICD9:4019" or "ICD10:I10"
      if c in code_to_group:
        label[group_to_id[code_to_group[c]]] = 1
  
  p['labels'] = label

#covert visits to no longer use ICD_Code but numberical codes
print("Converting Visits")
for p in data:
    new_visits = []
    for v in p['visits']:
        new_visit = []
        for c in v:
            new_visit.append(code_to_index[c])
                
        new_visits.append((list(set(new_visit)))) #removes duplicates (codes that appear more than once in the same hospital visit)
        
    p['visits'] = new_visits 

# Verify that no patient exceeds the visit limit
assert max(len(p['visits']) for p in data) <= MAX_VISITS, \
    "Some patient histories still exceed 46 visits!"

print("All patient histories fit within the 46-visit limit.")  

print(f"MAX LEN: {max([len(p['visits']) for p in data])}")
print(f"AVG LEN: {np.mean([len(p['visits']) for p in data])}")
print(f"MAX VISIT LEN: {max([len(v) for p in data for v in p['visits']])}")
print(f"AVG VISIT LEN: {np.mean([len(v) for p in data for v in p['visits']])}")
print(f"NUM RECORDS: {len(data)}")
print(f"NUM LONGITUDINAL RECORDS: {len([p for p in data if len(p['visits']) > 1])}")


# Initial data diagnostic
print("\nInitial Diagnostic")

num_patients = len(data)
num_visits = [len(p["visits"]) for p in data]
total_admissions = sum(num_visits)

print(f"Number of patients: {num_patients}")
print(f"Total admissions: {total_admissions}")

if num_visits:
    print(f"Median visits per patient: {np.median(num_visits):.2f}")
    print(f"P90 visits per patient: {np.percentile(num_visits, 90):.2f}")
    print(f"P95 visits per patient: {np.percentile(num_visits, 95):.2f}")
    print(f"P99 visits per patient: {np.percentile(num_visits, 99):.2f}")
    print(f"Maximum visits per patient: {max(num_visits)}")

    print(f"Patients with multiple admissions: "
          f"{sum(n > 1 for n in num_visits)}")

    # HALO uses n_ctx=48, including special tokens.
    # Two positions are reserved for special tokens, leaving 46 visits.
    max_visits = MAX_VISITS
    exceeding_context = sum(n > max_visits for n in num_visits)

    print(f"Patients exceeding {max_visits} visits: {exceeding_context}")
    print(f"Percentage exceeding visit context: "
          f"{100 * exceeding_context / num_patients:.2f}%")

# Codes per visit
codes_per_visit = [
    len(visit)
    for p in data
    for visit in p["visits"]
]

if codes_per_visit:
    print(f"\nMean codes per visit: {np.mean(codes_per_visit):.2f}")
    print(f"Median codes per visit: {np.median(codes_per_visit):.2f}")
    print(f"P95 codes per visit: {np.percentile(codes_per_visit, 95):.2f}")
    print(f"Maximum codes per visit: {max(codes_per_visit)}")

# Diagnosis vocabulary
print(f"\nVocabulary size: {len(code_to_index)}")

# Total diagnosis entries and unique codes
all_diagnoses = [
    code for p in data
    for visit in p["visits"]
    for code in visit
]

print(f"Total diagnosis entries: {len(all_diagnoses)}")
print(f"Unique diagnosis codes in data: {len(set(all_diagnoses))}")

print(f"Patients with no diagnoses: "
      f"{sum(not any(p['visits']) for p in data)}")


# Sample counts for each subgroup
print("\nSubgroup Counts")

# Sex
print("\nSex:")
print(pd.Series([p['gender'] for p in data])
      .value_counts(dropna=False))

# Age groups
print("\nAge groups:")
age_bins = [18, 40, 65, 80, float('inf')]
age_labels = ['18-39', '40-64', '65-79', '80+']

ages = pd.Series([p['last_age'] for p in data])

age_groups = pd.cut(
    ages,
    bins=age_bins,
    labels=age_labels,
    right=False
)

print(age_groups.value_counts(sort=False, dropna=False))

# Outcomes
print("\nOutcomes:")
num_mortality = sum(p['mortality'] for p in data)
num_readmission = sum(p['readmission'] for p in data)
print(f"In-hospital mortality: {num_mortality} "
      f"({100 * num_mortality / len(data):.2f}% of patients)")
print(f"30-day readmission: {num_readmission} "
      f"({100 * num_readmission / len(data):.2f}% of patients)")
print(f"Overlapping admissions (negative gap, counted as readmission): "
      f"{num_negative_gaps}")


# Train-Val-Test Split
print("Splitting Datasets")
train_dataset, test_dataset = train_test_split(data, test_size=0.2, random_state=4, shuffle=True) # 80 20 
train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.1, random_state=4, shuffle=True) #90 10 




# Save Everything
print("Saving Everything")
print(len(index_to_code))
pickle.dump(code_to_index, open("./data/codeToIndex.pkl", "wb"))
pickle.dump(index_to_code, open("./data/indexToCode.pkl", "wb"))
pickle.dump(id_to_group, open("./data/idToLabel.pkl", "wb"))
pickle.dump(train_dataset, open("./data/trainDataset.pkl", "wb"))
pickle.dump(val_dataset, open("./data/valDataset.pkl", "wb"))
pickle.dump(test_dataset, open("./data/testDataset.pkl", "wb"))
