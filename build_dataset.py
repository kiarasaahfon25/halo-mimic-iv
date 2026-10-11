import yaml
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

mimic_dir = "/fs02/datasets/MIMIC-III/physionet.org/files/mimic3-carevue/1.4/"
admissionFile = mimic_dir + "ADMISSIONS.csv.gz"
diagnosisFile = mimic_dir + "DIAGNOSES_ICD.csv.gz"
patientFile = mimic_dir + "PATIENTS.csv.gz"

print("Loading CSVs Into Dataframes")
#admissions
admissionDf = pd.read_csv(admissionFile, dtype=str)
admissionDf['admittime'] = pd.to_datetime(admissionDf['admittime']) #covert column into python datetime
admissionDf = admissionDf.sort_values('admittime') #oldest to newest
admissionDf = admissionDf.reset_index(drop=True) 

#diagnosis 
diagnosisDf = pd.read_csv(diagnosisFile, dtype=str).set_index("hadm_id")
diagnosisDf = diagnosisDf[diagnosisDf['icd9_code'].notnull()] #remove diagnoses without an ICD9_Code
diagnosisDf = diagnosisDf[['icd9_code']]

#patients
patientDf = pd.read_csv(patientFile, dtype=str)
patientDf['dob'] = pd.to_datetime(patientDf['dob']) #convert date of birth into python datetime
patientDf = patientDf.set_index('subject_id')

#calculate patient age
def calculate_age(dob, admittime):
    age = admittime.year - dob.year

    return age
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
        diagnoses = list(set(diagnosisDf.loc[[hadm_id]]["icd9_code"])) #get all ICD9 diagnosis code for this hospital admission
    else:
        diagnoses = []
    
    # Building the hospital admission data point
    if subject_id not in data:
      data[subject_id] = {
         'visits': [diagnoses],
         'admittimes': [row.admittime]} #create initial visit (new patient)
    else:
      data[subject_id]['visits'].append(diagnoses) #add subsequent visits (same patient)
      data[subject_id]['admittimes'].append(row.admittime)

code_to_index = {} #dictionary

#all_codes = list(set([c for p in data.values() for v in p['visits'] for c in v])) commenting out for readability

all_codes = []

for patient in data.values():
    for visit in patient['visits']:
        for code in visit:
            all_codes.append(code)

all_codes = list(set(all_codes)) #remove duplicate codes and turn into a list

np.random.shuffle(all_codes)
for c in all_codes:
    code_to_index[c] = len(code_to_index) #give each code a number
print(f"VOCAB SIZE: {len(code_to_index)}")

#reverse dictionary to show index, codex
index_to_code = {v: k for k, v in code_to_index.items()}

# Calculate age at the patient's last recorded admission
for subject_id, patient in data.items():
    dob = patientDf.loc[subject_id, 'dob']
    last_admittime = patient['admittimes'][-1] #since admissions are sorted from oldest to newest
    patient['last_age'] = calculate_age(dob, last_admittime)
    patient['gender'] = patientDf.loc[subject_id, 'gender']


    del patient['admittimes']


data = list(data.values()) #removes subject ids 

print("Adding Labels")
with open("hcup_ccs_2015_definitions_benchmark.yaml") as definitions_file:
    definitions = yaml.full_load(definitions_file)

code_to_group = {} # Associates code to a group
for group in definitions:
  if definitions[group]['use_in_benchmark'] == False:
      continue
  codes = definitions[group]['codes']
  for code in codes:
      if code not in code_to_group:
        code_to_group[code] = group
      else:
        assert code_to_group[code] == group #checks for  conflicts 

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
      if c in code_to_group:
        label[group_to_id[code_to_group[c]]] = 1
  
  p['labels'] = label

#covert visits to no longer use ICD9_Code but numberical codes
print("Converting Visits")
for p in data:
    new_visits = []
    for v in p['visits']:
        new_visit = []
        for c in v:
            new_visit.append(code_to_index[c])
                
        new_visits.append((list(set(new_visit)))) #removes duplicates (codes that appear more than once in the same hospital visit)
        
    p['visits'] = new_visits    

print(f"MAX LEN: {max([len(p['visits']) for p in data])}")
print(f"AVG LEN: {np.mean([len(p['visits']) for p in data])}")
print(f"MAX VISIT LEN: {max([len(v) for p in data for v in p['visits']])}")
print(f"AVG VISIT LEN: {np.mean([len(v) for p in data for v in p['visits']])}")
print(f"NUM RECORDS: {len(data)}")
print(f"NUM LONGITUDINAL RECORDS: {len([p for p in data if len(p['visits']) > 1])}")

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
