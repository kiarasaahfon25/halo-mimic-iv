'''
Map the ICD-9 codes of the HCUP CCS benchmark groups to ICD-10, replicating the
HiSGT conversion method:
https://github.com/jameszhou-gl/HiSGT/blob/535240413fce1bd4f7c3a129dd857bf29ad97dc0/data/mimiciv/2.2_icd9_subset_convert_icd10/icd10_part/gen_vocab_stoi_icd10_csv.py

  1. Each ICD-9 code gets ONE ICD-10 code from the main mapping file
     (dict(zip(...)), so the last row for a code wins).
  2. If that code is missing or cannot be aligned to the ICD-10 hierarchy,
     the supplementary mapping file is used instead.

Output: ccs_benchmark_icd10_codes.csv, read by build_dataset.py and check_icd.py
'''
import json
import yaml
import pandas as pd

mapping_dir = "mappings/hisgt"
mapping_file = mapping_dir + "/icd_cm_9_to_10_mapping_from_ethos.csv"
supplementary_file = mapping_dir + "/icd_cm_9_to_10_mapping_supplementary.csv"
hierarchy_file = mapping_dir + "/icd10_hierarchical_graph.json"
yaml_path = "hcup_ccs_2015_definitions_benchmark.yaml"
output_file = "ccs_benchmark_icd10_codes.csv"

# ICD-10 codes reached from more than one benchmark group. build_dataset.py needs
# one group per code, so these are assigned following HCUP's ICD-10 CCS.
OVERRIDES = {
    "I10": "Essential hypertension",
    "J99": "Respiratory failure; insufficiency; arrest (adult)",
    "J988": "Other lower respiratory disease",
}

mapping_df = pd.read_csv(mapping_file, dtype=str)
supplementary_mapping_df = pd.read_csv(supplementary_file, dtype=str)
with open(hierarchy_file, "r") as f:
    icd10_hierarchical_graph = json.load(f)
with open(yaml_path, "r") as f:
    definitions = yaml.safe_load(f)

icd9_to_icd10 = dict(zip(mapping_df['icd_9'], mapping_df['icd_10']))


# from HiSGT gen_vocab_stoi_icd10_csv.py
def normalize_icd10_code(icd10_code, icd10_hierarchy):
    """Normalize ICD-10 code by truncating or finding the nearest match."""
    if icd10_code in icd10_hierarchy:
        return icd10_code  # Code already exists
    if '.' not in icd10_code:
        icd10_code = icd10_code[:3] + '.' + icd10_code[3:]
    # Truncate trailing characters (e.g., "S065X9A" -> "S06.5X")
    flag = False
    search_code = None
    for length in range(3, len(icd10_code)+1):
        if length == 4:
            # Skip the fourth character, e.g., 'S06.'
            continue
        truncated_code = icd10_code[:length]
        if truncated_code in icd10_hierarchy:
            search_code = truncated_code
            flag = True
        else:
            if flag:
                break
    # Return None if no match is found
    return search_code


rows = []
unmapped = []
for group, info in definitions.items():
    if not info['use_in_benchmark']:
        continue
    for icd9_code in info['codes']:
        icd10_code = icd9_to_icd10.get(icd9_code, None)
        if not icd10_code:
            icd10_code = 'UNKNOWN'
        source = 'main'

        # Same fallback as HiSGT: codes that cannot be aligned to the hierarchy
        # are replaced by the supplementary mapping
        if normalize_icd10_code(icd10_code, icd10_hierarchical_graph) is None:
            supp_icd_10_code = supplementary_mapping_df[
                supplementary_mapping_df['icd_9'] == icd9_code]['icd_10'].values
            if len(supp_icd_10_code) == 0:
                # HiSGT raises an error here; these are non-billable ICD-9
                # category codes, so they are skipped instead
                unmapped.append((group, icd9_code))
                continue
            icd10_code = supp_icd_10_code[0]
            source = 'supplementary'

        rows.append({
            'group': group,
            'icd9_code': icd9_code,
            'icd10_code': icd10_code.replace('.', ''),  # MIMIC-IV codes have no dots
            'source': source,
        })

results_df = pd.DataFrame(rows)

# Resolve ICD-10 codes claimed by more than one group
for icd10_code, group in OVERRIDES.items():
    results_df.loc[results_df['icd10_code'] == icd10_code, 'group'] = group

groups_per_code = results_df.groupby('icd10_code')['group'].nunique()
conflicts = groups_per_code[groups_per_code > 1].index.tolist()
assert not conflicts, f"ICD-10 codes mapped to more than one group, add to OVERRIDES: {conflicts}"

results_df.to_csv(output_file, index=False)

print(f"Mapped from main file: {(results_df['source'] == 'main').sum()}")
print(f"Mapped from supplementary file: {(results_df['source'] == 'supplementary').sum()}")
print(f"No mapping in either file (skipped): {len(unmapped)}")
for group, icd9_code in unmapped:
    print(f"  {icd9_code} ({group})")
print(f"Unique ICD-10 codes: {results_df['icd10_code'].nunique()}")
print("\nICD-10 codes per group:")
print(results_df.groupby('group')['icd10_code'].nunique().to_string())
print(f"\nSaved to {output_file}")
