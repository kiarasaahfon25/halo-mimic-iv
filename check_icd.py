import pandas as pd
import yaml

diagnosisFile = "diagnoses_icd-2.csv"
yaml_path = "hcup_ccs_2015_definitions_benchmark.yaml"


# Load YAML definitions
with open(yaml_path, "r") as f:
    definitions = yaml.safe_load(f)

# Load only the columns we need
diagnoses = pd.read_csv(
    diagnosisFile,
    usecols=["subject_id", "hadm_id", "icd_code", "icd_version"],
    dtype=str
)

# Standardize codes (remove decimal points and whitespace)
diagnoses["icd_code"] = (
    diagnoses["icd_code"].str.strip().str.replace(".", "", regex=False).str.upper()
)

# These YAML definitions contain ICD-9 codes
diagnoses_icd9 = diagnoses[diagnoses["icd_version"] == "9"]

results = []

for category, info in definitions.items():
    if not info.get("use_in_benchmark", False):
        continue

    codes = {
        str(code).strip().replace(".", "").upper()
        for code in info.get("codes", [])
    }

    matches = diagnoses_icd9[diagnoses_icd9["icd_code"].isin(codes)]

    results.append({
        "category": category,
        "yaml_codes": len(codes),
        "matching_rows": len(matches),
        "unique_admissions": matches["hadm_id"].nunique(),
        "unique_patients": matches["subject_id"].nunique()
    })

results_df = pd.DataFrame(results)

print(results_df.to_string(index=False))