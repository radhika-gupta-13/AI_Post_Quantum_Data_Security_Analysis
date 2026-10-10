"""
This dataset is synthetically generated for academic research and methodological demonstration.
It does not represent real organizations, individuals, or observed cybersecurity incidents.
"""

import os
import numpy as np
import pandas as pd

# Set fixed seed for reproducibility
np.random.seed(42)

N_SAMPLES = 1000

# 1. Organization ID
org_ids = [f"ORG{i:03d}" for i in range(1, N_SAMPLES + 1)]

# 2. Legacy System Dependency (1 to 10) - base factor for technical debt
legacy_dependency = np.random.randint(1, 11, size=N_SAMPLES)

# 3. Cryptographic Agility (1 to 10) - negatively correlated with legacy dependency + noise
crypto_agility_raw = 11 - legacy_dependency + np.random.normal(0, 1.5, N_SAMPLES)
crypto_agility = np.clip(np.round(crypto_agility_raw), 1, 10).astype(int)

# 4. PQC Adoption - strongly influenced by Cryptographic Agility
pqc_probs_base = np.column_stack([
    np.maximum(0, 11 - crypto_agility),  # Prob for None
    np.maximum(0, crypto_agility - 2),   # Prob for Planned
    np.maximum(0, crypto_agility - 4),   # Prob for Partial
    np.maximum(0, crypto_agility - 6)    # Prob for Implemented
])
# Normalize probabilities per row
pqc_probs = pqc_probs_base / pqc_probs_base.sum(axis=1, keepdims=True)

pqc_categories = ['None', 'Planned', 'Partial', 'Implemented']
pqc_adoption = [np.random.choice(pqc_categories, p=pqc_probs[i]) for i in range(N_SAMPLES)]

# PQC numerical weight for downstream risk calculations
pqc_risk_weight_map = {'None': 1.0, 'Planned': 0.75, 'Partial': 0.4, 'Implemented': 0.1}
pqc_risk_factor = np.array([pqc_risk_weight_map[p] for p in pqc_adoption])

# 5. AI Usage Level
ai_usage_level = np.random.choice(['Low', 'Medium', 'High'], size=N_SAMPLES, p=[0.3, 0.4, 0.3])

# 6. Data Sensitivity
data_sensitivity = np.random.choice(['Low', 'Medium', 'High', 'Critical'], size=N_SAMPLES, p=[0.2, 0.3, 0.3, 0.2])

# 7. Data Volume (5 to 100)
data_volume = np.round(np.random.uniform(5, 100, N_SAMPLES), 2)

# 8. Defensive Controls (1 to 10)
# Organizations with low legacy systems tend to have better controls, plus random variance
access_control = np.clip(np.round(11 - legacy_dependency * 0.5 + np.random.normal(0, 2, N_SAMPLES)), 1, 10).astype(int)
privacy_preservation = np.clip(np.round(np.random.uniform(1, 10, N_SAMPLES) + np.random.normal(0, 1, N_SAMPLES)), 1, 10).astype(int)
encryption_strength = np.clip(np.round(crypto_agility * 0.6 + np.random.uniform(1, 5, N_SAMPLES)), 1, 10).astype(int)
security_monitoring = np.clip(np.round(access_control * 0.5 + np.random.uniform(1, 5, N_SAMPLES)), 1, 10).astype(int)

# 9. AI Security Risk (0 to 100)
ai_base_map = {'Low': 20, 'Medium': 50, 'High': 80}
ai_base = np.array([ai_base_map[val] for val in ai_usage_level])
ai_sec_risk = np.clip(np.round(ai_base + (data_volume * 0.15) - (access_control * 2) + np.random.normal(0, 10, N_SAMPLES)), 0, 100)

# 10. Privacy Risk (0 to 100)
sens_base_map = {'Low': 15, 'Medium': 40, 'High': 65, 'Critical': 85}
sens_base = np.array([sens_base_map[val] for val in data_sensitivity])
privacy_risk = np.clip(np.round(sens_base + (data_volume * 0.2) - (privacy_preservation * 4.5) + np.random.normal(0, 10, N_SAMPLES)), 0, 100)

# 11. Incident Probability (0 to 100)
incident_prob_raw = (
    0.25 * ai_sec_risk +
    0.25 * privacy_risk +
    2.5 * legacy_dependency -
    3.0 * access_control -
    3.0 * security_monitoring +
    np.random.normal(0, 8, N_SAMPLES)
)
incident_probability = np.clip(np.round(incident_prob_raw), 0, 100)

# 12. Overall Risk Level Calculation
composite_score = (
    0.20 * privacy_risk +
    0.20 * ai_sec_risk +
    0.25 * incident_probability +
    1.5 * legacy_dependency +
    10.0 * pqc_risk_factor -
    1.5 * access_control -
    1.5 * privacy_preservation -
    1.5 * encryption_strength -
    1.5 * security_monitoring +
    np.random.normal(0, 5, N_SAMPLES)  # Noise to prevent deterministic ML thresholds
)

# Binning composite score into categorical risk levels
quantiles = np.quantile(composite_score, [0.25, 0.60, 0.85])

def categorize_risk(score):
    if score <= quantiles[0]:
        return 'Low'
    elif score <= quantiles[1]:
        return 'Medium'
    elif score <= quantiles[2]:
        return 'High'
    else:
        return 'Critical'

overall_risk_level = [categorize_risk(s) for s in composite_score]

# Assemble DataFrame
df = pd.DataFrame({
    'Organization_ID': org_ids,
    'AI_Usage_Level': ai_usage_level,
    'Data_Sensitivity': data_sensitivity,
    'Data_Volume': data_volume,
    'Privacy_Risk': privacy_risk,
    'AI_Security_Risk': ai_sec_risk,
    'Access_Control_Strength': access_control,
    'Privacy_Preservation_Level': privacy_preservation,
    'Encryption_Strength': encryption_strength,
    'PQC_Adoption': pqc_adoption,
    'Cryptographic_Agility': crypto_agility,
    'Security_Monitoring': security_monitoring,
    'Legacy_System_Dependency': legacy_dependency,
    'Incident_Probability': incident_probability,
    'Overall_Risk_Level': overall_risk_level
})

# Display Output Details
print("=== Dataset Generation Complete ===")
print(f"Total Rows Generated: {len(df)}")
print(f"Total Columns Generated: {len(df.columns)}\n")

print("--- First 10 Rows ---")
print(df.head(10).to_string())

print("\n--- Basic Descriptive Statistics ---")
print(df.describe().to_string())

print("\n--- Distribution of Overall_Risk_Level ---")
print(df['Overall_Risk_Level'].value_counts())

print("\n--- Missing Values Check ---")
print(df.isnull().sum())

# Save to CSV
file_name = "AI_Post_Quantum_Data_Security_1000_Synthetic.csv"
df.to_csv(file_name, index=False)
file_path = os.path.abspath(file_name)

print(f"\nDataset successfully saved at:\n{file_path}")
