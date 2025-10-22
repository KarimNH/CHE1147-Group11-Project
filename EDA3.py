from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

FOLDER  = Path(r"D:\Documents\UofT\CHE1147") 

merged_file = FOLDER  / f"merged_rac_zeo_qmof.csv"

df = pd.read_csv(merged_file)


# removing the MOFs with Failed structures/ "Failed to featurize qmof-xx: No linkers were identified.""
df = df[df["FailedStructures"] != 0] 
df = df.drop(columns=["FailedStructures"])
 
print(df)
###################################################
# Descriptive EDA
###################################################

mof_id = df["MOFname"]          
df = df.drop(columns=["MOFname"], errors="ignore")


#  
###################################################
# columns to drop from the labels - read.txt explains why
# Based on physical understanding (corr(), var()...)
###################################################

cols_to_drop = [
    # metadata / IDs
    "name", "info.formula", "info.formula_reduced",
    "info.mofid.mofid", "info.mofid.mofkey",
    "info.mofid.smiles_nodes", "info.mofid.smiles_linkers", "info.mofid.smiles",
    "info.mofid.topology",
    "info.synthesized", "info.source", "info.doi",
    # symmetry text (we’ll keep only the numeric spacegroup elsewhere)
    "info.symmetry.spacegroup", "info.symmetry.spacegroup_crystal", "info.symmetry.pointgroup",
    # PBE inputs (constant / setup)
    "inputs.pbe.theory", "inputs.pbe.pseudopotentials", "inputs.pbe.encut",
    "inputs.pbe.kpoints", "inputs.pbe.gamma", "inputs.pbe.spin",
    # PBE redundant outputs
    "outputs.pbe.energy_vdw", "outputs.pbe.energy_elec", "outputs.pbe.directgap",
    "outputs.pbe.bandgap_spins", "outputs.pbe.cbm_spins",
    "outputs.pbe.vbm_spins", "outputs.pbe.directgap_spins",

    # inputs.hle17 and hse06, hse06_10hf, etc.
    "inputs.hle17.theory", "inputs.hle17.pseudopotentials", "inputs.hle17.encut",
    "inputs.hle17.kpoints", "inputs.hle17.gamma", "inputs.hle17.spin",
    "outputs.hle17.energy_total", "outputs.hle17.energy_vdw", "outputs.hle17.energy_elec",
    "outputs.hle17.net_magmom", "outputs.hle17.bandgap", "outputs.hle17.cbm",
    "outputs.hle17.vbm", "outputs.hle17.directgap", "outputs.hle17.bandgap_spins",
    "outputs.hle17.vbm_spins", "outputs.hle17.cbm_spins", "outputs.hle17.directgap_spins",

    "inputs.hse06.theory", "inputs.hse06.pseudopotentials", "inputs.hse06.encut",
    "inputs.hse06.kpoints", "inputs.hse06.gamma", "inputs.hse06.spin",
    "outputs.hse06.energy_total", "outputs.hse06.energy_vdw", "outputs.hse06.energy_elec",
    "outputs.hse06.net_magmom", "outputs.hse06.bandgap", "outputs.hse06.cbm",
    "outputs.hse06.vbm", "outputs.hse06.directgap", "outputs.hse06.bandgap_spins",
    "outputs.hse06.vbm_spins", "outputs.hse06.cbm_spins", "outputs.hse06.directgap_spins",

    "inputs.hse06_10hf.theory", "inputs.hse06_10hf.pseudopotentials", "inputs.hse06_10hf.encut",
    "inputs.hse06_10hf.kpoints", "inputs.hse06_10hf.gamma", "inputs.hse06_10hf.spin",
    "outputs.hse06_10hf.energy_total", "outputs.hse06_10hf.energy_vdw", "outputs.hse06_10hf.energy_elec",
    "outputs.hse06_10hf.net_magmom", "outputs.hse06_10hf.bandgap", "outputs.hse06_10hf.cbm",
    "outputs.hse06_10hf.vbm", "outputs.hse06_10hf.directgap", "outputs.hse06_10hf.bandgap_spins",
    "outputs.hse06_10hf.vbm_spins", "outputs.hse06_10hf.cbm_spins", "outputs.hse06_10hf.directgap_spins",

    # columns to drop from the zeoplus file, check doc for reason 
    								
    "ASA_m^2/cm^3", "ASA_m^2/g", "Largest_included_sphere_along_free_path",
    "NASA_m^2/cm^3", "NASA_m^2/g", "POAV_Volume_fraction",
    "POAV_cm^3/g", "PONAV_Volume_fraction", "PONAV_cm^3/g",

]

# Drop them safely

df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
df = df.dropna() # drop the two rows that have empty cells from the rac id:


num_df = df.select_dtypes(include='number')
low_var = num_df.var()[num_df.var() < 1e-4].index
print(low_var)
print(f"Found {len(low_var)} low-variance features to drop.")

# drop var
#df = num_df.drop(columns=list(set(low_var)))
#reduced_df = pd.concat([df, mof_col], axis=1)


# target feature correlation 
target_feature = "outputs.pbe.bandgap"
corr_matrix = df.drop(columns=[target_feature]).corr().abs() # compute correlation matrix for features only
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)) # keep upper triangle to avoid duplicate pairs
threshold = 0.9
#pearson_corr = df_train_stratified.corr(method='pearson')["TARGET"].sort_values(ascending=False)

to_drop = set()

for col in upper.columns:
    for row in upper.index:
        corr_value = upper.loc[row, col]
        if corr_value > threshold:  # threshold for high correlation
            # Compare correlations with target
            corr_row_target = abs(df[row].corr(df[target_feature]))
            corr_col_target = abs(df[col].corr(df[target_feature]))
            
            if corr_row_target >= corr_col_target:
                to_drop.add(col)
                print(f"Can drop '{col}' — correlated with '{row}' ({corr_value:.2f}); "
                      f"'{row}' should be kept because it correlates more with target "
                      f"({corr_row_target:.2f} vs {corr_col_target:.2f}).")
            else:
                to_drop.add(row)
                print(f"Can drop '{row}' — correlated with '{col}' ({corr_value:.2f}); "
                      f"'{col}' should be kept because it correlates more with target "
                      f"({corr_col_target:.2f} vs {corr_row_target:.2f}).")

# Drop the redundant columns
# df_reduced = df.drop(columns=to_drop)
print(len(to_drop))
df.to_csv(FOLDER  / f"merged_edited.csv", index=False)

