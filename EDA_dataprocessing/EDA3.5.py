from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

FOLDER  = Path(r"D:\Documents\UofT\CHE1147") 

merged_file = FOLDER  / f"merged_rac_zeo_bandgap.csv"

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
    # columns to drop from the zeoplus file, check doc for reason 
    								
    "ASA_m^2/cm^3", "ASA_m^2/g", "Largest_included_sphere_along_free_path",
    "NASA_m^2/cm^3", "NASA_m^2/g", "POAV_Volume_fraction",
    "POAV_cm^3/g", "PONAV_Volume_fraction", "PONAV_cm^3/g",

]

# Drop them safely

df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
df = df.dropna() # drop the two rows that have empty cells from the rac 2 MOFs id: 
df = df.drop_duplicates()

num_df = df.select_dtypes(include='number')
low_var = num_df.var()[num_df.var() < 1e-4].index
print(low_var)
print(f"Found {len(low_var)} low-variance features to drop.")

# drop var
#df = num_df.drop(columns=list(set(low_var)))
#reduced_df = pd.concat([df, mof_col], axis=1)


# target feature correlation 
target_feature = "outputs.pbe.bandgap"
corr_matrix = df.drop(columns=[target_feature]).corr(method= "pearson").abs() # compute correlation matrix for features only
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
#df = df.drop(columns=to_drop)
print(len(to_drop))
df.to_csv(FOLDER  / f"merged_edited.csv", index=False)

