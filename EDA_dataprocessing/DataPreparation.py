from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def DataPreparation(merged_file,RANDOM_SEED):

    """
    Takes the merged raw dataset as input, cleans and reduces features,
    splits into train/val/test, saves them to disk, and returns the three DataFrames.
    """   
    df = pd.read_csv(merged_file)

    # removing the MOFs with Failed structures/ "Failed to featurize qmof-xx: No linkers were identified.""
    df = df[df["FailedStructures"] != 0].drop(columns=["FailedStructures"])
    
    target_feature = "outputs.pbe.bandgap"

    cols_to_drop = [
        # columns to drop from the zeoplus file  								
        "ASA_m^2/cm^3", "ASA_m^2/g", "Largest_included_sphere_along_free_path",
        "NASA_m^2/cm^3", "NASA_m^2/g", "POAV_Volume_fraction",
        "POAV_cm^3/g", "PONAV_Volume_fraction", "PONAV_cm^3/g",
    ]

    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    df = df.dropna().drop_duplicates()

    mof_id = df["MOFname"]          
    df = df.drop(columns=["MOFname"], errors="ignore")

    df_train, df_test = train_test_split(df, test_size=0.15, random_state=RANDOM_SEED)


    feature_cols_train = [col for col in df_train.columns if col != target_feature]
    feature_var = df_train[feature_cols_train].var()
    low_var = feature_var[feature_var < 1e-4].index

    df_train = df_train.drop(columns=list(set(low_var)))
    df_test = df_test.drop(columns=list(set(low_var)))

    # target feature correlation 
    corr_matrix = df_train.drop(columns=[target_feature]).corr(method= "pearson").abs() # compute correlation matrix for features only
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)) # keep upper triangle to avoid duplicate pairs (.triu)
    threshold = 0.9

    to_drop = set()

    for col in upper.columns:
        for row in upper.index:
            corr_value = upper.loc[row, col]
            if corr_value > threshold:  
                corr_row_target = abs(df_train[row].corr(df_train[target_feature]))
                corr_col_target = abs(df_train[col].corr(df_train[target_feature]))
                if corr_row_target >= corr_col_target:
                    to_drop.add(col)
                else:
                    to_drop.add(row)

    df_train = df_train.drop(columns=to_drop)
    df_test = df_test.drop(columns=to_drop)


    if mof_id is not None:
        df_train = pd.concat([df_train, mof_id.loc[df_train.index]], axis=1)
        df_test  = pd.concat([df_test, mof_id.loc[df_test.index]], axis=1)

    return df_train, df_test
