from pathlib import Path
import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt

CSV_PATH = Path(r"D:\Documents\UofT\CHE1147\rac_features_linkers.csv") 

df = pd.read_csv(CSV_PATH)



def check(dataframe):
    dataframe["MOFname"].nunique()
    dataframe.isna().sum()
    dataframe.duplicated()
    return


summary = df.describe().T
#print(summary[["mean", "std", "min", "max"]])

df = df.dropna()

num_df = df.select_dtypes(include='number')
mof_col = df["MOFname"]  # keep this separately

print(f"Number of RAC descriptors/columns: {num_df.shape[1]}")
print(f"Number of samples/rows: {num_df.shape[0]}")

corr_matrix = num_df.corr()

to_drop = set()
threshold = 0.9

for col in corr_matrix.columns:
    for row in corr_matrix.index:
        if abs(corr_matrix.loc[row, col]) > threshold and row != col:
            to_drop.add(row)




all_to_drop = set(to_drop) | set(low_var)
print(f"Total features to drop: {len(all_to_drop)}")
print(f"Dropped {len(to_drop)} highly correlated features.")

reduced_num_df = num_df.drop(columns=list(all_to_drop))
reduced_df = pd.concat([reduced_num_df, mof_col], axis=1)
print(f"Reduced feature set: {reduced_num_df.shape[1]} numeric columns + MOFname column")

low_var = num_df.var()[num_df.var() < 1e-6].index
print(f"Found {len(low_var)} low-variance features to drop.")

output_path = CSV_PATH.parent / f"{CSV_PATH.stem}_reduced.csv"
reduced_df.to_csv(output_path, index=False)

