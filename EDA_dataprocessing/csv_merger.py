import pandas as pd
from pathlib import Path

FOLDER  = Path(r"D:\Documents\UofT\CHE1147") 

rac_file = FOLDER  / f"rac_featurization_frame.csv"
zeo_file = FOLDER  / f"zeoplus_features.csv"
labels_file = FOLDER  / f"qmof_labels.csv"


rac_df = pd.read_csv(rac_file)
zeo_df = pd.read_csv(zeo_file)
labels_df = pd.read_csv(labels_file)

id_col = "MOFname"
#print(rac_df["MOFname"])

set_rac = set(rac_df[id_col])
set_zeo = set(zeo_df[id_col])
set_labels = set(labels_df[id_col])

# find intersection (common MOFs)
common_mofs = set_rac & set_zeo & set_labels
print(f"Number of MOFs common between all three files: {len(common_mofs)}")


# removing duplciates 
rac  = rac_df.drop_duplicates(subset=[id_col])
zeo  = zeo_df.drop_duplicates(subset=[id_col])
qmof = labels_df.drop_duplicates(subset=[id_col])

# filter to only common MOFs
rac_c  = rac[rac[id_col].isin(common_mofs)]
zeo_c  = zeo[zeo[id_col].isin(common_mofs)]
qmof_c = qmof[qmof[id_col].isin(common_mofs)]
rac_c = rac_c.rename(columns={"0": "FailedStructures"})

# merging 
merged = (
    rac_c.merge(zeo_c, on=id_col, how="inner")
         .merge(qmof_c, on=id_col, how="inner")
)

merged_ids = set(merged[id_col])


out_path = FOLDER / "merged_rac_zeo_qmof.csv"
merged.to_csv(out_path, index=False)
print(f"Saved: {out_path}")
print(f"Final row count: {len(merged)} (should equal common MOFs)")




