import pandas as pd
from pathlib import Path


def CSVMerger(rac_file, zeo_file, labels_file, id_col="MOFname"):

    """
    Merge RAC, Zeo++, and QMOF label (bandgap) CSV files into a single dataset.

    This function:
    - Loads three CSV files: RAC descriptors, Zeo++ descriptors, and bandgap labels.
    - Ensures all files contain the same set of MOF identifiers.
    - Drops duplicate MOFs within each file.
    - Merges all three datasets on the MOF identifier.
    - Saves the merged dataset as "merged_rac_zeo_bandgap.csv".
    """

    rac_file, zeo_file, labels_file = map(Path, (rac_file, zeo_file, labels_file))
    out_dir = rac_file.parent

    rac_df = pd.read_csv(rac_file)
    zeo_df = pd.read_csv(zeo_file)
    labels_df = pd.read_csv(labels_file)
    labels_df = labels_df[[id_col, "outputs.pbe.bandgap"]]

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
    merged_rac_zeo_bandgap = (
        rac_c.merge(zeo_c, on=id_col, how="inner")
            .merge(qmof_c, on=id_col, how="inner")
    )

    #merged_ids = set(merged_rac_zeo_bandgap[id_col])
    return merged_rac_zeo_bandgap



