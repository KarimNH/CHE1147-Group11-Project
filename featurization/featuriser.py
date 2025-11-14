import os
import pandas as pd
import subprocess
import csv
from pymatgen.io.cif import CifParser
from molSimplify.Informatics.MOF.PBC_functions import overlap_removal, solvent_removal
from molSimplify.Informatics.MOF.MOF_descriptors import get_MOF_descriptors, get_primitive

def cif_number(cif_folder):
    all_files = os.listdir(cif_folder)
    cif_files = [f for f in all_files if f.endswith('.cif')]
    num_cif_files = len(cif_files)
    return num_cif_files

def get_rac(featurization_directory, occupancy_tolerance=0.9, wiggleroom=1.0, depth=3):
    
    cif_folder = os.path.join(featurization_directory, 'cif')
    primitive_folder = os.path.join(featurization_directory, 'primitive')
    overlap_free_folder = os.path.join(featurization_directory, 'no_overlap')
    solvent_free_folder = os.path.join(featurization_directory, 'no_solvent')
    xyz_folder = os.path.join(featurization_directory, 'xyz')
    os.makedirs(primitive_folder, exist_ok=True)
    os.makedirs(overlap_free_folder, exist_ok=True)
    os.makedirs(solvent_free_folder, exist_ok=True)
    os.makedirs(xyz_folder, exist_ok=True)

    featurization_list = []
    for cif_file in os.listdir(cif_folder):
        if not cif_file.lower().endswith('.cif'):
            continue

        cif_path = os.path.join(cif_folder, cif_file)
        primitive_path = os.path.join(primitive_folder, cif_file)
        overlap_free_path = os.path.join(overlap_free_folder, cif_file)
        solvent_free_path = os.path.join(solvent_free_folder, cif_file)
        xyz_path = os.path.join(xyz_folder, cif_file.replace(".cif", ".xyz"))

        try:
            # Convert to primitive cell
            get_primitive(cif_path, primitive_path)

            # Remove overlapping atoms
            overlap_removal(primitive_path, overlap_free_path)

            # Remove solvent
            solvent_removal(overlap_free_path, solvent_free_path, wiggle_room=wiggleroom)

            # Compute RACs
            full_names, full_descriptors = get_MOF_descriptors(
                data=solvent_free_path,
                depth=depth,
                path=featurization_directory,
                xyzpath=xyz_path,
                wiggle_room=wiggleroom
            )

            # Add filename to features
            full_names.append('MOFname')
            full_descriptors.append(cif_file)

            # Save all features into list of dictionary
            featurization = dict(zip(full_names, full_descriptors))
            featurization_list.append(featurization)

        except Exception as e:
            print(f"Skipping {cif_file} due to error: {e}")
            continue

    df = pd.DataFrame(featurization_list)
    return df

def is_windows():
    """Return True if running on Windows."""
    return os.name == "nt"

def to_cygwin_path(win_path):
    """Convert Windows Path to Cygwin-style path string"""
    win_path = os.path.abspath(win_path)
    drive = win_path[0].lower()
    path_without_drive = win_path[2:].replace("\\", "/")
    return f"/cygdrive/{drive}/{path_without_drive}"

def run_zeopp_command(args, zeopp_dir, cygwin_bash=None):
    if is_windows():
        # Convert zeopp_dir to Cygwin path
        network_cmd = ' '.join(args)
        bash_command = f'cd "{zeopp_dir}" && {network_cmd}'
        cmd = [cygwin_bash, "-l", "-c", bash_command]
    else:
        cmd = [os.path.join(zeopp_dir, args[0])] + args[1:]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {args[0]}:\n{e.stderr.strip()}")
        return False


def pore_diameter(cif_path, output_folder, zeopp_dir, network_bin, cygwin_bash=None):
    out_file = os.path.join(output_folder,
                            os.path.splitext(os.path.basename(cif_path))[0] + ".res")
    if is_windows():
        out_file = f'"{to_cygwin_path(out_file)}"'
        cif_path = f'"{to_cygwin_path(cif_path)}"'
    args = [network_bin, "-ha", "-res", out_file, cif_path]
    return run_zeopp_command(args, zeopp_dir, cygwin_bash)


def surface_area(cif_path, output_folder, zeopp_dir, network_bin,
                 chan_radius, probe_radius, sa_samples, cygwin_bash=None):
    out_file = os.path.join(output_folder,
                            os.path.splitext(os.path.basename(cif_path))[0] + ".sa")
    if is_windows():
        out_file = f'"{to_cygwin_path(out_file)}"'
        cif_path = f'"{to_cygwin_path(cif_path)}"'
    args = [
        network_bin,
        "-ha",
        "-sa",
        str(chan_radius),
        str(probe_radius),
        str(sa_samples),
        out_file,
        cif_path
    ]
    return run_zeopp_command(args, zeopp_dir, cygwin_bash)


def probe_occupiable_volume(cif_path, output_folder, zeopp_dir, network_bin,
                            chan_radius, probe_radius, volpo_samples, cygwin_bash=None):
    out_file = os.path.join(output_folder,
                            os.path.splitext(os.path.basename(cif_path))[0] + ".volpo")
    if is_windows():
        out_file = f'"{to_cygwin_path(out_file)}"'
        cif_path = f'"{to_cygwin_path(cif_path)}"'
    
    args = [
        network_bin,
        "-ha",
        "-volpo",
        str(chan_radius),
        str(probe_radius),
        str(volpo_samples),
        out_file,
        cif_path
    ]
    return run_zeopp_command(args, zeopp_dir, cygwin_bash)


def get_geo(featurization_directory, zeopp_dir, cygwin_bash,
            chan_radius=1.4, probe_radius=1.4, sa_samples=10000, volpo_samples=10000):
    
    network_bin = "./network" if is_windows() else "network"
    input_folder = os.path.join(featurization_directory, "primitive")
    output_folder = os.path.join(featurization_directory, "zeoplusplus_output")
    os.makedirs(output_folder, exist_ok=True)
    for cif_file in os.listdir(input_folder):
        if not cif_file.lower().endswith(".cif"):
            continue
        cif_path = os.path.join(input_folder, cif_file)
        success = True
        if not pore_diameter(cif_path, output_folder, zeopp_dir, network_bin, cygwin_bash):
            success = False
        if not surface_area(cif_path, output_folder, zeopp_dir, network_bin,
                            chan_radius, probe_radius, sa_samples, cygwin_bash):
            success = False
        if not probe_occupiable_volume(cif_path, output_folder, zeopp_dir, network_bin,
                                       chan_radius, probe_radius, volpo_samples, cygwin_bash):
            success = False
    return True

#Convert the zeoplusplus data into csv files

def feature_zeopp(featurization_directory):
    input_dir = os.path.join(featurization_directory, 'zeoplusplus_output')
    output_csv = os.path.join(featurization_directory, 'zeoplusplus_featurization_frame.csv')

    if not os.path.exists(input_dir):
        print(f"Input directory not found: {input_dir}")
        return False

    all_files = os.listdir(input_dir)
    res_files = [f for f in all_files if f.endswith(".res")]
    sa_files = [f for f in all_files if f.endswith(".sa")]
    volpo_files = [f for f in all_files if f.endswith(".volpo")]
    
    results = {}
    # --- For res files ---
    for filename in res_files:
        path = os.path.join(input_dir, filename)
        with open(path, "r") as f:
            line = f.readline().strip().split()
            name = os.path.splitext(filename)[0]
            Di, Df, Dif = map(float, line[-3:])
            results[name] = {
                "Largest_included_sphere": Di,
                "Largest_free_sphere": Df,
                "Largest_included_sphere_along_free_path": Dif
            }

    # --- For .sa files ---
    for filename in sa_files:
        name = os.path.splitext(filename)[0]
        if name not in results:
            results[name] = {}
        path = os.path.join(input_dir, filename)
        with open(path, "r") as f:
            for line in f:
                if line.startswith("@"):
                    tokens = line.strip().split()
                    i = 0
                    while i < len(tokens) - 1:
                        if tokens[i].endswith(":"):
                            key = tokens[i][:-1]
                            try:
                                value = float(tokens[i + 1])
                                results[name][key] = value
                            except ValueError:
                                pass
                            i += 2
                        else:
                            i += 1

    # --- For .volpo files ---
    for filename in volpo_files:
        name = os.path.splitext(filename)[0]
        if name not in results:
            results[name] = {}
        path = os.path.join(input_dir, filename)
        with open(path, "r") as f:
            for line in f:
                if line.startswith("@"):
                    tokens = line.strip().split()
                    i = 0
                    while i < len(tokens) - 1:
                        if tokens[i].endswith(":"):
                            key = tokens[i][:-1]
                            try:
                                value = float(tokens[i + 1])
                                results[name][key] = value
                            except ValueError:
                                pass
                            i += 2
                        else:
                            i += 1
    
    all_fields = set()
    for data in results.values():
        all_fields.update(data.keys())
    fieldnames = ["MOFname"] + sorted(all_fields)

    # --- Write to CSV (fill missing fields with 0) ---
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for mof, data in results.items():
            full_data = {key: data.get(key, 0) for key in fieldnames[1:]}
            full_data["MOFname"] = mof
            writer.writerow(full_data)

    return True

def merging():
    csv1 = pd.read_csv("zeoplusplus_featurization_frame.csv")
    csv2 = pd.read_csv("rac_featurization_frame.csv")
    csv2["MOFname"] = csv2["MOFname"].str.replace(".cif", "", regex=False)
    merged = pd.merge(csv1, csv2, on="MOFname", how="inner")
    return merged
