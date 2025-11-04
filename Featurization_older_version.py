import os
import pandas as pd
import subprocess
from pathlib import Path
import csv
from pymatgen.io.cif import CifParser
from molSimplify.Informatics.MOF.PBC_functions import overlap_removal, solvent_removal
from molSimplify.Informatics.MOF.MOF_descriptors import get_MOF_descriptors
from pymatgen.core import Structure

import warnings
from pymatgen.io.cif import CifParser

# First function: Counting number of cifs to process
def cif_number(cif_folder):
    all_files = os.listdir(cif_folder)
    cif_files = [f for f in all_files if f.endswith('.cif')]
    num_cif_files = len(cif_files)
    return num_cif_files

# Second function: Featurization using molSimplify to get RACs

def _load_structure(cif_path, occupancy_tolerance=None):
    # Try modern parser first
    if occupancy_tolerance is None:
        parser = CifParser(str(cif_path))
    else:
        parser = CifParser(str(cif_path), occupancy_tolerance=occupancy_tolerance)
    structs = parser.parse_structures(primitive=False)
    if structs:
        return structs[0]

    # Fallback: pymatgen's generic loader
    return Structure.from_file(str(cif_path))

def debug_parse_cif(cif_path):
    print(f"[DEBUG] Trying to parse: {cif_path}")
    if not os.path.exists(cif_path):
        raise FileNotFoundError(f"Path does not exist: {cif_path}")

    # Try a few parser settings commonly needed for QMOF CIFs
    tries = [
        dict(occupancy_tolerance=None, primitive=False),
        dict(occupancy_tolerance=1.0, primitive=False),   # relax occupancy check
        dict(occupancy_tolerance=0.01, primitive=False),  # strict occupancy
        dict(occupancy_tolerance=1.0, primitive=True),    # also try primitive parse
    ]

    for i, opts in enumerate(tries, 1):
        try:
            print(f"[DEBUG] Attempt {i}: {opts}")
            parser = CifParser(str(cif_path), occupancy_tolerance=opts["occupancy_tolerance"]) \
                     if opts["occupancy_tolerance"] is not None else CifParser(str(cif_path))
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                structs = parser.parse_structures(primitive=opts["primitive"])
                if structs:
                    print(f"[DEBUG] Parsed OK with attempt {i}")
                    return structs[0]
                else:
                    print(f"[DEBUG] Attempt {i} returned 0 structures (warnings: {len(w)})")
        except Exception as e:
            print(f"[DEBUG] Attempt {i} failed: {e}")

    # Final fallback: generic loader (may still fail if CIF is malformed)
    from pymatgen.core import Structure
    print("[DEBUG] Falling back to Structure.from_file(...)")
    return Structure.from_file(str(cif_path))

def get_rac(featurization_directory, occupancy_tolerance=0.01, wiggleroom=1.0, depth=3):
    cif_folder = os.path.join(featurization_directory, 'cif')
    primitive_folder = os.path.join(featurization_directory, 'primitive')
    overlap_free_folder = os.path.join(featurization_directory, 'no_overlap')
    solvent_free_folder = os.path.join(featurization_directory, 'no_solvent')
    xyz_folder = os.path.join(featurization_directory, 'xyz')
    for p in (primitive_folder, overlap_free_folder, solvent_free_folder, xyz_folder):
        os.makedirs(p, exist_ok=True)

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
            structure = debug_parse_cif(cif_path)

            if structure is None:
                raise ValueError("Parser returned no structures")

            primitive_structure = structure.get_primitive_structure()
            primitive_structure.to(fmt="cif", filename=primitive_path)

            overlap_removal(primitive_path, overlap_free_path)
            solvent_removal(overlap_free_path, solvent_free_path, wiggle_room=wiggleroom)

            full_names, full_descriptors = get_MOF_descriptors(
                data=solvent_free_path,
                depth=depth,
                path=featurization_directory,
                xyzpath=xyz_path,
                wiggle_room=wiggleroom
            )

            full_names.append('MOFname')
            full_descriptors.append(cif_file)
            featurization_list.append(dict(zip(full_names, full_descriptors)))

        except Exception as e:
            print(f"[WARN] Skipping {cif_file}: {e}")
            continue

    return pd.DataFrame(featurization_list)

# Third function: Featurization using zeoplusplus

# === Run Zeo++ command ===
def run_zeopp_command(args, zeopp_dir):
    """Run Zeo++ command using subprocess."""
    try:
        # use subprocess run to make it run like as in the terminal
        result = subprocess.run(
            args,
            cwd=zeopp_dir,
            check=True,
            capture_output=True,
            text=True
        )
        return True
    except subprocess.CalledProcessError as e:
        return False

# === Zeoplusplus arguments ===
def pore_diameter(cif_path, output_folder, zeopp_dir, network_bin):
    out_file = os.path.join(output_folder, os.path.splitext(os.path.basename(cif_path))[0] + ".res")
    args = [
        os.path.join(zeopp_dir, network_bin),
        "-ha",
        "-res",
        out_file,
        cif_path
    ]
    return run_zeopp_command(args, zeopp_dir)

def surface_area(cif_path, output_folder, zeopp_dir, network_bin, chan_radius, probe_radius, sa_samples):
    out_file = os.path.join(output_folder, os.path.splitext(os.path.basename(cif_path))[0] + ".sa")
    args = [
        os.path.join(zeopp_dir, network_bin),
        "-ha",
        "-sa",
        str(chan_radius),
        str(probe_radius),
        str(sa_samples),
        out_file,
        cif_path
    ]
    return run_zeopp_command(args, zeopp_dir)

def probe_occupiable_volume(cif_path, output_folder, zeopp_dir, network_bin, chan_radius, probe_radius, volpo_samples):
    out_file = os.path.join(output_folder, os.path.splitext(os.path.basename(cif_path))[0] + ".volpo")
    args = [
        os.path.join(zeopp_dir, network_bin),
        "-ha",
        "-volpo",
        str(chan_radius),
        str(probe_radius),
        str(volpo_samples),
        out_file,
        cif_path
    ]
    return run_zeopp_command(args, zeopp_dir)

# === Main Function ===
def get_geo(featurization_directory,
            zeopp_dir,
            network_bin="network",
            chan_radius=1.4,
            probe_radius=1.4,
            sa_samples=10000,
            volpo_samples=10000):

    input_folder = os.path.join(featurization_directory, "primitive")
    output_folder = os.path.join(featurization_directory, "zeoplusplus_output")
    os.makedirs(output_folder, exist_ok=True)

    for cif_file in os.listdir(input_folder):
        if not cif_file.lower().endswith(".cif"):
            continue

        cif_path = os.path.join(input_folder, cif_file)
        print(f"Processing {cif_file}")
        success = True

        if not pore_diameter(cif_path, output_folder, zeopp_dir, network_bin):
            success = False
        if not surface_area(cif_path, output_folder, zeopp_dir, network_bin, chan_radius, probe_radius, sa_samples):
            success = False
        if not probe_occupiable_volume(cif_path, output_folder, zeopp_dir, network_bin, chan_radius, probe_radius, volpo_samples):
            success = False

        if success:
            print(f"Finished {cif_file}")
        else:
            print(f"Issues with {cif_file}")

    return True

#Convert the zeoplusplus data into csv files

def feature_zeopp(featurization_directory):
    input_dir = os.path.join(featurization_directory, 'zeoplusplus_output')
    output_csv = os.path.join(featurization_directory, 'zeoplusplus_featurization_frame.csv')

    if not os.path.exists(input_dir):
        print(f"Input directory not found: {input_dir}")
        return False
    
    # === Get files ===
    all_files = os.listdir(input_dir)
    res_files = [f for f in all_files if f.endswith(".res")]
    sa_files = [f for f in all_files if f.endswith(".sa")]
    volpo_files = [f for f in all_files if f.endswith(".volpo")]
    vol_files = [f for f in all_files if f.endswith(".vol")]
    
    results = {}
    # --- Parse .res ---
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

    # --- Parse .sa ---
    for filename in sa_files:
        name = os.path.splitext(filename)[0]
        if name not in results:
            results[name] = {}
    
        path = os.path.join(input_dir, filename)
        with open(path, "r") as f: #r is read mode, f is a variable
            for line in f:
                if line.startswith("@"):
                    tokens = line.strip().split() #a list of strings
                    #strip meanins to remove space in the beginning and the end
                    #split is by space
                    i = 0
                    while i < len(tokens) - 1: #loop starting from 0
                        if tokens[i].endswith(":"):
                            key = tokens[i][:-1]  # remove the trailing colon ":" 
                            try:
                                value = float(tokens[i + 1])
                                results[name][key] = value
                            except ValueError:
                                pass  # skip if the value is not a float
                            i += 2 # move forward by two tokens
                        else:
                            i += 1 # move forward by one tokens

    # --- Parse .vol ---
    for filename in vol_files:
        name = os.path.splitext(filename)[0]
        if name not in results:
            results[name] = {}

        path = os.path.join(input_dir, filename)
        with open(path, "r") as f: #r is read mode, f is a variable
            for line in f:
                if line.startswith("@"):
                    tokens = line.strip().split() #a list of strings
                    #strip meanins to remove space in the beginning and the end
                    #split is by space
                    i = 0
                    while i < len(tokens) - 1: #loop starting from 0
                        if tokens[i].endswith(":"):
                            key = tokens[i][:-1]  # remove the trailing colon ":" 
                            try:
                                value = float(tokens[i + 1])
                                results[name][key] = value
                            except ValueError:
                                pass  # skip if the value is not a float
                            i += 2 #move forward by two tokens
                        else:
                            i += 1 #move forward by one tokens

    # --- Parse .volpo ---
    for filename in volpo_files:
        name = os.path.splitext(filename)[0]
        if name not in results:
            results[name] = {}

        path = os.path.join(input_dir, filename)
        with open(path, "r") as f: #r is read mode, f is a variable
            for line in f:
                if line.startswith("@"):
                    tokens = line.strip().split() #a list of strings
                    #strip meanins to remove space in the beginning and the end
                    #split is by space
                    i = 0
                    while i < len(tokens) - 1: #loop starting from 0
                        if tokens[i].endswith(":"):
                            key = tokens[i][:-1]  # remove the trailing colon ":" 
                            try:
                                value = float(tokens[i + 1])
                                results[name][key] = value
                            except ValueError:
                                pass  # skip if the value is not a float
                            i += 2 #move forward by two tokens
                        else:
                            i += 1 #move forward by one tokens


    # --- Build column list ---
    all_fields = set()
    for data in results.values():
        all_fields.update(data.keys())
    fieldnames = ["MOFname"] + sorted(all_fields)

    # --- Write to CSV (fill missing fields with 0) ---
    with open(output_csv, "w", newline="") as f: #w stands for writing
        writer = csv.DictWriter(f, fieldnames=fieldnames) #this writes filemane as the first row
        writer.writeheader()
        for mof, data in results.items(): # mof is the key, data is the features
            full_data = {key: data.get(key, 0) for key in fieldnames[1:]} #fills missing values as 0
            full_data["MOFname"] = mof
            writer.writerow(full_data)

    return True

featurization_directory = r"D:\scratch\QMOFtest"
cif_folder = os.path.join(featurization_directory, 'cif')
print(f"Number of .cif files: {cif_number(cif_folder)}")

df = get_rac(featurization_directory)
df.to_csv(os.path.join(featurization_directory, 'rac_featurization_frame.csv'), index=False)

# zeopp_dir = "/Users/sze/zeo++-0.3"
# get_geo(
#     featurization_directory=featurization_directory,
#     zeopp_dir=zeopp_dir
# )
# feature_zeopp(featurization_directory=featurization_directory)
