import concurrent.futures
import click
import os
import time
from click import style
import pandas as pd
import util.data as db
import util.dataframe as df
from fractions import Fraction
import preprocess.cif_parser as cif_parser 
import util.folder as folder
import util.log as log
import preprocess.supercell as supercell
import featurizer.interatomic as interatomic_featurizer
import featurizer.environment_wyckoff as env_wychoff_featurizer
import featurizer.environment_binary as env_featurizer_binary
import featurizer.environment_ternary as env_featurizer_ternary
import featurizer.environment_dataframe as env_dataframe
import featurizer.coordinate_number_dataframe as coordinate_number_dataframe
from collections import defaultdict

# Helper: returns a rounded df (only float64 columns)
def round_df(df):
    numeric_cols = df.select_dtypes(include=['float64']).columns
    df[numeric_cols] = df[numeric_cols].round(4)
    return df

# Helper: prints pair_info
def print_atom_pairs_info(atom_pairs_info_dict):
    for label, data in atom_pairs_info_dict.items():
        print("Pair info:")
        for pair_info in data["pair_info"][:10]:
            print(pair_info)

# Define a function to process each CIF file
def process_cif_file(args):
    # Assignment: Read args
    filename, cif_folder_directory, MAX_ATOMS_COUNT, radii_data, loop_tags, xl = args
    
    # Initialize empty DFs
    ib_df = pd.DataFrame()
    it_df = pd.DataFrame()
    iu_df = pd.DataFrame()
    aewb_df = pd.DataFrame()
    aewt_df = pd.DataFrame()
    aewu_df = pd.DataFrame()
    aeb_df = pd.DataFrame()
    aet_df = pd.DataFrame()
    cnb_df = pd.DataFrame()
    cnt_df = pd.DataFrame()
    
    # Time: start
    start_time = time.time()
    
    # Get: filename
    filename_base = os.path.basename(filename)
    print(f"Processing {filename_base}...")
    
    # Initialize: Result
    results = {
        "filename": filename,
        "processed": False,
        "execution_time": 0,
        "interatomic_binary_df": ib_df,
        "interatomic_ternary_df": it_df,
        "interatomic_universal_df": iu_df,
        "atomic_env_wyckoff_binary_df": aewb_df,
        "atomic_env_wyckoff_ternary_df": aewt_df,
        "atomic_env_wyckoff_universal_df": aewu_df,
        "atomic_env_binary_df": aeb_df,
        "atomic_env_ternary_df": aet_df,
        "coordinate_number_binary_df": cnb_df,
        "coordinate_number_ternary_df": cnt_df,
        "feature_log": {}
    }
    
    # Check: Reject if not a valid CIF file
    if not cif_parser.valid_cif(filename):
        print(f"Rejecting {filename_base}... No structure found.")
        return results
    
    # Initialize: isBinary, isTernary booleans to False
    isBinary = isTernary = False
    
    # Initialize: Atom symbols to empty string
    A = B = R = M = X = ''
    
    # Preprocess: Inplace changes to CIF prior to gemmi
    cif_parser.preprocess_cif_file(filename)
    cif_parser.take_care_of_atomic_site(filename)
    cif_parser.remove_text_after_author(filename)
    
    # Get: CIF block and Print
    print(cif_parser.get_CIF_block(filename))
    
    # Assignment: variables based on CIF block
    CIF_block = cif_parser.get_CIF_block(filename)
    CIF_id = CIF_block.name
    cell_lengths, cell_angles_rad = supercell.process_cell_data(CIF_block)
    CIF_loop_values = cif_parser.get_loop_values(CIF_block, loop_tags)
    all_coords_list = supercell.get_coords_list(CIF_block, CIF_loop_values)
    all_points, unique_labels, unique_atoms_tuple = supercell.get_points_and_labels(all_coords_list, CIF_loop_values)
    
    # Check: Number of atoms in the supercell to skip the file
    if cif_parser.exceeds_atom_count_limit(all_points, MAX_ATOMS_COUNT):
        click.echo(style(f"Skipped - {filename_base} has {len(all_points)} atoms [Atoms Exceeded]", fg="yellow"))
        return results
        
    # Count++: Number of files processed
    unique_atoms_tuple, num_of_unique_atoms, formula_string = cif_parser.extract_formula_and_atoms(CIF_block)
    atomic_pair_list = supercell.get_atomic_pair_list(all_points, cell_lengths, cell_angles_rad)
    atom_pair_info_dict = supercell.get_atom_pair_info_dict(unique_labels, atomic_pair_list)
    
    # Check: Type of compound: unary, binary, or ternary
    isBinary = num_of_unique_atoms == 2
    isTernary = num_of_unique_atoms == 3
    
    # Store: Processed CIF data
    CIF_data = (CIF_id, cell_lengths, cell_angles_rad, CIF_loop_values, formula_string)
        
    # Initialize: Variables for atomic enviornment
    atom_counts = defaultdict(int)
    unique_shortest_labels, atom_counts = env_featurizer_binary.get_unique_shortest_labels(atom_pair_info_dict, unique_labels, cif_parser)
    
    # Check: Binary?
    if isBinary:
        # Assignment: A, B
        A, B = unique_atoms_tuple[0][0], unique_atoms_tuple[1][0]
        
        # Compute: Interatomic features
        ib_df, iu_df = interatomic_featurizer.get_interatomic_binary_df(filename,
                                                    ib_df,
                                                    iu_df,
                                                    all_points,
                                                    unique_atoms_tuple,
                                                    atomic_pair_list,
                                                    CIF_data,
                                                    radii_data)
        
        # Compute: Environment Wyckoff features
        aewb_df, aewu_df = env_wychoff_featurizer.get_env_wychoff_binary_df(filename,
                                                    xl,
                                                    aewb_df,
                                                    aewu_df,
                                                    unique_atoms_tuple,
                                                    CIF_loop_values,
                                                    radii_data,
                                                    CIF_data,
                                                    atomic_pair_list)
                    
        # Compute: Atomic environment features
        aeb_df = env_dataframe.get_env_binary_df(
                                                aeb_df,
                                                unique_atoms_tuple,
                                                unique_labels,
                                                unique_shortest_labels,
                                                atom_pair_info_dict,
                                                atom_counts,
                                                CIF_data)
        
        # Compute: Coordinate number features
        cnb_df = coordinate_number_dataframe.get_coordinate_number_binary_df(
                                                    isBinary,
                                                    cnb_df,
                                                    unique_atoms_tuple,
                                                    unique_labels,
                                                    atomic_pair_list,
                                                    atom_pair_info_dict,
                                                    CIF_data,
                                                    radii_data)
        
    # Check: Ternary?      
    if isTernary:
        # Assignement: R, M, X
        R, M, X = unique_atoms_tuple[0][0], unique_atoms_tuple[1][0], unique_atoms_tuple[2][0]
        
        # Compute: Interatomic features 
        it_df, iu_df = interatomic_featurizer.get_interatomic_ternary_df(filename,
                                                    it_df,
                                                    iu_df,
                                                    all_points,
                                                    unique_atoms_tuple,
                                                    atomic_pair_list,
                                                    CIF_data,
                                                    radii_data)
        
        # Compute: Environment Wyckoff features
        aewt_df, aewu_df = env_wychoff_featurizer.get_env_wychoff_ternary_df(filename,
                                                    xl,
                                                    aewt_df,
                                                    aewu_df,
                                                    unique_atoms_tuple,
                                                    CIF_loop_values,
                                                    radii_data,
                                                    CIF_data,
                                                    atomic_pair_list)
        
        # Compute: Atomic environment features
        aet_df = env_dataframe.get_env_ternary_df(
                                                aet_df,
                                                unique_atoms_tuple,
                                                unique_labels,
                                                unique_shortest_labels,
                                                atom_pair_info_dict,
                                                atom_counts,
                                                CIF_data)      
        
        # Compute: Coordinate number features
        cnt_df = coordinate_number_dataframe.get_coordinate_number_ternary_df(
                                                    isBinary,
                                                    cnt_df,
                                                    unique_atoms_tuple,
                                                    unique_labels,
                                                    atomic_pair_list,
                                                    atom_pair_info_dict,
                                                    CIF_data,
                                                    radii_data)
       
    # Time: stop
    end_time = time.time()
    execution_time = end_time - start_time
    click.echo(style(f"{execution_time:.2f}s to process {len(all_points)} atoms", fg="green"))
    
    # Write: log
    featurizer_log = {
        "Filename": filename_base,
        "CIF": CIF_id,
        "Compound": formula_string,
        "Number of atoms": len(all_points),
        "Execution time (s)": execution_time
    }
    
    # Write: results
    results = {
        "filename": filename,
        "processed": True,
        "execution_time": execution_time,
        "interatomic_binary_df": ib_df,
        "interatomic_ternary_df": it_df,
        "interatomic_universal_df": iu_df,
        "atomic_env_wyckoff_binary_df": aewb_df,
        "atomic_env_wyckoff_ternary_df": aewt_df,
        "atomic_env_wyckoff_universal_df": aewu_df,
        "atomic_env_binary_df": aeb_df,
        "atomic_env_ternary_df": aet_df,
        "coordinate_number_binary_df": cnb_df,
        "coordinate_number_ternary_df": cnt_df,
        "featurizer_log": featurizer_log
    }
    
    return results

def main_parallel():
    # Choose: Directory
    script_directory = os.path.dirname(os.path.abspath(__file__))
    cif_folder_directory = folder.choose_CIF_directory(script_directory)
    
    # Initialize: CIF file list
    files_lst = [os.path.join(cif_folder_directory, file) for file in os.listdir(cif_folder_directory) if file.endswith('.cif')]
    total_files = len(files_lst)
    
    # Get the default atom count or prompt the user
    skip_based_on_atoms = click.confirm('Do you want to skip any CIF files based on the number of unique atoms in the supercell (Default: N)?')

    if skip_based_on_atoms:
        MAX_ATOMS_COUNT = click.prompt('Enter the threshold for the maximum number of atoms in the supercell. Files with atoms exceeding this count will be skipped', type=int)
    else:
        MAX_ATOMS_COUNT = 2560  # A large number to essentially disable skipping
    
    # Initialize: Variables
    num_files_processed = 0 
    running_total_time = 0
    
    # Reading: Element Database
    radii_data = db.get_radii_data()
    loop_tags = cif_parser.get_loop_tags()
    property_file = './element_database/element_properties_for_ML-my elements.xlsx'
    xl = pd.read_excel(property_file, engine="openpyxl")
    
    # Initialize: DataFrames and lists to store results
    interatomic_binary_df = interatomic_ternary_df = interatomic_universal_df = pd.DataFrame()
    atomic_env_wyckoff_binary_df = atomic_env_wyckoff_ternary_df = atomic_env_wyckoff_universal_df = pd.DataFrame()
    atomic_env_binary_df = atomic_env_ternary_df = pd.DataFrame()
    featurizer_log_entries = []

    coordinate_number_binary_df = pd.DataFrame()
    coordinate_number_binary_max_df = pd.DataFrame()
    coordinate_number_binary_min_df = pd.DataFrame()
    coordinate_number_binary_avg_df = pd.DataFrame()

    coordinate_number_ternary_df = pd.DataFrame()
    coordinate_number_ternary_max_df = pd.DataFrame()
    coordinate_number_ternary_min_df = pd.DataFrame()
    coordinate_number_ternary_avg_df = pd.DataFrame()

    # Prepare arguments for parallel processing
    process_args = [(filename, cif_folder_directory, MAX_ATOMS_COUNT, radii_data, loop_tags, xl) for filename in files_lst]

    # Use ProcessPoolExecutor to execute processing in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_file = {executor.submit(process_cif_file, args): args[0] for args in process_args}
        
        for future in concurrent.futures.as_completed(future_to_file):
            filename = future_to_file[future]
            try:
                result = future.result()
                if result['processed']:
                    # Count++: number of files processed
                    num_files_processed += 1
                    
                    # Aggregate
                    interatomic_binary_df = pd.concat([interatomic_binary_df, result["interatomic_binary_df"]], ignore_index=True)
                    interatomic_ternary_df = pd.concat([interatomic_ternary_df, result["interatomic_ternary_df"]], ignore_index=True)
                    interatomic_universal_df = pd.concat([interatomic_universal_df, result["interatomic_universal_df"]], ignore_index=True)
                    atomic_env_wyckoff_binary_df = pd.concat([atomic_env_wyckoff_binary_df, result["atomic_env_wyckoff_binary_df"]], ignore_index=True)
                    atomic_env_wyckoff_ternary_df = pd.concat([atomic_env_wyckoff_ternary_df, result["atomic_env_wyckoff_ternary_df"]], ignore_index=True)
                    atomic_env_wyckoff_universal_df = pd.concat([atomic_env_wyckoff_universal_df, result["atomic_env_wyckoff_universal_df"]], ignore_index=True)
                    atomic_env_binary_df = pd.concat([atomic_env_binary_df, result["atomic_env_binary_df"]], ignore_index=True)
                    atomic_env_ternary_df = pd.concat([atomic_env_ternary_df, result["atomic_env_ternary_df"]], ignore_index=True)
                    coordinate_number_binary_df = pd.concat([coordinate_number_binary_df, result["coordinate_number_binary_df"]], ignore_index=True)
                    coordinate_number_ternary_df = pd.concat([interatomic_ternary_df, result["coordinate_number_ternary_df"]], ignore_index=True)
                    featurizer_log_entries.append(result["featurizer_log"])
                
            except Exception as exc:
                print(f"File {filename} generated an exception: {exc}")
                
    
    # Any post-processing or saving of aggregated results can occur after all files have been processed
    featurizer_log_df = pd.DataFrame(featurizer_log_entries)
    featurizer_log_df = featurizer_log_df.round(3)

    if num_files_processed != 0:
        cols_to_keep = ['CIF_id', 'Compound', 'Central atom']
        click.echo(style(f"Saving csv files in the csv folder", fg="blue"))
        atomic_env_wyckoff_universal_df = df.join_columns_with_comma(atomic_env_wyckoff_universal_df)
        if not coordinate_number_binary_df.empty:
            # Save the original DataFrame to CSV before any modification
            binary_non_numeric_cols_to_remove = coordinate_number_binary_df.select_dtypes(include=['object']).columns.difference(cols_to_keep)
            coordinate_number_binary_df = coordinate_number_binary_df.drop(binary_non_numeric_cols_to_remove, axis=1)                        
            atomic_env_wyckoff_binary_df = df.wyckoff_mapping_to_number_binary(atomic_env_wyckoff_binary_df)
            coordinate_number_binary_avg_df = coordinate_number_binary_df.groupby(cols_to_keep).mean().reset_index()
            coordinate_number_binary_min_df = coordinate_number_binary_df.groupby(cols_to_keep).min().reset_index()
            coordinate_number_binary_max_df = coordinate_number_binary_df.groupby(cols_to_keep).max().reset_index()
            folder.save_to_csv_directory(cif_folder_directory, round_df(coordinate_number_binary_df), "coordination_number_binary_all")
            folder.save_to_csv_directory(cif_folder_directory, round_df(coordinate_number_binary_avg_df), "coordination_number_binary_avg")
            folder.save_to_csv_directory(cif_folder_directory, round_df(coordinate_number_binary_min_df), "coordination_number_binary_min")
            folder.save_to_csv_directory(cif_folder_directory, round_df(coordinate_number_binary_max_df), "coordination_number_binary_max")
            folder.save_to_csv_directory(cif_folder_directory, round_df(interatomic_binary_df), "interatomic_features_binary")
            folder.save_to_csv_directory(cif_folder_directory, round_df(atomic_env_binary_df), "atomic_environment_features_binary")
            folder.save_to_csv_directory(cif_folder_directory, round_df(atomic_env_wyckoff_binary_df), "atomic_environment_wyckoff_multiplicity_features_binary")


        if not coordinate_number_ternary_df.empty:
            ternary_non_numeric_cols_to_remove = coordinate_number_ternary_df.select_dtypes(include=['object']).columns.difference(cols_to_keep)
            coordinate_number_ternary_df = coordinate_number_ternary_df.drop(ternary_non_numeric_cols_to_remove, axis=1)   
            coordinate_number_ternary_avg_df = coordinate_number_ternary_df.groupby(cols_to_keep).mean().reset_index()
            coordinate_number_ternary_min_df = coordinate_number_ternary_df.groupby(cols_to_keep).min().reset_index()
            coordinate_number_ternary_max_df = coordinate_number_ternary_df.groupby(cols_to_keep).max().reset_index()
            atomic_env_wyckoff_ternary_df = df.wyckoff_mapping_to_number_ternary(atomic_env_wyckoff_ternary_df)            
            folder.save_to_csv_directory(cif_folder_directory, round_df(coordinate_number_ternary_df), "coordination_number_ternary_all")
            folder.save_to_csv_directory(cif_folder_directory, round_df(coordinate_number_ternary_avg_df), "coordination_number_ternary_avg")
            folder.save_to_csv_directory(cif_folder_directory, round_df(coordinate_number_ternary_min_df), "coordination_number_ternary_min")
            folder.save_to_csv_directory(cif_folder_directory, round_df(coordinate_number_ternary_max_df), "coordination_number_ternary_max")
            folder.save_to_csv_directory(cif_folder_directory, round_df(interatomic_ternary_df), "interatomic_features_ternary")
            folder.save_to_csv_directory(cif_folder_directory, round_df(atomic_env_ternary_df), "atomic_environment_features_ternary")
            folder.save_to_csv_directory(cif_folder_directory, round_df(atomic_env_wyckoff_ternary_df), "atomic_environment_wyckoff_multiplicity_features_tenary")
 
        
        folder.save_to_csv_directory(cif_folder_directory, round_df(interatomic_universal_df), "interatomic_features_universal")
        folder.save_to_csv_directory(cif_folder_directory, round_df(atomic_env_wyckoff_universal_df), "atomic_environment_wyckoff_multiplicity_features_universal")
        folder.save_to_csv_directory(cif_folder_directory, round_df(featurizer_log_df), "featurizer_log")

if __name__ == "__main__":
    main_parallel()
