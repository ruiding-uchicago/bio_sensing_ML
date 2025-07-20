import os
import pandas as pd
import numpy as np
import json
import re
import warnings
from pathlib import Path
from openpyxl import load_workbook
import xlrd
import multiprocessing
from functools import partial
import copy 
# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module=".*xlrd.*")
warnings.filterwarnings("ignore", category=UserWarning, module=".*openpyxl.*")

# Define the base directory and output directories
base_dir = "."  # Current directory
output_dir = "./analyzed_full"
preprocessed_dir = os.path.join(output_dir, "preprocessed")

# Create output directories
os.makedirs(preprocessed_dir, exist_ok=True)

# Define unit conversion factors and molecular weights
# Molecular weights in g/mol (Da)
molecular_weights = {
    'CFU': None,  # CFU uses Avogadro's number directly
    'IgG': 160000,  # 160 kDa
    'HRP': 44000,   # 44 kDa
    'SP': 42000,    # 42 kDa
}

# Base mass unit conversion factors to grams (used internally for molarity calculation)
base_unit_factors = {
    'pg': 1e-12,
    'ng': 1e-9,
    'ug': 1e-6,
    'µg': 1e-6,
    'mg': 1e-3,
    'g': 1.0,
}

# Avogadro's number
AVOGADRO = 6.022e23

def convert_to_molar(value, unit, target_type):
    """
    Convert concentration to molar (mol/L) based on target type and molecular weight.
    The input value is assumed to be in units per mL (e.g., pg/mL, CFU/mL).
    
    Args:
        value: Numerical concentration value (per mL)
        unit: Unit string (pg, ng, ug, µg, mg, g, CFU)
        target_type: Target molecule type (IgG, HRP, SP, Bacteria, etc.)
    
    Returns:
        Concentration in mol/L (Molar)
    """
    mol_per_ml = 0.0
    if unit == 'CFU':
        # For bacteria: 1 CFU = 1 bacterial cell = 1/Avogadro mol.
        # value is in CFU/mL, so result is in mol/mL.
        mol_per_ml = value / AVOGADRO
    
    elif unit in base_unit_factors:
        # value is in `unit`/mL. Convert to g/mL.
        mass_in_grams_per_ml = value * base_unit_factors[unit]
        
        # Get molecular weight for the target type
        if target_type in molecular_weights and molecular_weights[target_type] is not None:
            mw = molecular_weights[target_type]
            # Convert g/mL to mol/mL: (g/mL) / (g/mol) = mol/mL
            mol_per_ml = mass_in_grams_per_ml / mw
        else:
            # Fallback for unknown target types
            print(f"Warning: Unknown molecular weight for target type '{target_type}', using 0")
            mol_per_ml = 0.0
    
    else:
        # If unit is not recognized
        print(f"Warning: Unknown unit '{unit}', using 0")
        mol_per_ml = 0.0
    
    # Convert mol/mL to mol/L (Molar)
    molar_concentration = mol_per_ml * 1000
    return molar_concentration

def transform_concentration(concentration):
    """
    Apply unified log transformation to concentration value.
    Use log10(1e-19 + concentration) for all values
    This handles zero concentration gracefully while maintaining continuity
    Input: concentration in mol/L
    """
    return np.log10(1e-19 + concentration)

def extract_cycle_data_safe(file_path):
    """Extract all cycle data from an XLS file with multiple fallback methods."""
    cycles_data = []
    
    # Try multiple methods to read the Excel file
    methods = [
        lambda sheet_name: pd.read_excel(file_path, sheet_name=sheet_name, header=None, engine='xlrd'),
        lambda sheet_name: pd.read_excel(file_path, sheet_name=sheet_name, header=None, engine='openpyxl'),
        lambda sheet_name: extract_data_with_xlrd(file_path, sheet_name),
    ]
    
    # Try reading the 'Data' tab (Cycle 1) with each method
    for method in methods:
        try:
            df = method('Data')
            if df is not None and df.shape[1] >= 3:
                df = df.iloc[:, :3]
                df.columns = ['GateI', 'GateV', 'DrainI']
                cycles_data.append(df.to_numpy())
                break
        except Exception as e:
            # Continue to the next method if this one fails
            continue
    
    # If no method worked for the 'Data' tab, return None
    if not cycles_data:
        print(f"Error: Could not read 'Data' tab from {file_path} with any method")
        return None
    
    # Now read Cycles 2-100 with the method that worked for Cycle 1
    successful_method = methods[methods.index(method)]
    for i in range(2, 101):
        try:
            cycle_df = successful_method(f'Cycle{i}')
            if cycle_df is not None and cycle_df.shape[1] >= 3:
                cycle_df = cycle_df.iloc[:, :3]
                cycle_df.columns = ['GateI', 'GateV', 'DrainI']
                cycles_data.append(cycle_df.to_numpy())
            else:
                print(f"Warning: Cycle{i} tab in {file_path} has fewer than 3 columns")
                cycles_data.append(np.zeros_like(cycles_data[0]))
        except Exception as e:
            print(f"Warning: Could not read Cycle{i} from {file_path}, using zeros")
            cycles_data.append(np.zeros_like(cycles_data[0]))
    
    return cycles_data

def extract_data_with_xlrd(file_path, sheet_name):
    """Fallback method using xlrd directly with error suppression."""
    try:
        # Open workbook with error suppression
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            book = xlrd.open_workbook(file_path, logfile=open(os.devnull, 'w'))
            
        # Get sheet by name
        if isinstance(sheet_name, int):
            sheet = book.sheet_by_index(sheet_name)
        else:
            try:
                sheet = book.sheet_by_name(sheet_name)
            except xlrd.XLRDError:
                return None
        
        # Extract data to DataFrame
        data = []
        for i in range(sheet.nrows):
            row_values = sheet.row_values(i)
            data.append(row_values[:3])  # Take first 3 columns
        
        return pd.DataFrame(data)
    except Exception as e:
        return None

def parse_concentration(filename, target_type):
    """Parse concentration value from filename based on target type and convert to mol/L."""
    # Default to 0 if no concentration is found
    concentration = 0.0
    
    # For 0g or 0ng cases
    # Check for explicit zero concentration (with negative lookbehind to ensure it's not part of another number)
    if re.search(r'(?<!\d)0\s*g', filename) or re.search(r'(?<!\d)0\s*ng', filename) or re.search(r'(?<!\d)0\s*pg', filename) or re.search(r'(?<!\d)0\s*CFU', filename):
        return 0.0
    
    # Extract based on target type
    if target_type == 'Bacteria':
        # Pattern for Bacteria (e.g., 80000CFU)
        match = re.search(r'(\d+)CFU', filename)
        if match:
            value = float(match.group(1))
            concentration = convert_to_molar(value, 'CFU', target_type)
    elif target_type in ['IgG', 'SP', 'HRP']:
        # Pattern for IgG, SP, or HRP (e.g., 1ng, 10pg, 1ug, 320 ng)
        match = re.search(r'(\d+)\s*([pnuµm]g)', filename)
        if match:
            value = float(match.group(1))
            unit = match.group(2).lower()
            concentration = convert_to_molar(value, unit, target_type)
    
    return concentration

def extract_conditions_from_filename(filename, folder_name):
    """Extract experimental conditions from the filename and folder name."""
    # Initialize the condition dictionary
    conditions = {
        'Folder': folder_name,
        'Target': '',
        'Setup': '',
        'Time': 0,
        'Concentration': 0.0,
        'Concentration_transformed': 0.0,
        'Original_filename': filename
    }
    
    # Extract Target and Setup from folder name
    if folder_name == "Control":
        conditions['Target'] = "Control"
        conditions['Setup'] = "Control"
        conditions['IsNonTarget'] = False
        # For Control folder, check if it's HRP
        if 'HRP' in filename:
            conditions['Target'] = 'HRP'
    elif "Bacteria HRPAB NonTarget" in folder_name:
        conditions['Target'] = 'Bacteria'
        conditions['Setup'] = 'HRPAB'
        conditions['IsNonTarget'] = True
        # Force concentration to 0 for NonTarget
        concentration = 0.0
        conditions['Concentration'] = concentration
        conditions['Concentration_transformed'] = transform_concentration(concentration)
    elif "IgG Au40 NonTarget" in folder_name:
        conditions['Target'] = 'IgG'
        conditions['Setup'] = 'Au40'
        conditions['IsNonTarget'] = True
        # Force concentration to 0 for NonTarget
        concentration = 0.0
        conditions['Concentration'] = concentration
        conditions['Concentration_transformed'] = transform_concentration(concentration)
    else:
        # Parse folder name to extract target and setup
        folder_parts = folder_name.split()
        if len(folder_parts) >= 1:
            conditions['Target'] = folder_parts[0]
        if len(folder_parts) >= 2:
            conditions['Setup'] = folder_parts[1]
        conditions['IsNonTarget'] = False
    
    # Extract time from filename (e.g., 20min, 20 min)
    time_match = re.search(r'(\d+)\s*min', filename)
    if time_match:
        conditions['Time'] = int(time_match.group(1))
    
    # Extract concentration only if not already set (for non-NonTarget cases)
    if conditions['Concentration'] == 0.0 and not conditions.get('IsNonTarget', False):
        target_type = conditions['Target']
        if target_type:
            concentration = parse_concentration(filename, target_type)
            conditions['Concentration'] = concentration
            conditions['Concentration_transformed'] = transform_concentration(concentration)
    
    return conditions

def normalize_filename(filename):
    """Normalize filename to get a consistent base name without c100/c200 suffix."""
    # Remove c100.xls or c200.xls suffix
    base_name = filename
    if 'c100.xls' in filename:
        base_name = filename.replace('c100.xls', '')
    elif 'c200.xls' in filename:
        base_name = filename.replace('c200.xls', '')
    
    # Strip any trailing or leading whitespace
    base_name = base_name.strip()
    
    return base_name

def create_output_filename(folder_name, base_filename):
    """Create a descriptive output filename based on folder and file names."""
    # Clean folder name for filename
    clean_folder = re.sub(r'[^\w\s-]', '', folder_name).strip()
    clean_folder = re.sub(r'[-\s]+', '_', clean_folder)
    
    # Clean base filename
    clean_base = re.sub(r'[^\w\s-]', '', base_filename).strip()
    clean_base = re.sub(r'[-\s]+', '_', clean_base)
    
    # Combine folder and file information
    output_name = f"{clean_folder}__{clean_base}.json"
    
    return output_name

def process_file_group(base_name, files, folder_path, folder_name, output_folder):
    """Process a group of files for a single experimental condition."""
    combined_data = {
        'conditions': None,
        'cycles': []
    }
    
    # Process c100 files first (cycles 1-100), then c200 files (cycles 101-200)
    c100_file = next((f for f in files if 'c100.xls' in f), None)
    c200_file = next((f for f in files if 'c200.xls' in f), None)
    
    # Process c100 file if it exists
    if c100_file:
        file_path = os.path.join(folder_path, c100_file)
        cycles_data = extract_cycle_data_safe(file_path)
        if cycles_data:
            combined_data['conditions'] = extract_conditions_from_filename(c100_file, folder_name)
            combined_data['cycles'].extend(cycles_data)
    
    # Process c200 file if it exists
    if c200_file:
        file_path = os.path.join(folder_path, c200_file)
        cycles_data = extract_cycle_data_safe(file_path)
        if cycles_data:
            # If conditions are not set yet (no c100 file processed), extract them
            if combined_data['conditions'] is None:
                combined_data['conditions'] = extract_conditions_from_filename(c200_file, folder_name)
            # Append cycles data (these are cycles 101-200)
            combined_data['cycles'].extend(cycles_data)
    
    # Skip if no data was collected
    if not combined_data['cycles'] or combined_data['conditions'] is None:
        print(f"Warning: No valid data found for {base_name}")
        return
    
    # Create JSON-serializable structure
    serializable_data = {
        'conditions': combined_data['conditions'],
        'cycles': [[row for i, row in enumerate(cycle.tolist()) if i > 0] for cycle in combined_data['cycles']]
    }
    
    # Create a descriptive output filename
    output_filename = create_output_filename(folder_name, base_name)
    output_path = os.path.join(output_folder, output_filename)

    with open(output_path, 'w') as f:
        json.dump(serializable_data, f, indent=2)

    print(f"Processed {base_name} from {folder_name} and saved to {output_path}")

def process_folder(folder_path, output_folder):
    """Process all files in a folder using parallel processing."""
    folder_name = os.path.basename(folder_path)
    
    # Get all .xls files in the folder
    xls_files = [f for f in os.listdir(folder_path) if f.endswith('.xls')]
    
    # Group files by base name (without c100/c200 suffix)
    file_groups = {}
    for file in xls_files:
        base_name = normalize_filename(file)
        if base_name not in file_groups:
            file_groups[base_name] = []
        file_groups[base_name].append(file)
    
    # Skip if no valid files were found
    if not file_groups:
        print(f"Skipping folder {folder_name}: No valid Excel files found.")
        return
    
    # Process each group of files in parallel
    pool = multiprocessing.Pool(processes=min(multiprocessing.cpu_count(), len(file_groups)))
    process_func = partial(process_file_group, 
                         folder_path=folder_path, 
                         folder_name=folder_name, 
                         output_folder=output_folder)
    
    # Create a list of arguments for each file group
    args = [(base_name, files) for base_name, files in file_groups.items()]
    
    # Execute in parallel
    pool.starmap(process_func, args)
    pool.close()
    pool.join()

def main():
    """Main function to process all folders with parallel processing."""
    print(f"Starting data preprocessing...")
    print(f"Output directory: {output_dir}")
    print(f"Using {multiprocessing.cpu_count()} CPU cores for parallel processing")
    
    # Process each subfolder sequentially (but files within folders in parallel)
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        
        # Skip non-directories, output directory, and hidden folders
        if not os.path.isdir(folder_path) or folder_name == "analyzed" or folder_name.startswith('.'):
            continue
        
        print(f"Processing folder: {folder_name}")
        
        # All files go to the same preprocessed directory with descriptive names
        process_folder(folder_path, preprocessed_dir)
    
    print(f"Data preprocessing completed.")
    print(f"All processed data saved to: {preprocessed_dir}")

if __name__ == "__main__":
    # This protects against issues with multiprocessing on Windows
    multiprocessing.freeze_support()
    main()