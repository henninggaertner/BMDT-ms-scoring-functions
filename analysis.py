import numpy as np
import pandas as pd
from pyteomics import fasta

# Load the FASTA protein data for humans
fasta_df = pd.DataFrame(columns=['protein_id', 'sequence'])
fasta_df_slices = []
for header, sequence in fasta.read("data/uniprotkb_human_proteins_isoforms.fasta"):
    fasta_df_slices.append(pd.DataFrame({'protein_id': [header.split('|')[1]], 'sequence': [sequence]}))

fasta_df = pd.concat(fasta_df_slices)
fasta_df

import re
from pyteomics import mass
from decoy_database import generate_decoy_peptides


def check(str, pattern):
    if re.search(pattern, str):
        return True
    else:
        return False


# Generate / load the peptide database for the human proteome
PEPTIDE_DF_PATH = "data/peptide_df.csv"
try:
    peptide_df = pd.read_csv(PEPTIDE_DF_PATH, index_col=0)
except FileNotFoundError:
    results = generate_decoy_peptides(fasta_df)
    target_peptide_df = results['target_peptides']
    decoy_peptide_df = results['decoy_peptides']
    peptide_df = pd.concat([target_peptide_df, decoy_peptide_df])
    pattern = re.compile('^[ACDEFGHIKLMNPQRSTVWYp]+$')
    peptide_df['peptide_mass'] = -1
    peptide_df['fragments'] = None
    peptide_df = peptide_df[peptide_df['sequence'].apply(lambda x: check(x, pattern))]
    peptide_df['peptide_mass'] = peptide_df['sequence'].apply(lambda x: mass.calculate_mass(x))
    peptide_df.sort_values(by=['peptide_mass'], ascending=True, inplace=True)
    #peptide_df.drop_duplicates(subset=["sequence"], inplace=True)
    peptide_df.to_csv(PEPTIDE_DF_PATH)

from pyteomics import mass, parser
import numpy as np
import re
from functools import lru_cache
from math import comb

aa_comp = dict(mass.std_aa_comp)


def fragments(peptide, types=('b', 'y'), maxcharge=1):
    """
    The function generates all possible m/z for fragments of types
    `types` and of charges from 1 to `maxcharge`.
    """
    parsed_parts = parser.parse(peptide)
    for i in range(1, len(parsed_parts)):
        for ion_type in types:
            for charge in range(1, maxcharge + 1):
                if ion_type[0] in 'abc':
                    yield mass.calculate_mass("".join(parsed_parts[:i]), ion_type=ion_type, charge=charge,
                        aa_comp=aa_comp)
                else:
                    yield mass.calculate_mass("".join(parsed_parts[i:]), ion_type=ion_type, charge=charge,
                        aa_comp=aa_comp)


def calculate_peptide_mass(peptide_mz: float, charge: int) -> float:
    proton_mass = 1.007276466812
    return (peptide_mz * charge) - (proton_mass * charge)


def get_scan_id(text: str) -> int:
    matches = re.findall(r'scan=\d+', text)
    return matches[0].split('=')[1]

from pyteomics import mgf
from tqdm import tqdm

# Load the mass spectrometry raw data from the experiments MGF files
mgf_df = pd.DataFrame(
    columns=['experiment_name', 'precursor_mass', 'precursor_mz', 'mz_array', 'intensity_array, scan_id'])
mgf_df_slices = []

file_list = ["data/new_CTR03_BA46_INSOLUBLE_01.mgf", "data/new_CTR08_BA46_INSOLUBLE_01.mgf",
             "data/new_CTR45_BA46_INSOLUBLE_01.mgf"]
for file in file_list:
    with mgf.read(file) as reader:
        for spectrum in tqdm(reader):
            if spectrum is None or spectrum['params'] is None:
                continue
            params = spectrum['params']
            scan = get_scan_id(params['title'])
            mz_array = spectrum['m/z array']
            intensity_array = spectrum['intensity array']
            charge = int(params['charge'][0])
            precursor_mz = params['pepmass'][0]
            peptide_mass = calculate_peptide_mass(precursor_mz, charge)
            mgf_df_slices.append(pd.DataFrame(
                {'experiment_name': file.split("/")[-1].split(".")[0], 'precursor_mass': [peptide_mass],
                 'precursor_mz': [precursor_mz], 'mz_array': [mz_array], 'intensity_array': [intensity_array],
                 "scan_id": scan}))
mgf_df = pd.concat(mgf_df_slices)
mgf_df




from scoring_function import optimize_q_wrapper
from simple_scoring_function import simple_scoring_function
import pandas as pd
from tqdm import tqdm
import numpy as np
from pyteomics import mass
import multiprocessing as mp
from functools import partial
import concurrent.futures

# Match peptide to experimental spectra
MATCHED_SPECTRA_DF_PATH = "data/matched_spectra_df.csv"

def process_spectrum(spectrum, peptide_df, scoring_function):
    """Process a single spectrum against candidates"""
    # find candidates with ppm of 10
    precursor_mass = spectrum['precursor_mass']
    tolerance = 10 / 1_000_000  # Define the tolerance as a fraction
    candidates = peptide_df[(precursor_mass >= peptide_df['peptide_mass'] * (1 - tolerance)) & (
                precursor_mass <= peptide_df['peptide_mass'] * (1 + tolerance))]

    if len(candidates) == 0:
        return None

    best_match = (-1, "", "")
    for __, candidate in candidates.iterrows():
        sequence = candidate['sequence']
        theoretical_spectrum = np.array(list(fragments(sequence)))
        experimental_mz_array = spectrum['mz_array']
        experimental_intensity_array = spectrum['intensity_array']
        experimental_spectrum = pd.DataFrame(
            {'mz': experimental_mz_array, 'intensity': experimental_intensity_array}).to_numpy()

        if len(theoretical_spectrum) == 0:
            continue

        score = scoring_function(theoretical_spectrum, experimental_spectrum, 20)
        if score > best_match[0]:
            best_match = (score, sequence, candidate['protein_id'])

    if best_match[0] == -1:
        return None

    match_score, sequence, source_protein_id = best_match
    bm = candidates[candidates['sequence'] == sequence]
    peptide_mass = bm['peptide_mass'].values[0]

    return {
        'experiment_name': spectrum['experiment_name'],
        'protein_id': source_protein_id,
        'precursor_mass': spectrum['precursor_mass'],
        'precursor_mz': spectrum['precursor_mz'],
        'mz_array': spectrum['mz_array'],
        'intensity_array': spectrum['intensity_array'],
        'sequence': sequence,
        'peptide_mass': peptide_mass,
        'fragments': theoretical_spectrum,
        'match_score': match_score,
        'scan_id': spectrum['scan_id'],
        'scoring_function': scoring_function.__name__
    }

def run_parallel_analysis(mgf_df, peptide_df, scoring_functions, n_processes=None):
    """Run the analysis in parallel for all scoring functions"""
    if n_processes is None:
        n_processes = mp.cpu_count() - 1  # Leave one CPU for system tasks

    matched_spectra_df_slices = []

    for scoring_function in scoring_functions:
        spectra_to_process = mgf_df.copy()

        print(f"Scoring using {scoring_function.__name__} with {n_processes} processes")

        # Create a partial function with fixed parameters
        process_func = partial(process_spectrum, peptide_df=peptide_df, scoring_function=scoring_function)

        # Process spectra in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_processes) as executor:
            # Map the function to all spectra and track progress with tqdm
            results = list(tqdm(
                executor.map(process_func, [spectrum for _, spectrum in spectra_to_process.iterrows()]),
                total=len(spectra_to_process),
                desc=f"Scoring using {scoring_function.__name__}"
            ))

        # Filter out None results and convert to DataFrame
        valid_results = [r for r in results if r is not None]
        if valid_results:
            batch_df = pd.DataFrame(valid_results)
            matched_spectra_df_slices.append(batch_df)

    if matched_spectra_df_slices:
        matched_spectra_df = pd.concat(matched_spectra_df_slices)

        # Ensure arrays remain as numpy arrays
        matched_spectra_df['intensity_array'] = matched_spectra_df['intensity_array'].apply(lambda x: np.array(x) if not isinstance(x, np.ndarray) else x)
        matched_spectra_df['mz_array'] = matched_spectra_df['mz_array'].apply(lambda x: np.array(x) if not isinstance(x, np.ndarray) else x)
        matched_spectra_df['fragments'] = matched_spectra_df['fragments'].apply(lambda x: np.array(x) if not isinstance(x, np.ndarray) else x)

        # Sort and save
        matched_spectra_df.sort_values(by=['match_score'], ascending=False, inplace=True)
        matched_spectra_df.to_csv(MATCHED_SPECTRA_DF_PATH)
        return matched_spectra_df
    else:
        return pd.DataFrame(
            columns=['experiment_name', 'protein_id', 'precursor_mass', 'precursor_mz', 'mz_array', 'intensity_array',
                   'sequence', 'peptide_mass', 'fragments', 'match_score', "scan_id", "scoring_function"]
        )

# Main execution
if __name__ == "__main__":
    # This ensures proper multiprocessing behavior, especially on Windows
    scoring_functions = [optimize_q_wrapper, simple_scoring_function]
    result_df = run_parallel_analysis(mgf_df, peptide_df, scoring_functions)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming df is already loaded as matched_spectra_df
df = matched_spectra_df.copy()

# 1. Create log2 transformed match scores (handle zero/negative values if they exist)
df['log2_match_score'] = np.log2(df['match_score'])

# Clean infinite values from log transformation
df = df.replace([-np.inf, np.inf], np.nan).dropna(subset=['log2_match_score'])

# 2. Create the facet grid
g = sns.FacetGrid(df, col='experiment_name', row='scoring_function', hue='side',
    palette={'Decoys': 'blue', 'Non-decoys': 'red'}, height=4, aspect=1.5, sharex=True, sharey=True)

# 3. Map KDE plots to the grid
g.map(sns.kdeplot, 'log2_match_score', fill=True, alpha=0.5, linewidth=1.5)

# 4. Customize plot appearance
g.set_axis_labels('log2(Match Score)', 'Density')
g.set_titles(col_template='Experiment: {col_name}', row_template='Scoring: {row_name}')
g.add_legend(title='Category')

# Adjust layout and show
plt.tight_layout()
plt.show()

#%%
import pandas as pd
import plotly.express as px

df = matched_spectra_df

# Load your DataFrame here (replace this with your actual data)
# df = pd.read_csv(...)

# Check if 'protein_id' exists and has no NaN values
if 'protein_id' not in df.columns or df['protein_id'].isnull().all():
    raise ValueError("Column 'protein_id' is missing or contains no valid data.")

# Identify decoys and non-decoys
df['side'] = df['protein_id'].str.contains('XXX_', na=False).map({True: 'Decoys', False: 'Non-decoys'})

# Check if both categories exist
side_counts = df['side'].value_counts()
if 'Decoys' not in side_counts or 'Non-decoys' not in side_counts:
    print("Warning: One of the categories (Decoys/Non-decoys) is missing. Adjusting plot...")

# Ensure 'experiment_name' and 'scoring_function' have valid data
required_columns = ['experiment_name', 'scoring_function', 'match_score']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Required column '{col}' is missing.")

# Generate the violin plot
try:
    fig = px.violin(df, x='scoring_function', y='match_score', color='side', facet_col='experiment_name', box=True,
        points="outliers", title='Match Score Distribution: Decoys (Left) vs. Non-decoys (Right)',
        category_orders={'side': ['Decoys', 'Non-decoys'],  # Explicit order for left/right placement
            'scoring_function': sorted(df['scoring_function'].unique()),  # Ensure consistent order
            'experiment_name': sorted(df['experiment_name'].unique())},
        labels={'match_score': 'Match Score', 'scoring_function': 'Scoring Function'}, log_y=True)

    # Adjust layout to group violins
    fig.update_layout(violinmode='group', violingap=0, violingroupgap=0)

    # Show the plot
    fig.show()

except Exception as e:
    print(f"Plotting failed: {e}")
    # Optionally, display partial data for debugging
    print("\nSample of DataFrame used for plotting:")
    print(df[['experiment_name', 'scoring_function', 'side', 'match_score']].head())
#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming df is already loaded as matched_spectra_df
df = matched_spectra_df.copy()

# 1. Create 'side' column first!
df['side'] = df['protein_id'].str.contains('XXX_', na=False).map({True: 'Decoys', False: 'Non-decoys'})

# 2. Convert match_score to numeric
df['match_score'] = pd.to_numeric(df['match_score'], errors='coerce')
df = df.dropna(subset=['match_score'])
df = df[df['match_score'] > 0]  # Remove zero/negative values

# 3. Apply log2 transformation with protection
df['log2_match_score'] = np.log2(df['match_score'].clip(lower=1e-9))

# 4. Verify 'side' exists
if 'side' not in df.columns:
    raise KeyError("'side' column creation failed - check protein_id values")

# 5. Create facet grid with color mapping
g = sns.FacetGrid(df, col='experiment_name', row='scoring_function', hue='side',
    palette={'Decoys': 'blue', 'Non-decoys': 'red'}, height=4, aspect=1.5, sharex=True, sharey=True)

# 6. Add KDE plots
g.map(sns.kdeplot, 'log2_match_score', fill=True, alpha=0.5, linewidth=1.5)

# 7. Customize labels and titles
g.set_axis_labels('log2(Match Score)', 'Density')
g.set_titles(col_template='Experiment: {col_name}', row_template='Scoring: {row_name}')
g.add_legend(title='Category')

# Adjust layout
plt.tight_layout()
plt.show()
#%%
from matplotlib import pyplot as plt

matched_spectra_df.plot(kind='scatter', x='peptide_mass', y='match_score', s=32, alpha=.8)
plt.gca().spines[['top', 'right', ]].set_visible(False)
#%%
from matplotlib import pyplot as plt

matched_spectra_df['match_score'].plot(kind='hist', bins=20, title='match_score')
plt.gca().spines[['top', 'right', ]].set_visible(False)
#%%
matched_spectra_df.sort_values(by=['match_score'], ascending=False, inplace=True)
confidently_matched_spectra_df = matched_spectra_df[matched_spectra_df['match_score'] > 0.5]
print(
    f"{len(confidently_matched_spectra_df)} spectra could be identified with a score above 0.5 (50% of theoretical peaks matched).")
confidently_matched_spectra_df