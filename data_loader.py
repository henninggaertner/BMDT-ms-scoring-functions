from typing import List, Dict, Any
import pandas as pd
import os
import pickle
from pyteomics import fasta, mgf, mass
import re
from tqdm import tqdm
from utils import calculate_peptide_mass, get_scan_id
from decoy_database import generate_decoy_peptides
import numpy as np

class DataLoader:
    def load_fasta(self, fasta_path: str) -> pd.DataFrame:
        """Load FASTA protein data into a DataFrame."""
        fasta_df = pd.DataFrame(columns=['protein_id', 'sequence'])
        fasta_df_slices = []
        for header, sequence in fasta.read(fasta_path):
            fasta_df_slices.append(pd.DataFrame({'protein_id': [header.split('|')[1]], 'sequence': [sequence]}))
        return pd.concat(fasta_df_slices)

    def load_or_generate_peptide_df(self, fasta_df: pd.DataFrame, peptide_df_path: str) -> pd.DataFrame:
        """Load existing peptide database or generate a new one."""
        try:
            return pd.read_csv(peptide_df_path, index_col=0)
        except FileNotFoundError:
            return self._generate_peptide_df(fasta_df, peptide_df_path)

    def _generate_peptide_df(self, fasta_df: pd.DataFrame, save_path: str) -> pd.DataFrame:
        """Generate peptide database from FASTA data."""
        results = generate_decoy_peptides(fasta_df)
        target_peptide_df, decoy_peptide_df = results['target_peptides'], results['decoy_peptides']
        target_peptide_df = self._process_peptide_df(target_peptide_df)
        target_peptide_df['is_target'] = True
        decoy_peptide_df = self._process_peptide_df(decoy_peptide_df)
        decoy_peptide_df['is_target'] = False
        peptide_df = pd.concat([target_peptide_df, decoy_peptide_df])
        peptide_df.to_csv(save_path)
        return peptide_df

    def load_mgf_files(self, file_list: List[str], save_path: str) -> pd.DataFrame:
        """Load mass spectrometry data from MGF files."""
        csv_path = save_path
        arrays_path = save_path.replace('.csv', '_arrays.pkl')

        try:
            if os.path.exists(csv_path) and os.path.exists(arrays_path):
                return self._load_mgf_from_saved(csv_path, arrays_path)
        except FileNotFoundError:
            pass

        mgf_df_slices = []
        spectrum_arrays = {}

        for file in file_list:
            with mgf.read(file) as reader:
                file_results = self._process_mgf_file(file, reader, spectrum_arrays)
                mgf_df_slices.extend(file_results)

        mgf_df = pd.concat(mgf_df_slices)
        self._save_mgf_data(mgf_df, csv_path, spectrum_arrays, arrays_path)
        return mgf_df

    def _process_mgf_file(self, filename: str, reader: Any, spectrum_arrays: Dict = None) -> List[pd.DataFrame]:
        """Process a single MGF file."""
        slices = []
        for spectrum in tqdm(reader, desc=f"Loading {filename}"):
            if spectrum is None or spectrum['params'] is None:
                continue
            processed = self._process_spectrum(spectrum, filename, spectrum_arrays)
            if processed is not None:
                slices.append(pd.DataFrame([processed]))
        return slices

    def _process_spectrum(self, spectrum: Dict, filename: str, spectrum_arrays: Dict = None) -> Dict[str, Any]:
        """Process a single spectrum from MGF file."""
        params = spectrum['params']
        scan = get_scan_id(params['title'])
        charge = int(params['charge'][0])
        precursor_mz = params['pepmass'][0]

        experiment_name = filename.split("/")[-1].split(".")[0]
        spectrum_id = f"{experiment_name}_{scan}"

        result = {'experiment_name': experiment_name, 'precursor_mass': calculate_peptide_mass(precursor_mz, charge),
            'precursor_mz': precursor_mz, 'scan_id': scan, 'spectrum_id': spectrum_id  # Add unique identifier
        }

        if spectrum_arrays is not None:
            # Store arrays in memory dict for later saving
            spectrum_arrays[spectrum_id] = {'mz_array': spectrum['m/z array'],
                'intensity_array': spectrum['intensity array']}
        else:
            # Keep arrays in memory if not saving
            result['mz_array'] = spectrum['m/z array']
            result['intensity_array'] = spectrum['intensity array']

        return result

    def _save_mgf_data(self, df: pd.DataFrame, csv_path: str, spectrum_arrays: Dict = None,
                       arrays_path: str = None) -> None:
        """Save MGF data to CSV and arrays to a single pickle file."""
        save_df = df.copy()

        # Remove actual arrays before saving to CSV
        if 'mz_array' in save_df.columns:
            save_df.drop('mz_array', axis=1, inplace=True)
        if 'intensity_array' in save_df.columns:
            save_df.drop('intensity_array', axis=1, inplace=True)

        save_df.to_csv(csv_path, index=False)

        # Save all arrays to a single file
        if spectrum_arrays and arrays_path:
            with open(arrays_path, 'wb') as f:
                pickle.dump(spectrum_arrays, f)

    def _load_mgf_from_saved(self, csv_path: str, arrays_path: str) -> pd.DataFrame:
        """Load MGF data from saved CSV and arrays pickle file."""
        df = pd.read_csv(csv_path)

        # Load the arrays dictionary
        with open(arrays_path, 'rb') as f:
            spectrum_arrays = pickle.load(f)

        # Add arrays back to dataframe as needed
        def get_mz_array(spectrum_id):
            return spectrum_arrays.get(spectrum_id, {}).get('mz_array', None)

        def get_intensity_array(spectrum_id):
            return spectrum_arrays.get(spectrum_id, {}).get('intensity_array', None)

        df['mz_array'] = df['spectrum_id'].apply(get_mz_array)
        df['intensity_array'] = df['spectrum_id'].apply(get_intensity_array)

        return df

    @staticmethod
    def _process_peptide_df(df: pd.DataFrame) -> pd.DataFrame:
        """Process and clean peptide DataFrame."""
        pattern = re.compile('^[ACDEFGHIKLMNPQRSTVWYp]+$')
        df = df[df['sequence'].apply(lambda x: bool(pattern.match(x)))]
        df['peptide_mass'] = df['sequence'].apply(mass.calculate_mass)
        df.sort_values(by=['peptide_mass'], ascending=True, inplace=True)
        return df

    def create_directories(self):
        """Create necessary directories for the analysis."""
        directories = ['data', 'results']
        for directory in directories:
            os.makedirs(directory, exist_ok=True)