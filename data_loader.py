from typing import List, Dict, Any
import pandas as pd
from pyteomics import fasta, mgf, mass
import re
from tqdm import tqdm
from .utils import calculate_peptide_mass, get_scan_id
from .decoy_database import generate_decoy_peptides

class DataLoader:
    def load_fasta(self, fasta_path: str) -> pd.DataFrame:
        """Load FASTA protein data into a DataFrame."""
        fasta_df = pd.DataFrame(columns=['protein_id', 'sequence'])
        fasta_df_slices = []
        for header, sequence in fasta.read(fasta_path):
            fasta_df_slices.append(pd.DataFrame({
                'protein_id': [header.split('|')[1]], 
                'sequence': [sequence]
            }))
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
        peptide_df = pd.concat([results['target_peptides'], results['decoy_peptides']])
        peptide_df = self._process_peptide_df(peptide_df)
        peptide_df.to_csv(save_path)
        return peptide_df

    def load_mgf_files(self, file_list: List[str]) -> pd.DataFrame:
        """Load mass spectrometry data from MGF files."""
        mgf_df_slices = []
        for file in file_list:
            with mgf.read(file) as reader:
                mgf_df_slices.extend(self._process_mgf_file(file, reader))
        return pd.concat(mgf_df_slices)

    def _process_mgf_file(self, filename: str, reader: Any) -> List[pd.DataFrame]:
        """Process a single MGF file."""
        slices = []
        for spectrum in tqdm(reader, desc=f"Loading {filename}"):
            if spectrum is None or spectrum['params'] is None:
                continue
            processed = self._process_spectrum(spectrum, filename)
            if processed is not None:
                slices.append(pd.DataFrame([processed]))
        return slices

    def _process_spectrum(self, spectrum: Dict, filename: str) -> Dict[str, Any]:
        """Process a single spectrum from MGF file."""
        params = spectrum['params']
        scan = get_scan_id(params['title'])
        charge = int(params['charge'][0])
        precursor_mz = params['pepmass'][0]
        
        return {
            'experiment_name': filename.split("/")[-1].split(".")[0],
            'precursor_mass': calculate_peptide_mass(precursor_mz, charge),
            'precursor_mz': precursor_mz,
            'mz_array': spectrum['m/z array'],
            'intensity_array': spectrum['intensity array'],
            'scan_id': scan
        }

    @staticmethod
    def _process_peptide_df(df: pd.DataFrame) -> pd.DataFrame:
        """Process and clean peptide DataFrame."""
        pattern = re.compile('^[ACDEFGHIKLMNPQRSTVWYp]+$')
        df = df[df['sequence'].apply(lambda x: bool(pattern.match(x)))]
        df['peptide_mass'] = df['sequence'].apply(mass.calculate_mass)
        df.sort_values(by=['peptide_mass'], ascending=True, inplace=True)
        return df
