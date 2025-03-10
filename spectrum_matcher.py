from typing import Dict, Any, List, Callable
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import generate_fragments

class SpectrumMatcher:
    def __init__(self, peptide_df: pd.DataFrame, tolerance: float = 10e-6):
        self.peptide_df = peptide_df
        self.tolerance = tolerance

    def match_spectrum(self, spectrum: Dict[str, Any], scoring_function: Callable, is_target_db: bool, database_df:
    pd.DataFrame) -> dict[str, Any]:
        """Match a single spectrum against peptide candidates."""
        candidates = self._find_candidates(spectrum['precursor_mass'], database_df=database_df)
        if len(candidates) == 0:
            return None

        best_match = self._find_best_match(spectrum, candidates, scoring_function)
        if best_match[0] == -1:
            return None

        return self._create_match_result(spectrum, best_match, candidates, is_target_db)

    def match_spectra(self, 
                     spectra_df: pd.DataFrame, 
                     scoring_functions: List[Callable]) -> pd.DataFrame:
        """Match spectra using multiple scoring functions sequentially."""
        matched_spectra = []

        for scoring_function in scoring_functions:
            print(f"Scoring using {scoring_function.__name__}")
            for is_target_db, database_df in self.peptide_df.groupby('is_target'):
                for _, spectrum in tqdm(spectra_df.iterrows(), total=len(spectra_df)):
                    result = self.match_spectrum(
                        spectrum, 
                        scoring_function=scoring_function,
                        is_target_db=is_target_db, 
                        database_df=database_df
                    )
                    if result is not None:
                        matched_spectra.append(result)

        if not matched_spectra:
            return pd.DataFrame()

        return self._process_results(pd.DataFrame(matched_spectra))

    def _find_candidates(self, precursor_mass: float, database_df: pd.DataFrame) -> pd.DataFrame:
        """Find peptide candidates within mass tolerance."""
        return database_df[
            (precursor_mass >= database_df['peptide_mass'] * (1 - self.tolerance)) &
            (precursor_mass <= database_df['peptide_mass'] * (1 + self.tolerance))
        ]

    def _find_best_match(self, 
                        spectrum: Dict[str, Any], 
                        candidates: pd.DataFrame, 
                        scoring_function: Callable) -> tuple:
        """Find the best matching peptide for a spectrum."""
        best_match = (-1, "", "")
        
        for _, candidate in candidates.iterrows():
            theoretical_spectrum = np.array(list(generate_fragments(candidate['sequence'])))
            experimental_spectrum = pd.DataFrame({
                'mz': spectrum['mz_array'],
                'intensity': spectrum['intensity_array']
            }).to_numpy()

            if len(theoretical_spectrum) == 0:
                continue

            score = scoring_function(theoretical_spectrum, experimental_spectrum, 20)
            if score > best_match[0]:
                best_match = (score, candidate['sequence'], candidate['protein_id'])

        return best_match

    def _create_match_result(self, 
                           spectrum: Dict[str, Any], 
                           best_match: tuple, 
                           candidates: pd.DataFrame,
                           is_target_db: bool) -> Dict[str, Any]:
        """Create result dictionary for a matched spectrum."""
        match_score, sequence, source_protein_id = best_match
        bm = candidates[candidates['sequence'] == sequence]
        
        return {
            'experiment_name': spectrum['experiment_name'],
            'protein_id': source_protein_id,
            'precursor_mass': spectrum['precursor_mass'],
            'precursor_mz': spectrum['precursor_mz'],
            'mz_array': spectrum['mz_array'],
            'intensity_array': spectrum['intensity_array'],
            'sequence': sequence,
            'peptide_mass': bm['peptide_mass'].values[0],
            'fragments': np.array(list(generate_fragments(sequence))),
            'match_score': match_score,
            'scan_id': spectrum['scan_id'],
            'is_target': is_target_db,
        }

    @staticmethod
    def _process_results(df: pd.DataFrame) -> pd.DataFrame:
        """Process and clean up the results DataFrame."""
        df['intensity_array'] = df['intensity_array'].apply(
            lambda x: np.array(x) if not isinstance(x, np.ndarray) else x
        )
        df['mz_array'] = df['mz_array'].apply(
            lambda x: np.array(x) if not isinstance(x, np.ndarray) else x
        )
        df['fragments'] = df['fragments'].apply(
            lambda x: np.array(x) if not isinstance(x, np.ndarray) else x
        )
        
        return df.sort_values(by=['match_score'], ascending=False)
