from typing import Dict, Any, List, Callable
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
from .utils import generate_fragments

class SpectrumMatcher:
    def __init__(self, peptide_df: pd.DataFrame, tolerance: float = 10e-6):
        self.peptide_df = peptide_df
        self.tolerance = tolerance

    def match_spectrum(self, spectrum: Dict[str, Any], scoring_function: Callable) -> Dict[str, Any]:
        """Match a single spectrum against peptide candidates."""
        candidates = self._find_candidates(spectrum['precursor_mass'])
        if len(candidates) == 0:
            return None

        best_match = self._find_best_match(spectrum, candidates, scoring_function)
        if best_match[0] == -1:
            return None

        return self._create_match_result(spectrum, best_match, candidates)

    def match_spectra_parallel(self, 
                             spectra_df: pd.DataFrame, 
                             scoring_functions: List[Callable], 
                             n_processes: int) -> pd.DataFrame:
        """Match multiple spectra in parallel using multiple scoring functions."""
        matched_spectra_df_slices = []

        for scoring_function in scoring_functions:
            process_func = partial(self.match_spectrum, scoring_function=scoring_function)
            
            with ProcessPoolExecutor(max_workers=n_processes) as executor:
                results = list(tqdm(
                    executor.map(process_func, [spectrum for _, spectrum in spectra_df.iterrows()]),
                    total=len(spectra_df),
                    desc=f"Scoring using {scoring_function.__name__}"
                ))

            valid_results = [r for r in results if r is not None]
            if valid_results:
                matched_spectra_df_slices.append(pd.DataFrame(valid_results))

        if not matched_spectra_df_slices:
            return pd.DataFrame()

        return self._process_results(pd.concat(matched_spectra_df_slices))

    def _find_candidates(self, precursor_mass: float) -> pd.DataFrame:
        """Find peptide candidates within mass tolerance."""
        return self.peptide_df[
            (precursor_mass >= self.peptide_df['peptide_mass'] * (1 - self.tolerance)) & 
            (precursor_mass <= self.peptide_df['peptide_mass'] * (1 + self.tolerance))
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
                           candidates: pd.DataFrame) -> Dict[str, Any]:
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
            'scan_id': spectrum['scan_id']
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
