from typing import List, Tuple
import re
import pandas as pd
from pyteomics import mass, parser

def calculate_peptide_mass(peptide_mz: float, charge: int) -> float:
    """Calculate peptide mass from m/z and charge."""
    proton_mass = 1.007276466812
    return (peptide_mz * charge) - (proton_mass * charge)

def get_scan_id(text: str) -> str:
    """Extract scan ID from spectrum title."""
    matches = re.findall(r'scan=\d+', text)
    return matches[0].split('=')[1]


def calculate_fdr(df: pd.DataFrame, target_fdr: float = 0.01) -> pd.DataFrame:
    """
    Calculate false discovery rate and determine score threshold for desired FDR.

    Args:
        df: DataFrame with matched spectra results
        target_fdr: Target FDR threshold (default: 0.01 or 1%)

    Returns:
        DataFrame with added FDR column and filtered to target FDR
    """
    result_dfs = []

    # Process each scoring function separately
    for scoring_function, df_slice in df.groupby('scoring_function'):
        # Sort by match score in descending order (better scores first)
        df_slice = df_slice.sort_values(by='match_score', ascending=False).reset_index(drop=True)

        # Count cumulative targets and decoys at each threshold
        df_slice['cumulative_targets'] = df_slice['is_target'].cumsum()
        df_slice['cumulative_decoys'] = (~df_slice['is_target']).cumsum()

        # Calculate FDR at each position using FDR = FP / (FP + TP)
        # where FP = decoys and (FP + TP) = total accepted matches
        df_slice['fdr'] = df_slice['cumulative_decoys'] / (
                    df_slice['cumulative_decoys'] + df_slice['cumulative_targets'])

        # Find the threshold where FDR is closest to but not exceeding target_fdr
        threshold_idx = df_slice[df_slice['fdr'] <= target_fdr].index.max()

        if threshold_idx is not None:
            # Get the score threshold
            score_threshold = df_slice.loc[threshold_idx, 'match_score']

            # Filter dataset to keep only results above the threshold
            filtered_df = df_slice[df_slice['match_score'] >= score_threshold].copy()

            # Add threshold information
            filtered_df['score_threshold'] = score_threshold
            filtered_df['achieved_fdr'] = filtered_df.loc[threshold_idx, 'fdr']

            result_dfs.append(filtered_df)

    # Combine results from all scoring functions
    if result_dfs:
        return pd.concat(result_dfs, ignore_index=True)
    else:
        return pd.DataFrame()


def generate_fragments(peptide: str, types: Tuple[str, ...]=('b', 'y'), maxcharge: int=1) -> List[float]:
    """Generate theoretical fragment masses for a peptide."""
    aa_comp = dict(mass.std_aa_comp)
    parsed_parts = parser.parse(peptide)
    fragments = []

    for i in range(1, len(parsed_parts)):
        for ion_type in types:
            for charge in range(1, maxcharge + 1):
                if ion_type[0] in 'abc':
                    peptide_part = "".join(parsed_parts[:i])
                else:
                    peptide_part = "".join(parsed_parts[i:])
                
                fragments.append(mass.calculate_mass(
                    peptide_part, 
                    ion_type=ion_type, 
                    charge=charge,
                    aa_comp=aa_comp
                ))
    
    return fragments
