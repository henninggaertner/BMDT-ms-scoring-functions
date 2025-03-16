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


def calculate_fdr(df: pd.DataFrame, target_fdr: float = 0.01, filter: bool = False) -> pd.DataFrame:
    """
    Calculate false discovery rate and determine score threshold for desired FDR,
    processing each experiment and scoring function combination separately.

    Args:
        df: DataFrame with matched spectra results
        target_fdr: Target FDR threshold (default: 0.01 or 1%)
        filter: Whether to filter results to those above score threshold

    Returns:
        DataFrame with added FDR columns and filtered to target FDR
    """
    result_dfs = []

    # Group by both experiment and scoring function
    for (experiment, scoring_function), df_slice in df.groupby(['experiment_name', 'scoring_function']):
        # Sort by match score in descending order (better scores first)
        df_slice = df_slice.sort_values(by='match_score', ascending=False).reset_index(drop=True)

        # Calculate cumulative counts within this experiment/scoring function combination
        df_slice['cumulative_targets'] = df_slice['is_target'].cumsum()
        df_slice['cumulative_decoys'] = (~df_slice['is_target']).cumsum()

        # Calculate FDR at each position: FDR = decoys / (decoys + targets)
        df_slice['fdr'] = df_slice['cumulative_decoys'] / (
                df_slice['cumulative_decoys'] + df_slice['cumulative_targets'])

        # Find the threshold where FDR first exceeds target
        valid_fdr = df_slice[df_slice['fdr'] <= target_fdr]
        threshold_idx = valid_fdr.index.max() if not valid_fdr.empty else None

        if threshold_idx is not None:
            # Get threshold values
            score_threshold = df_slice.loc[threshold_idx, 'match_score']
            achieved_fdr = df_slice.loc[threshold_idx, 'fdr']

            # Filter if requested
            if filter:
                filtered_df = df_slice[df_slice['match_score'] >= score_threshold].copy()
            else:
                filtered_df = df_slice.copy()

            # Add threshold metadata
            filtered_df['score_threshold'] = score_threshold
            filtered_df['achieved_fdr'] = achieved_fdr

            result_dfs.append(filtered_df)
        else: # No valid threshold found
            result_dfs.append(df_slice)

    return pd.concat(result_dfs, ignore_index=True) if result_dfs else pd.DataFrame()


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
