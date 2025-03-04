from typing import List, Tuple
import re
from pyteomics import mass, parser

def calculate_peptide_mass(peptide_mz: float, charge: int) -> float:
    """Calculate peptide mass from m/z and charge."""
    proton_mass = 1.007276466812
    return (peptide_mz * charge) - (proton_mass * charge)

def get_scan_id(text: str) -> str:
    """Extract scan ID from spectrum title."""
    matches = re.findall(r'scan=\d+', text)
    return matches[0].split('=')[1]

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
