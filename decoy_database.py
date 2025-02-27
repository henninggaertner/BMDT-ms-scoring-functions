from random import shuffle

import pandas as pd
from pyteomics import fasta, parser
from tqdm import tqdm


def shuffle_sequence(sequence: str) -> (str):
    sequence = list(sequence)
    shuffle(sequence)
    return ''.join(sequence)


def generate_decoy_peptides(target_database: pd.DataFrame, max_peptide_length=35, min_peptide_length=7,
                            missed_cleavages=0) -> (pd.DataFrame):
    decoy_database = target_database.copy()

    # 1. replace K and R with the preceding amino acid
    # 2. reverse the sequence
    # 3. digest the reversed sequence and the target database with with trypsin
    # 4. if peptide is in the target database, shuffle the sequence max 100 times
    # 5. if peptide is still in the target database, discard it

    def switch_K_R(sequence: str) -> (str):
        sequence = list(sequence)
        for idx, char in enumerate(sequence):
            if char in ['K', 'R']:
                sequence[idx], sequence[idx - 1] = sequence[idx - 1], sequence[idx]
        return ''.join(sequence)

    decoy_database['sequence'] = decoy_database['sequence'].apply(switch_K_R)
    decoy_database['sequence'] = decoy_database['sequence'].apply(lambda x: x[::-1])

    decoy_peptide_database_slices = []
    for index, row in tqdm(decoy_database.iterrows(), total=len(decoy_database), desc="Digesting decoy database"):
        sequence = row['sequence']
        digested_peptides = list(
            parser.cleave(sequence, parser.expasy_rules['trypsin'], missed_cleavages, min_peptide_length,
                          max_peptide_length))
        # numbers
        decoy_peptide_database_slices.append(
            pd.DataFrame({'protein_id': [row['protein_id']] * len(digested_peptides), 'sequence': digested_peptides}))

    decoy_peptide_database = pd.concat(decoy_peptide_database_slices)

    target_peptide_database_slices = []
    for index, row in tqdm(target_database.iterrows(), total=len(target_database), desc="Digesting target database"):
        sequence = row['sequence']
        digested_peptides = list(
            parser.cleave(sequence, parser.expasy_rules['trypsin'], missed_cleavages, min_peptide_length,
                          max_peptide_length))
        # numbers
        target_peptide_database_slices.append(
            pd.DataFrame({'protein_id': [row['protein_id']] * len(digested_peptides), 'sequence': digested_peptides}))

    target_peptide_database = pd.concat(target_peptide_database_slices)

    redundant_peptides_indices = decoy_peptide_database['sequence'].isin(target_peptide_database[ 'sequence'])
    for idx, row in tqdm(decoy_peptide_database[redundant_peptides_indices].iterrows(), desc="Shuffling redundant decoy peptides"):
        attempts = 0
        while row['sequence'] in target_peptide_database['sequence'].values and attempts < 100:
            row['sequence'] = shuffle_sequence(row['sequence'])
            attempts += 1
        if attempts == 100:
            # mark to be deleted
            print("Could not generate decoy peptide")
            row['sequence'] = None


    decoy_peptide_database = decoy_peptide_database.dropna()
    # change protein prefix to XXX_
    decoy_peptide_database['protein_id'] = 'XXX_' + decoy_peptide_database['protein_id']

    return {'target_peptides': target_peptide_database, 'decoy_peptides': decoy_peptide_database}


def generate_decoys():
    fasta_df = pd.DataFrame(columns=['protein_id', 'sequence'])
    fasta_df_slices = []
    for header, sequence in fasta.read("data/uniprotkb_human_proteins.fasta"):
        fasta_df_slices.append(pd.DataFrame({'protein_id': [header.split('|')[1]], 'sequence': [sequence]}))

    fasta_df = pd.concat(fasta_df_slices)
    decoy_peptide_database = generate_decoy_peptides(fasta_df)
    print(f"Length of target database: {len(decoy_peptide_database['target_peptides'])}")
    print(f"Length of decoy database: {len(decoy_peptide_database['decoy_peptides'])}")


if __name__ == '__main__':
   generate_decoys()