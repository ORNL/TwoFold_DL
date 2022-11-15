from rdkit import Chem
import numpy as np

import re

# all punctuation
punctuation_regex  = r"""(\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""

# tokenization regex (Schwaller)
molecule_regex = r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""

def get_token_coords(mol):
    smi = Chem.MolToSmiles(mol)

    # position of atoms in SMILES (not counting punctuation)
    atom_order = [int(s) for s in list(filter(None,re.sub(r'[\[\]]','',mol.GetProp("_smilesAtomOutputOrder")).split(',')))]

    # tokenize the SMILES
    tokens = list(filter(None, re.split(molecule_regex, smi)))

    # remove punctuation
    masked_tokens = [re.sub(punctuation_regex,'',s) for s in tokens]

    k = 0
    token_pos = []
    atom_idx = []
    for i,token in enumerate(masked_tokens):
        if token != '':
            token_pos.append(tuple(mol.GetConformer().GetAtomPosition(atom_order[k])))
            atom_idx.append(atom_order[k])
            k += 1
        else:
            token_pos.append([np.nan, np.nan, np.nan])
            atom_idx.append(None)

    return smi, token_pos, tokens, atom_idx
