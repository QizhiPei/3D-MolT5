import logging
logging.basicConfig(level=logging.WARNING)
from transformers import AutoTokenizer
from rdkit import Chem
from rdkit.Geometry import Point3D
import numpy as np
import selfies as sf
from e3fp.pipeline import fprints_from_mol_verbose
from e3fp.fingerprint.fprinter import signed_to_unsigned_int

def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        'google/t5-v1_1-base',
        use_fast=True
    )
    tokenizer.model_max_length = int(1e9)

    selfies_dict_list = [line.strip() for line in open('../dict/selfies_dict.txt')]
    tokenizer.add_tokens(selfies_dict_list, special_tokens=True)
    
    special_tokens_dict = {'additional_special_tokens': ['<bom>', '<eom>']}
    tokenizer.add_special_tokens(special_tokens_dict, replace_additional_special_tokens=False)

    origin_len = len(tokenizer)
    tokenizer.add_tokens([f'{i}' for i in range(0, 10)], special_tokens=True)
    assert len(tokenizer) == origin_len

    return tokenizer

t5_selfies_tokenizer = get_tokenizer()

def identifier_to_bit(identifier: int):
    return signed_to_unsigned_int(identifier) % fprint_params['bits']

def get_num_atoms_woH(mol):
    return np.array([x.GetIdx() for x in mol.GetAtoms() if x.GetAtomicNum() > 1]).shape[0]

def get_num_atoms_wH(mol):
    return np.array([x.GetIdx() for x in mol.GetAtoms()]).shape[0]

def check_identifier_in_fprints_list(fprints_list, fingerprinter, double_check=False):
    fingerprinter_modify_dict = {}
    count = 0
    for i in range(len(fingerprinter.all_shells)): # fingerprinter.level_shells[fprint_params['level']] have been merged
        shell_i = fingerprinter.all_shells[i]
        if identifier_to_bit(shell_i.identifier) not in fprints_list[0].indices:
            flag = 0
            for shell_j in fingerprinter.all_shells:
                if shell_i.substruct == shell_j.substruct and shell_i.identifier != shell_j.identifier and identifier_to_bit(shell_j.identifier) in fprints_list[0].indices:
                    flag = 1
                    # print('to be merged: ', identifier_to_bit(shell_i.identifier), identifier_to_bit(shell_j.identifier))
                    fingerprinter_modify_dict[i] = shell_j.identifier
                    break
            if flag:
                continue
            print('not merged and not in fprints_list[0].indices: ', identifier_to_bit(shell_i.identifier))
            print(shell_i)
            count += 1
    if count > 0:
        print('can not be merged count: ', count)

    # merge the identifier
    for k, v in fingerprinter_modify_dict.items():
        fingerprinter.all_shells[k].identifier = v

    if double_check:
        for shell in fingerprinter.all_shells:
            fp_i = identifier_to_bit(shell.identifier)
            if fp_i not in fprints_list[0].indices:
                raise ValueError('identifier not in fprints_list[0].indices')

def check_all_shells(fingerprinter, mol, level):
    num_atom = get_num_atoms_woH(mol)
    print('num_atom: ', num_atom)

def all_shell_identifier_to_fp(fingerprinter, mol, level):
    num_atom = get_num_atoms_wH(mol)
    fprints_all_atom = -1 * np.ones((num_atom, level + 1), dtype=np.int32)
    fp_num_atom = len(fingerprinter.all_shells) // len(fingerprinter.level_shells.keys())
    for i, shell in enumerate(fingerprinter.all_shells):
        fp_i = identifier_to_bit(shell.identifier)
        shell.fp = fp_i
        fprints_all_atom[shell.center_atom, i // fp_num_atom] = fp_i
    return fprints_all_atom

def get_atoms_woH(smi):
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.RemoveHs(mol)
    atoms = [x.GetSymbol() for x in mol.GetAtoms()]
    return atoms

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--fp-bits', type=int, default=4096)
parser.add_argument('--fp-level', type=int, default=3)
args = parser.parse_args()
print('fp_bits:', args.fp_bits, 'fp_level:', args.fp_level)
fprint_params = {'bits': args.fp_bits, 'rdkit_invariants': True, 'level': args.fp_level, 'all_iters': True, 'exclude_floating': False}

def data_process(sdf_path):
    suppl = Chem.SDMolSupplier(sdf_path)
    mol = suppl[0]

    data_i_smiles = Chem.MolToSmiles(mol, kekuleSmiles=True)
    m_order = list(mol.GetPropsAsDict(includePrivate=True, includeComputed=True)['_smilesAtomOutputOrder'])
    mol = Chem.RenumberAtoms(mol, m_order)

    try:
        sfi, attr = sf.encoder(data_i_smiles, attribute=True)
    except:
        try:
            sfi, attr = sf.encoder(data_i_smiles, attribute=True, strict=False)
        except:
            return None, None, None
    
    sfi_t5_tokenized = t5_selfies_tokenizer.tokenize(sfi)
    sfi_tokenized = list(sf.split_selfies(sfi))
    err_return_fprints = -1 * np.ones((len(sfi_t5_tokenized), fprint_params['level'] + 1), dtype=np.int32)
    if sfi_t5_tokenized != sfi_tokenized:
        print('t5_selfies_tokenizer.tokenize(sfi) != list(sf.split_selfies(sfi))')
        return None, None, None

    _, attr_rev = sf.decoder(sfi, attribute=True)

    try:
        all_rev_indexes = [item[-1].index for item in [item.attribution for item in attr_rev]]
    except:
        return err_return_fprints, sfi, data_i_smiles
    
    all_rev_indexes_no_repeat = list(dict.fromkeys(all_rev_indexes))

    if len(all_rev_indexes_no_repeat) != mol.GetNumAtoms():
        print('len(all_rev_indexes_no_repeat) != mol.GetNumAtoms()')
        return err_return_fprints, sfi

    mol.SetProp('_Name', str(sdf_path.split('/')[-1]))
    coords = mol.GetConformer().GetPositions()
    coords = coords - coords.mean(axis=0)

    conf = mol.GetConformer()
    conf.Set3D(True)
    if mol.GetNumAtoms() != len(coords):
        print('unequal atom and coords/atom in smi')
        return err_return_fprints, sfi, data_i_smiles
        
    for i in range(mol.GetNumAtoms()):
        x, y, z = coords[i]
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    try:
        fprints_list, fingerprinter = fprints_from_mol_verbose(mol, fprint_params=fprint_params)
        check_identifier_in_fprints_list(fprints_list, fingerprinter, double_check=False)
        
        fprints_all_atom = all_shell_identifier_to_fp(fingerprinter, mol, fprint_params['level'])
        fprints_selfies = -1 * np.ones((len(sfi_t5_tokenized), fprint_params['level'] + 1), dtype=np.int32)
        
        for i, sfi_idx in enumerate(all_rev_indexes_no_repeat):
            fprints_selfies[sfi_idx] = fprints_all_atom[i]
        return fprints_selfies, sfi, data_i_smiles
    except ValueError as e:
        print(e)
        return err_return_fprints, sfi, data_i_smiles

fp, sfi, smi = data_process('example.sdf')

print(f"E3FP Fingerprint Shape: {fp.shape}\n")
print(f"E3FP Fingerprint:\n {fp}")
print('='*100)
print(f"SELFIES: {sfi}")
print('='*100)
print(f"SMILES: {smi}")