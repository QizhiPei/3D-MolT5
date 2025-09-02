import math
import torch
from torch.utils.data import get_worker_info
from datasets.iterable_dataset import IterableDataset

class MixedDataset(IterableDataset):
    def __init__(self, dataset_text, dataset_molecule, dataset_molecule_fp, dataset_fp2selfies, dataset_incontext, dataset_mol2text, dataset_text2mol, dataset_pubmed_text, split='train'):
        super().__init__(self, split=split)
        if split == 'train':
            self.dataset_text = dataset_text
            self.dataset_molecule = dataset_molecule
            self.dataset_molecule_fp = dataset_molecule_fp
            self.dataset_fp2selfies = dataset_fp2selfies
            self.dataset_incontext = dataset_incontext
            self.dataset_mol2text = dataset_mol2text
            self.dataset_text2mol = dataset_text2mol
            self.dataset_pubmed_text = dataset_pubmed_text
        elif split == 'test':
            self.dataset_mol2text = dataset_mol2text
        else:
            raise NotImplementedError
    
    def __iter__(self):
        if self.split == 'train':
            worker_info = get_worker_info()
            if worker_info is None:
                text_iter = iter(self.dataset_text)
                molecule_iter = iter(self.dataset_molecule)
                molecule_fp_iter = iter(self.dataset_molecule_fp)
                fp2selfies_iter = iter(self.dataset_fp2selfies)
                incontext_iter = iter(self.dataset_incontext)
                mol2text_iter = iter(self.dataset_mol2text)
                text2mol_iter = iter(self.dataset_text2mol)
                pubmed_text_iter = iter(self.dataset_pubmed_text)

                while True:
                    try:
                        text_batch = next(text_iter)
                    except StopIteration:
                        text_iter = iter(self.dataset_text)
                        text_batch = next(text_iter)

                    try:
                        molecule_batch = next(molecule_iter)
                    except StopIteration:
                        molecule_iter = iter(self.dataset_molecule)
                        molecule_batch = next(molecule_iter)

                    try:
                        molecule_fp_batch = next(molecule_fp_iter)
                    except StopIteration:
                        molecule_fp_iter = iter(self.dataset_molecule_fp)
                        molecule_fp_batch = next(molecule_fp_iter)

                    try:
                        fp2selfies_batch = next(fp2selfies_iter)
                    except StopIteration:
                        fp2selfies_iter = iter(self.dataset_fp2selfies)
                        fp2selfies_batch = next(fp2selfies_iter)
                    
                    try:
                        incontext_batch = next(incontext_iter)
                    except StopIteration:
                        incontext_iter = iter(self.dataset_incontext)
                        incontext_batch = next(incontext_iter)
                    
                    try:
                        mol2text_batch = next(mol2text_iter)
                    except StopIteration:
                        mol2text_iter = iter(self.dataset_mol2text)
                        mol2text_batch = next(mol2text_iter)

                    try:
                        text2mol_batch = next(text2mol_iter)
                    except StopIteration:
                        text2mol_iter = iter(self.dataset_text2mol)
                        text2mol_batch = next(text2mol_iter)

                    try:
                        pubmed_text_batch = next(pubmed_text_iter)
                    except StopIteration:
                        pubmed_text_iter = iter(self.dataset_pubmed_text)
                        pubmed_text_batch = next(pubmed_text_iter) 

                    # Due to the multiple workers, the data in batch may be in random order
                    yield text_batch, molecule_batch, molecule_fp_batch, fp2selfies_batch, incontext_batch, mol2text_batch, text2mol_batch, pubmed_text_batch
            else:
                worker_id = worker_info.id
                if worker_id % 8 == 0:
                    text_iter = iter(self.dataset_text)
                    while True:
                        try:
                            text_batch = next(text_iter)
                        except StopIteration:
                            text_iter = iter(self.dataset_text)
                            text_batch = next(text_iter)
                        yield text_batch
                elif worker_id % 8 == 1:
                    molecule_iter = iter(self.dataset_molecule)
                    while True:
                        try:
                            molecule_batch = next(molecule_iter)
                        except StopIteration:
                            molecule_iter = iter(self.dataset_molecule)
                            molecule_batch = next(molecule_iter)
                        yield molecule_batch
                elif worker_id % 8 == 2:
                    molecule_fp_iter = iter(self.dataset_molecule_fp)
                    while True:
                        try:
                            molecule_fp_batch = next(molecule_fp_iter)
                        except StopIteration:
                            molecule_fp_iter = iter(self.dataset_molecule_fp)
                            molecule_fp_batch = next(molecule_fp_iter)
                        yield molecule_fp_batch
                elif worker_id % 8 == 3:
                    fp2selfies_iter = iter(self.dataset_fp2selfies)
                    while True:
                        try:
                            fp2selfies_batch = next(fp2selfies_iter)
                        except StopIteration:
                            fp2selfies_iter = iter(self.dataset_fp2selfies)
                            fp2selfies_batch = next(fp2selfies_iter)
                        yield fp2selfies_batch
                elif worker_id % 8 == 4:
                    incontext_iter = iter(self.dataset_incontext)
                    while True:
                        try:
                            incontext_batch = next(incontext_iter)
                        except StopIteration:
                            incontext_iter = iter(self.dataset_incontext)
                            incontext_batch = next(incontext_iter)
                        yield incontext_batch
                elif worker_id % 8 == 5:
                    mol2text_iter = iter(self.dataset_mol2text)
                    while True:
                        try:
                            mol2text_batch = next(mol2text_iter)
                        except StopIteration:
                            mol2text_iter = iter(self.dataset_mol2text)
                            mol2text_batch = next(mol2text_iter)
                        yield mol2text_batch
                elif worker_id % 8 == 6:
                    text2mol_iter = iter(self.dataset_text2mol)
                    while True:
                        try:
                            text2mol_batch = next(text2mol_iter)
                        except StopIteration:
                            text2mol_iter = iter(self.dataset_text2mol)
                            text2mol_batch = next(text2mol_iter)
                        yield text2mol_batch
                elif worker_id % 8 == 7:
                    pubmed_text_iter = iter(self.dataset_pubmed_text)
                    while True:
                        try:
                            pubmed_text_batch = next(pubmed_text_iter)
                        except StopIteration:
                            pubmed_text_iter = iter(self.dataset_pubmed_text)
                            pubmed_text_batch = next(pubmed_text_iter)
                        yield pubmed_text_batch
                    
        elif self.split == 'test':
            mol2text_start = 0
            mol2text_end = len(self.dataset_mol2text)
            worker_info = get_worker_info()
            if worker_info is None:  # single-process data loading, return the full iterator
                iter_start = mol2text_start
                iter_end = mol2text_end
            else:  # in a worker process
                # split workload
                per_worker = int(math.ceil((mol2text_end - mol2text_start) / float(worker_info.num_workers)))
                worker_id = worker_info.id
                iter_start = mol2text_start + worker_id * per_worker
                iter_end = min(iter_start + per_worker, mol2text_end)
            mol2text_iter = iter(self.dataset_mol2text.select(range(iter_start,iter_end)))
            for mol2text_batch in mol2text_iter:
                yield mol2text_batch

        else:
            raise NotImplementedError
