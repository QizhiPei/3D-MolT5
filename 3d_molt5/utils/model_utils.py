import torch
import datasets
from torch.utils.data import DataLoader
from omegaconf import open_dict
from datasets.iterable_dataset import IterableDataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
)

from .FPT5ForConditionalGeneration import FPT5ForConditionalGeneration
from .copied_utils import (
    compute_input_and_target_lengths,
    tokenize_function,
)
from .custom_utils import (
    tokenize_function_seq2desc_fp,
    tokenize_function_desc2seq_fp,
    tokenize_function_fp,
    tokenize_function_fp2selfies,
    DataCollatorForUnimptT5,
    DataCollatorForFPT5Finetune,
    tokenize_function_ft,
    tokenize_function_ft_no_fp,
)
import os
import numpy as np
from .dataset_utils import MixedDataset
from .balance_dataset import BalanceDataset


def get_model(args, config, tokenizer, logger):
    # for finetuning
    if args.model.checkpoint_path:
        model = FPT5ForConditionalGeneration(
            config, args
        )
        model.resize_token_embeddings(len(tokenizer))
        model.load_state_dict(torch.load(args.model.checkpoint_path, map_location='cpu'), strict=True)
        torch.cuda.empty_cache()
        logger.log_message(f"Loaded model from {args.model.checkpoint_path}")

    elif args.model.random_init:
        model = FPT5ForConditionalGeneration(
            config, args
        )
        model.resize_token_embeddings(len(tokenizer))
        logger.log_message(f"Random init")

    else:
        raise NotImplementedError
    
    return model


def get_config(args):
    config = AutoConfig.from_pretrained(
        args.model.name,
    )
    config.dropout_rate = args.model.dropout
    return config


def get_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model.name,
        use_fast=True
    )
    tokenizer.model_max_length = int(1e9)

    
    selfies_dict_list = [line.strip() for line in open(os.path.join(__file__.split('3d_molt5/utils')[0], args.molecule_dict))]
    tokenizer.add_tokens(selfies_dict_list, special_tokens=True)
    
    special_tokens_dict = {'additional_special_tokens': ['<bom>', '<eom>']}
    tokenizer.add_special_tokens(special_tokens_dict, replace_additional_special_tokens=False)

    origin_len = len(tokenizer)
    tokenizer.add_tokens([f'{i}' for i in range(0, 10)], special_tokens=True)
    assert len(tokenizer) == origin_len
    return tokenizer

def get_tokenizer_pred(args, input_tokenizer):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model.name,
        use_fast=True
    )
    tokenizer.model_max_length = int(1e9)

    for i in range(32100, len(input_tokenizer)):
        tokenizer.add_tokens(input_tokenizer.convert_ids_to_tokens(i), special_tokens=False)

    assert len(tokenizer) == len(input_tokenizer)
    for i in range(len(input_tokenizer)):
        assert tokenizer.convert_ids_to_tokens(i) == input_tokenizer.convert_ids_to_tokens(i)
        
    return tokenizer

def smart_load_dataset(path_or_name):
    if os.path.exists(path_or_name):
        return datasets.load_from_disk(path_or_name)
    else:
        return datasets.load_dataset(path_or_name)

def load_dataset_splits(args):
    if args.mode == 'pt':
        raise NotImplementedError

    elif args.mode == 'ft':
        if args.ft_setting == 'specialist':
            dataset_splits = smart_load_dataset(
                args.data.data_dir,
            )
            assert 'train' in dataset_splits.keys() and 'validation' in dataset_splits.keys() and 'test' in dataset_splits.keys()
            if args.test_task == 'pubchem_cap':
                def mol_text_process(example):
                    example_fp = np.array(example['molecule_fp'])
                    pend_fp = np.full((1, example_fp.shape[-1]), -1)
                    if args.enriched_output:
                        return {'src': '<bom>' + example['selfies'] + '<eom>', 'tgt': example['enriched_output'].strip(), 'molecule_fp': np.concatenate([pend_fp, example_fp, pend_fp], axis=0)}
                    else:
                        return {'src': '<bom>' + example['selfies'] + '<eom>', 'tgt': example['output'].strip(), 'molecule_fp': np.concatenate([pend_fp, example_fp, pend_fp], axis=0)}
                
                dataset_splits = dataset_splits.map(mol_text_process, num_proc=8)
                col_remove = ['cid', 'task', 'coord_norm', 'smiles', 'output', 'enriched_output', 'selfies']

            elif args.test_task in ['pqc_prop', 'pubchem_com', 'pubchem_des', 'chebi_cap', 'molnet_cls', 'molnet_reg', 'molinst_qm9', 'lm24_cap']:
                def mol_text_process(example):
                    example_fp = np.array(example['molecule_fp'])
                    pend_fp = np.full((1, example_fp.shape[-1]), -1)
                    return {'src': '<bom>' + example['selfies'] + '<eom>', 'tgt': example['output'].strip(), 'molecule_fp': np.concatenate([pend_fp, example_fp, pend_fp], axis=0)}
                
                dataset_splits = dataset_splits.map(mol_text_process, num_proc=8)
                if args.test_task == 'pqc_prop':
                    col_remove = ['idx_3d', 'task', 'smiles', 'output', 'selfies']
                elif args.test_task == 'chebi_cap':
                    col_remove = ['cid', 'smiles', 'output', 'selfies']
                elif args.test_task == 'lm24_cap':
                    col_remove = ['smiles', 'output', 'selfies']
                elif args.test_task in ['molnet_cls', 'molnet_reg', 'molinst_qm9']:
                    col_remove = ['smiles', 'output', 'selfies']
                else: # pubchem_com and pubchem_des
                    col_remove = ['cid', 'task', 'coord_norm', 'smiles', 'output', 'selfies']
            elif args.test_task in ['chebi_molgen', 'lm24_molgen']:
                def mol_text_process(example): # use chebi caption, reverse the src and tgt
                    example_fp = np.array(example['molecule_fp'])
                    pend_fp = np.full((1, example_fp.shape[-1]), -1)
                    return {'tgt': '<bom>' + example['selfies'] + '<eom>.', 'src': example['output'].strip(), 'molecule_fp': np.concatenate([pend_fp, example_fp, pend_fp], axis=0), 
                            'instruction': 'Generate a molecule that fits the input description.'}
                
                dataset_splits = dataset_splits.map(mol_text_process, num_proc=8)
                if args.test_task == 'chebi_molgen':
                    col_remove = ['cid', 'smiles', 'output', 'selfies']
                elif args.test_task == 'lm24_molgen':
                    col_remove = ['smiles', 'output', 'selfies']
            elif args.test_task == 'molinst_react':
                def mol_text_process(example): # use chebi caption, reverse the src and tgt
                    example_fp = np.array(example['molecule_fp'])
                    pend_fp = np.full((1, example_fp.shape[-1]), -1)
                    return {'src': '<bom>' + example['input'].strip() + '<eom>', 
                            'molecule_fp': np.concatenate([pend_fp, example_fp, pend_fp], axis=0), 
                            'tgt': '<bom>' + example['output'].strip() + '<eom>', 
                            'instruction': example['instruction']}
                
                dataset_splits = dataset_splits.map(mol_text_process, num_proc=8)
                col_remove = ['input', 'output']
            elif args.test_task == 'uspto':
                def mol_text_process(example): # use chebi caption, reverse the src and tgt
                    example_fp = np.array(example['molecule_fp'])
                    pend_fp = np.full((1, example_fp.shape[-1]), -1)
                    inst_molx = "Provide SELFIES strings of possible reactants used in the molecule's synthesis.\nThe reactants should be split by '.'."
                    return {'src': '<bom>' + example['prod_selfies'].strip() + '<eom>', 
                            'molecule_fp': np.concatenate([pend_fp, example_fp, pend_fp], axis=0), 
                            'tgt': '<bom>' + example['rect_selfies'].strip() + '<eom>', 
                            'instruction': inst_molx}
                
                dataset_splits = dataset_splits.map(mol_text_process, num_proc=8)
                col_remove = ['prod_selfies', 'rect_selfies']
            else:
                raise NotImplementedError
            
            if not args.inst_format:
                col_remove.append('instruction')
            dataset_splits = dataset_splits.remove_columns(col_remove)
        elif args.ft_setting == 'generalist':
            def mol_text_process(example):
                example_fp = np.array(example['molecule_fp'])
                pend_fp = np.full((1, example_fp.shape[-1]), -1)
                return {'src': '<bom>' + example['selfies'] + '<eom>', 'tgt': example['output'].strip(), 'molecule_fp': np.concatenate([pend_fp, example_fp, pend_fp], axis=0)}
            dataset_splits_pqc_prop = smart_load_dataset("QizhiPei/e3fp-pubchemqc-prop")
            dataset_splits_pqc_prop = dataset_splits_pqc_prop.remove_columns(['idx_3d', 'task', 'smiles'])
            dataset_splits_pubchem_cap = smart_load_dataset("QizhiPei/e3fp-pubchem-cap")
            dataset_splits_pubchem_cap = dataset_splits_pubchem_cap.remove_columns(['cid', 'task', 'coord_norm', 'smiles', 'enriched_output'])
            dataset_splits_pubchem_des = smart_load_dataset("QizhiPei/e3fp-pubchem-des")
            dataset_splits_pubchem_des = dataset_splits_pubchem_des.remove_columns(['cid', 'task', 'coord_norm', 'smiles'])
            dataset_splits_pubchem_com = smart_load_dataset("QizhiPei/e3fp-pubchem-com")
            dataset_splits_pubchem_com = dataset_splits_pubchem_com.remove_columns(['cid', 'task', 'coord_norm', 'smiles'])
            dataset_splits_list = [dataset_splits_pqc_prop, dataset_splits_pubchem_cap, dataset_splits_pubchem_des, dataset_splits_pubchem_com]
            expected_column_names = ['selfies', 'output', 'molecule_fp', 'instruction']
            # assert all dataset_splits have the same columns
            assert all([set(dataset['train'].column_names) == set(expected_column_names) for dataset in dataset_splits_list])

            dataset_splits_list = [dataset.map(mol_text_process, num_proc=8) for dataset in dataset_splits_list]
            col_remove = ['output', 'selfies']
            if not args.inst_format:
                col_remove.append('instruction')
            dataset_splits_list = [dataset.remove_columns(col_remove) for dataset in dataset_splits_list]
            dataset_splits = {
                'train': [dataset['train'] for dataset in dataset_splits_list],
                'validation': [dataset['validation'] for dataset in dataset_splits_list],
                'test': [dataset['test'] for dataset in dataset_splits_list],
            }
            
        else:
            raise NotImplementedError
        return dataset_splits
    else:
        raise NotImplementedError

    return dataset_splits_text, dataset_splits_molecule, dataset_splits_molecule_fp, dataset_splits_incontext, dataset_splits_mol_text, dataset_splits_pubmed_text


def process_dataset_molecule_3d(dataset_splits, args, tokenizer):
    if args.mode == 'pt':
        final_datasets = {}

        for split, dataset_split in dataset_splits.items():

            # We increase the input_length, because instead of masking tokens T5 replaces
            # masked spans with a single token, therefore to avoid padding we need to have
            # longer sequences at the start, before masking
            before_mask_input_length, target_length = compute_input_and_target_lengths(
                inputs_length=args.data.input_length,
                noise_density=args.data.mlm_probability,
                mean_noise_span_length=args.data.mean_noise_span_length,
            )

            with open_dict(args):
                args.data.before_mask_input_length = before_mask_input_length
                args.data.target_length = target_length

            dataset_split = dataset_split.map(
                tokenize_function_fp,
                batched=True,
                fn_kwargs={
                    'tokenizer': tokenizer,
                    'in_length': before_mask_input_length,
                },
                remove_columns=['text', 'molecule_fp'],
                num_proc=10,
            )

            dataset_split = dataset_split.shuffle(seed=args.seed)
            final_datasets[split] = dataset_split
    elif args.mode == 'ft':
        final_datasets = dataset_splits
    else:
        raise NotImplementedError

    return final_datasets

def process_dataset_molecule_1d(dataset_splits, args, tokenizer):
    if args.mode == 'pt':
        final_datasets = {}

        for split, dataset_split in dataset_splits.items():

            # We increase the input_length, because instead of masking tokens T5 replaces
            # masked spans with a single token, therefore to avoid padding we need to have
            # longer sequences at the start, before masking
            before_mask_input_length, target_length = compute_input_and_target_lengths(
                inputs_length=args.data.input_length,
                noise_density=args.data.mlm_probability,
                mean_noise_span_length=args.data.mean_noise_span_length,
            )

            with open_dict(args):
                args.data.before_mask_input_length = before_mask_input_length
                args.data.target_length = target_length

            dataset_split = dataset_split.map(
                tokenize_function,
                batched=True,
                fn_kwargs={
                    'tokenizer': tokenizer,
                    'in_length': before_mask_input_length,
                },
                remove_columns=['text'],
                num_proc=10,
            )

            dataset_split = dataset_split.shuffle(seed=args.seed)
            final_datasets[split] = dataset_split
    elif args.mode == 'ft':
        final_datasets = dataset_splits
    else:
        raise NotImplementedError

    return final_datasets

def process_dataset(dataset_splits, args, tokenizer):
    if args.mode == 'pt':
        final_datasets = {}

        for split, dataset_split in dataset_splits.items():

            # We increase the input_length, because instead of masking tokens T5 replaces
            # masked spans with a single token, therefore to avoid padding we need to have
            # longer sequences at the start, before masking
            before_mask_input_length, target_length = compute_input_and_target_lengths(
                inputs_length=args.data.input_length,
                noise_density=args.data.mlm_probability,
                mean_noise_span_length=args.data.mean_noise_span_length,
            )

            with open_dict(args):
                args.data.before_mask_input_length = before_mask_input_length
                args.data.target_length = target_length

            dataset_split = dataset_split.map(
                tokenize_function,
                batched=True,
                fn_kwargs={
                    'tokenizer': tokenizer,
                    'in_length': before_mask_input_length,
                },
                remove_columns=['text'],
            )

            dataset_split = dataset_split.shuffle(buffer_size=10_000, seed=args.seed)
            final_datasets[split] = dataset_split
    elif args.mode == 'ft':
        final_datasets = dataset_splits
    else:
        raise NotImplementedError

    return final_datasets

def process_dataset_fp2selfies(dataset_splits, args, tokenizer):
    if args.mode == 'pt':
        final_datasets = {}

        for split, dataset_split in dataset_splits.items():

            dataset_split = dataset_split.map(
                tokenize_function_fp2selfies,
                batched=True,
                fn_kwargs={
                    'tokenizer': tokenizer,
                    'max_length': args.data.input_length,
                },
                remove_columns=['text', 'molecule_fp'],
                num_proc=10,
            )

            dataset_split = dataset_split.shuffle(seed=args.seed)
            final_datasets[split] = dataset_split
    else:
        raise NotImplementedError

    return final_datasets

def process_dataset_seq2desc_fp(dataset_splits, args, tokenizer):
    if args.mode == 'pt':
        final_datasets = {}

        for split, dataset_split in dataset_splits.items():

            dataset_split = dataset_split.map(
                tokenize_function_seq2desc_fp,
                batched=True,
                fn_kwargs={
                    'tokenizer': tokenizer,
                    'max_length': args.data.input_length,
                },
                remove_columns=['seq', 'desc', 'molecule_fp'],
                num_proc=8,
            )

            dataset_split = dataset_split.shuffle(seed=args.seed)
            final_datasets[split] = dataset_split
    else:
        raise NotImplementedError

    return final_datasets

def process_dataset_desc2seq_fp(dataset_splits, args, tokenizer):
    if args.mode == 'pt':
        final_datasets = {}

        for split, dataset_split in dataset_splits.items():

            dataset_split = dataset_split.map(
                tokenize_function_desc2seq_fp,
                batched=True,
                fn_kwargs={
                    'tokenizer': tokenizer,
                    'max_length': args.data.input_length,
                },
                remove_columns=['seq', 'desc', 'molecule_fp'], # this direction doesn't need molecule_fp
                num_proc=8,
            )

            dataset_split = dataset_split.shuffle(seed=args.seed)
            final_datasets[split] = dataset_split
    else:
        raise NotImplementedError

    return final_datasets

def process_dataset_ft(dataset_splits, args, tokenizer):
    assert args.mode == 'ft', "This process function is only for fine-tuning"
    final_datasets = {}

    for split, dataset_split in dataset_splits.items():
        if args.test_task in ['chebi_molgen', 'lm24_molgen', 'molinst_react', 'uspto']:
            tokenize_function_i = tokenize_function_ft_no_fp
        else:
            tokenize_function_i = tokenize_function_ft
        if isinstance(dataset_split, datasets.Dataset):
            dataset_split = dataset_split.map(
                tokenize_function_i,
                batched=True,
                fn_kwargs={
                    'tokenizer': tokenizer,
                    'max_source_length': args.data.max_seq_len,
                    'max_target_length': args.data.max_target_len,
                    'inst_format': args.inst_format,
                },
                remove_columns=['src', 'tgt', 'molecule_fp'] if not args.inst_format else ['src', 'tgt', 'molecule_fp', 'instruction'],
                num_proc=8,
            )
            if split == 'train':
                dataset_split = dataset_split.shuffle(seed=args.seed)
        elif isinstance(dataset_split, list) and isinstance(dataset_split[0], datasets.Dataset):
            dataset_split = [dataset.map(
                tokenize_function_i,
                batched=True,
                fn_kwargs={
                    'tokenizer': tokenizer,
                    'max_source_length': args.data.max_seq_len,
                    'max_target_length': args.data.max_target_len,
                    'inst_format': args.inst_format,
                },
                remove_columns=['src', 'tgt', 'molecule_fp'] if not args.inst_format else ['src', 'tgt', 'molecule_fp', 'instruction'],
                num_proc=8,
            ) for dataset in dataset_split]
            if split == 'train':
                dataset_split = BalanceDataset(dataset_split)
                # dataset_split = dataset_split.shuffle(seed=args.seed) # no shuffle for torch.utils.data.Dataset
            else:
                dataset_split = dataset_split[1] # use pubchem cap as the test set
        else:
            raise NotImplementedError
        final_datasets[split] = dataset_split

    return final_datasets

def get_data_collator(tokenizer, tokenizer_pred, config, args):
    if args.mode == 'pt':
        data_collator = DataCollatorForUnimptT5(
            tokenizer=tokenizer,
            noise_density=args.data.mlm_probability,
            mean_noise_span_length=args.data.mean_noise_span_length,
            input_length=args.data.input_length,
            target_length=args.data.target_length,
            pad_token_id=config.pad_token_id,
            fp_bits=args.fp_bits,
            fp_level=args.fp_level,
            no_fp=args.no_fp,
        )
    elif args.mode == 'ft':
        data_collator = DataCollatorForFPT5Finetune(
            pad_token_id=config.pad_token_id,
            fp_bits=args.fp_bits,
            fp_level=args.fp_level,
            no_fp=args.no_fp,
        )

    else:
        raise NotImplementedError

    return data_collator


def get_dataloaders(tokenizer, tokenizer_pred, config, args):
    if args.mode == 'pt':
        dataset_splits_text, dataset_splits_molecule, dataset_splits_molecule_fp,\
        dataset_splits_incontext, dataset_splits_mol_text, dataset_splits_pubmed_text = load_dataset_splits(args)
        dataset_text = process_dataset(dataset_splits=dataset_splits_text, args=args, tokenizer=tokenizer)
        if not args.debug:
            dataset_molecule = process_dataset_molecule_1d(dataset_splits=dataset_splits_molecule, args=args, tokenizer=tokenizer)
        dataset_molecule_fp = process_dataset_molecule_3d(dataset_splits=dataset_splits_molecule_fp, args=args, tokenizer=tokenizer)
        dataset_fp2selfies = process_dataset_fp2selfies(dataset_splits=dataset_splits_molecule_fp, args=args, tokenizer=tokenizer)
        if not args.debug:
            dataset_incontext = process_dataset(dataset_splits=dataset_splits_incontext, args=args, tokenizer=tokenizer)
        dataset_mol2text = process_dataset_seq2desc_fp(dataset_splits=dataset_splits_mol_text, args=args, tokenizer=tokenizer)
        dataset_text2mol = process_dataset_desc2seq_fp(dataset_splits=dataset_splits_mol_text, args=args, tokenizer=tokenizer)
        dataset_pubmed_text = process_dataset(dataset_splits=dataset_splits_pubmed_text, args=args, tokenizer=tokenizer)
        # dataset_name_selfies = process_dataset(dataset_splits=dataset_splits_name_selfies, args=args, tokenizer=tokenizer)
        data_collator = get_data_collator(tokenizer=tokenizer, tokenizer_pred=tokenizer_pred, config=config, args=args)

        is_iterable = isinstance(dataset_text['train'], IterableDataset)

        dataloaders = {}

        for split in ['train', 'test']:
            batch_size = args.optim.batch_size // args.optim.grad_acc

            if split in ['test']:
                batch_size *= 2

            shuffle = (split == 'train') and not is_iterable

            if args.mode == 'ft' and split == 'train':
                assert shuffle is True
            else:
                assert shuffle is False

            mixed_dataset_split = MixedDataset(dataset_text[split], dataset_molecule[split], dataset_molecule_fp[split], dataset_fp2selfies[split], dataset_incontext[split], dataset_mol2text[split], dataset_text2mol[split], dataset_pubmed_text[split], split=split)

            dataloaders[split] = DataLoader(
                mixed_dataset_split,
                shuffle=shuffle,
                collate_fn=data_collator,
                batch_size=batch_size,
                num_workers=args.data.num_workers,
                pin_memory=True,
                drop_last=False,
            )
    elif args.mode == 'ft':
        dataset_splits = load_dataset_splits(args)
        dataset = process_dataset_ft(dataset_splits=dataset_splits, args=args, tokenizer=tokenizer)
        data_collator = get_data_collator(tokenizer=tokenizer, tokenizer_pred=tokenizer_pred, config=config,
                                        args=args)

        is_iterable = isinstance(dataset['train'], IterableDataset)

        dataloaders = {}

        for split in ['train', 'validation', 'test']:
            batch_size = args.optim.batch_size // args.optim.grad_acc

            if split in ['validation', 'test']:
                batch_size *= args.optim.test_bsz_multi

            shuffle = (split == 'train') and not is_iterable

            if args.mode == 'ft' and split == 'train':
                assert shuffle is True
            else:
                assert shuffle is False

            dataloaders[split] = DataLoader(
                dataset[split],
                shuffle=shuffle,
                collate_fn=data_collator,
                batch_size=batch_size,
                num_workers=args.data.num_workers,
                pin_memory=True,
                drop_last=False,
            )

    # Add & Check args about data loaders
    with open_dict(args):
        if not is_iterable:
            args.data.train_batches = len(dataloaders['train'])
            args.data.validation_batches = len(dataloaders['validation'])
            args.data.test_batches = len(dataloaders['test'])

        args.eval.corrected_steps = args.eval.steps / args.optim.grad_acc
    if args.mode == 'pt':
        return dataloaders['train'], dataloaders['test']
    elif args.mode == 'ft':
        return dataloaders['train'], dataloaders['validation'], dataloaders['test']
    else:
        raise NotImplementedError


def get_optimizer(model, args):
    no_decay = ["bias", "LayerNorm", "layernorm", "layer_norm", "ln"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.optim.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    if args.optim.name == 'adamw':
        from transformers import AdamW
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.optim.base_lr,
        )
    elif args.optim.name == 'adamwscale':
        from .copied_utils import AdamWScale
        optimizer = AdamWScale(
            optimizer_grouped_parameters,
            lr=args.optim.base_lr,
        )
    elif args.optim.name == 'adafactor':
        from transformers import Adafactor
        optimizer = Adafactor(
            optimizer_grouped_parameters,
            lr=args.optim.base_lr,
            relative_step=False,
        )
    else:
        raise NotImplementedError

    return optimizer


def get_lr_scheduler(optimizer, args, logger):
    if args.optim.lr_scheduler == 'cosine':
        from torch.optim.lr_scheduler import (
            SequentialLR,
            LinearLR,
            CosineAnnealingLR,
        )

        scheduler1 = LinearLR(
            optimizer,
            start_factor=0.5,
            end_factor=1,
            total_iters=args.optim.warmup_steps,
            last_epoch=-1,
        )

        scheduler2 = CosineAnnealingLR(
            optimizer,
            T_max=args.optim.total_steps - args.optim.warmup_steps,
            eta_min=args.optim.final_cosine,
        )

        lr_scheduler = SequentialLR(
            optimizer,
            schedulers=[scheduler1, scheduler2],
            milestones=[args.optim.warmup_steps]
        )
    elif args.optim.lr_scheduler == 'legacy':
        import math
        from torch.optim.lr_scheduler import (
            SequentialLR,
            LinearLR,
            LambdaLR,
        )

        msg = "You are using T5 legacy LR Schedule, it's independent from the optim.base_lr"
        logger.log_message(msg)

        num_steps_optimizer1 = math.ceil(args.optim.total_steps * 0.9)
        iters_left_for_optimizer2 = args.optim.total_steps - num_steps_optimizer1

        scheduler1 = LambdaLR(
            optimizer,
            lambda step: min(
                1e-2, 1.0 / math.sqrt(step)
            ) / args.optim.base_lr if step else 1e-2 / args.optim.base_lr
        )

        scheduler2 = LinearLR(
            optimizer,
            start_factor=(
                min(1e-2, 1.0 / math.sqrt(num_steps_optimizer1)) / args.optim.base_lr
            ),
            end_factor=0,
            total_iters=iters_left_for_optimizer2,
            last_epoch=-1,
        )

        lr_scheduler = SequentialLR(
            optimizer,
            schedulers=[scheduler1, scheduler2],
            milestones=[num_steps_optimizer1]
        )
    elif args.optim.lr_scheduler == 'constant':
        from transformers import get_scheduler
        lr_scheduler = get_scheduler(
            name=args.optim.lr_scheduler,
            optimizer=optimizer,
        )
    elif args.optim.lr_scheduler == 'linear':
        from transformers import get_scheduler
        lr_scheduler = get_scheduler(
            name=args.optim.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.optim.warmup_steps,
            num_training_steps=args.optim.total_steps,
        )
    else:
        raise NotImplementedError

    return lr_scheduler
