from typing import Dict, List
import numpy as np
from transformers import BatchEncoding
from dataclasses import dataclass
from transformers import AutoTokenizer
import torch

# from transformers.utils import logging
# logger = logging.get_logger(__name__)

@dataclass
class DataCollatorForUnimptT5:
    """
    [Copied from https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_t5_mlm_flax.py]
    Data collator used for T5 span-masked language modeling.
    It is made sure that after masking the inputs are of length `data_args.max_seq_length` and targets are also of fixed length.
    For more information on how T5 span-masked language modeling works, one can take a look
    at the `official paper <https://arxiv.org/pdf/1910.10683.pdf>`__
    or the `official code for preprocessing <https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py>`__ .
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        noise_density (:obj:`float`):
            The probability with which to (randomly) mask tokens in the input.
        mean_noise_span_length (:obj:`float`):
            The average span length of the masked tokens.
        input_length (:obj:`int`):
            The expected input length after masking.
        target_length (:obj:`int`):
            The expected target length after masking.
        pad_token_id: (:obj:`int`):
            The pad token id of the model
        decoder_start_token_id: (:obj:`int):
            The decoder start token id of the model
    """

    tokenizer: AutoTokenizer
    noise_density: float
    mean_noise_span_length: float
    input_length: int
    target_length: int
    pad_token_id: int
    fp_bits: int
    fp_level: int
    no_fp: bool

    def __call__(self, examples: List[Dict[str, np.ndarray]]) -> BatchEncoding:
        # convert list to dict and tensorize input
        # flatten list of dicts in tuple to list of dicts
        if isinstance(examples[0], tuple):
            examples = [sample_dict for sample_tuple in examples for sample_dict in sample_tuple]

        examples_mlm = []
        examples_src_tgt = []
        for example_i in examples:
            if "labels" in example_i:
                examples_src_tgt.append(example_i)
            else:
                examples_mlm.append(example_i)
        if len(examples_mlm) > 0 and len(examples_src_tgt) > 0:
            batch = BatchEncoding(
                {
                    k: np.array([examples_mlm[i][k] for i in range(len(examples_mlm))])
                    for k, v in examples_mlm[0].items()
                }
            )

            input_ids = batch["input_ids"]
            batch_size, expandend_input_length = input_ids.shape

            mask_indices = np.asarray(
                [
                    self.random_spans_noise_mask(expandend_input_length)
                    for i in range(batch_size)
                ]
            )
            labels_mask = ~mask_indices

            input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
            labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))

            batch["input_ids"] = self.filter_input_ids(input_ids, input_ids_sentinel)
            batch["labels"] = self.filter_input_ids(input_ids, labels_sentinel)

            if batch["input_ids"].shape[-1] != self.input_length:
                raise ValueError(
                    f"`input_ids` are incorrectly preprocessed. `input_ids` length is {batch['input_ids'].shape[-1]}, but"
                    f" should be {self.input_length}."
                )

            if batch["labels"].shape[-1] != self.target_length:
                raise ValueError(
                    f"`labels` are incorrectly preprocessed. `labels` length is {batch['labels'].shape[-1]}, but should be"
                    f" {self.target_length}."
                )

            batch_src_tgt = BatchEncoding(
                {
                    k: np.array([examples_src_tgt[i][k] for i in range(len(examples_src_tgt))])
                    for k, v in examples_src_tgt[0].items()
                }
            )
            batch_src_tgt['labels'][batch_src_tgt['labels'] == self.pad_token_id] = -100
            # pad batch['labels'] to the same the batch['input_ids']
            batch['labels'] = np.concatenate((batch['labels'], np.full((batch_size, self.input_length - self.target_length), -100)), axis=1)

            batch = {k: np.concatenate((batch[k], batch_src_tgt[k]), axis=0) for k in batch}

            batch = {k: torch.from_numpy(v) for k, v in batch.items()}
        
        elif len(examples_mlm) == 0 and len(examples_src_tgt) > 0:
            batch = BatchEncoding(
                {
                    k: np.array([examples_src_tgt[i][k] for i in range(len(examples_src_tgt))])
                    for k, v in examples_src_tgt[0].items()
                }
            )

            batch['labels'][batch['labels'] == self.pad_token_id] = -100

            batch = {k: torch.from_numpy(v) for k, v in batch.items()}

        elif len(examples_mlm) > 0 and len(examples_src_tgt) == 0:
            batch = BatchEncoding(
                {
                    k: np.array([examples_mlm[i][k] for i in range(len(examples_mlm))])
                    for k, v in examples_mlm[0].items()
                }
            )

            input_ids = batch["input_ids"]
            batch_size, expandend_input_length = input_ids.shape

            mask_indices = np.asarray(
                [
                    self.random_spans_noise_mask(expandend_input_length)
                    for i in range(batch_size)
                ]
            )
            labels_mask = ~mask_indices

            input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
            labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))

            # if "molecule_fp_ids" in batch:
            #     print('yes')
            batch["input_ids"] = self.filter_input_ids(input_ids, input_ids_sentinel)
            batch["labels"] = self.filter_input_ids(input_ids, labels_sentinel)

            if "molecule_fp_ids" in batch:
                molecule_fp_ids = batch["molecule_fp_ids"]
                batch["molecule_fp_ids"] = self.filter_molecule_fp_ids(molecule_fp_ids, input_ids_sentinel)
                if batch["molecule_fp_ids"].shape[-2] != self.input_length:
                    raise ValueError(
                        f"`input_ids` are incorrectly preprocessed. `molecule_fp_ids` length is {batch['molecule_fp_ids'].shape[-1]}, but"
                        f" should be {self.input_length}."
                    )

            if batch["input_ids"].shape[-1] != self.input_length:
                raise ValueError(
                    f"`input_ids` are incorrectly preprocessed. `input_ids` length is {batch['input_ids'].shape[-1]}, but"
                    f" should be {self.input_length}."
                )

            if batch["labels"].shape[-1] != self.target_length:
                raise ValueError(
                    f"`labels` are incorrectly preprocessed. `labels` length is {batch['labels'].shape[-1]}, but should be"
                    f" {self.target_length}."
                )
            batch['labels'] = np.concatenate((batch['labels'], np.full((batch_size, self.input_length - self.target_length), -100)), axis=1)
            batch = {k: torch.from_numpy(v) for k, v in batch.items()}
            
        if "molecule_fp_ids" not in batch:
            batch["molecule_fp_ids"] = torch.full((batch["input_ids"].shape[0], batch["input_ids"].shape[1], self.fp_level + 1), -1, dtype=torch.int32)
        if self.no_fp:
            batch["molecule_fp_ids"] = torch.full((batch["input_ids"].shape[0], batch["input_ids"].shape[1], self.fp_level + 1), -1, dtype=torch.int32)
        return batch

    def create_sentinel_ids(self, mask_indices):
        """
        Sentinel ids creation given the indices that should be masked.
        The start indices of each mask are replaced by the sentinel ids in increasing
        order. Consecutive mask indices to be deleted are replaced with `-1`.
        """
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(
            start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices
        )
        sentinel_ids = np.where(
            sentinel_ids != 0, (self.tokenizer.vocab_size - sentinel_ids), 0
        )
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def filter_input_ids(self, input_ids, sentinel_ids):
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """
        batch_size = input_ids.shape[0]

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed
        input_ids = input_ids_full[input_ids_full >= 0].reshape((batch_size, -1))
        input_ids = np.concatenate(
            [
                input_ids,
                np.full((batch_size, 1), self.tokenizer.eos_token_id, dtype=np.int32),
            ],
            axis=-1,
        )
        return input_ids

    def filter_molecule_fp_ids(self, molecule_fp_ids, sentinel_ids):
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """
        batch_size = molecule_fp_ids.shape[0]
        fp_level = molecule_fp_ids.shape[-1]
        # replace the -1 in molecule_fp_ids with a special token id (larger than fp_bits)
        molecule_fp_ids = np.where(molecule_fp_ids == -1, 99999, molecule_fp_ids)
        sentinel_ids_expand = np.repeat(sentinel_ids, fp_level, axis=-1).reshape((batch_size, -1, fp_level))
        molecule_fp_ids_full = np.where(sentinel_ids_expand != 0, sentinel_ids_expand, molecule_fp_ids)
        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed
        molecule_fp_ids = molecule_fp_ids_full[molecule_fp_ids_full >= 0].reshape((batch_size, -1, fp_level))
        molecule_fp_ids = np.concatenate(
            [
                molecule_fp_ids,
                np.full((batch_size, 1, fp_level), -1, dtype=np.int32),
            ],
            axis=-2,
        )
        # set molecule_fp_ids to -1 if it is greater than self.fp_bits for sentinel tokens
        molecule_fp_ids = np.where(molecule_fp_ids >= self.fp_bits, -1, molecule_fp_ids)
        return molecule_fp_ids

    def random_spans_noise_mask(self, length):
        """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .

        Noise mask consisting of random spans of noise tokens.
        The number of noise tokens and the number of noise spans and non-noise spans
        are determined deterministically as follows:
        num_noise_tokens = round(length * noise_density)
        num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
        Spans alternate between non-noise and noise, beginning with non-noise.
        Subject to the above restrictions, all masks are equally likely.

        Args:
            length: an int32 scalar (length of the incoming token sequence)
            noise_density: a float - approximate density of output mask
            mean_noise_span_length: a number

        Returns:
            a boolean tensor with shape [length]
        """

        orig_length = length

        num_noise_tokens = int(np.round(length * self.noise_density))
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = int(np.round(num_noise_tokens / self.mean_noise_span_length))

        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            """Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0
                num_segments: an integer scalar in [1, num_items]
            Returns:
                a Tensor with shape [num_segments] containing positive integers that add
                up to num_items
            """
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(
            num_nonnoise_tokens, num_noise_spans
        )

        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
            [num_noise_spans * 2],
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length,), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)

        return is_noise[:orig_length]

def tokenize_function_seq2desc_fp(examples, tokenizer, max_length):
    tokenizer_seq_out = tokenizer(
        text=examples["seq"],
        return_attention_mask=False,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

    tokenizer_desc_out = tokenizer(
        text=examples["desc"],
        return_attention_mask=False,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

    input_ids = tokenizer_seq_out["input_ids"]
    label_ids = tokenizer_desc_out["input_ids"]
    
    padded_fp = np.full((len(examples["molecule_fp"]), max_length, len(examples["molecule_fp"][0][0])), -1)
    # the eos is also included in the following padding
    for i, fp in enumerate(examples["molecule_fp"]):
        if len(fp) > max_length-1:
            padded_fp[i, :max_length-1] = fp[:max_length-1]
        else:
            padded_fp[i, :len(fp)] = fp

    result = {"input_ids": input_ids, "labels": label_ids, "molecule_fp_ids": padded_fp}

    return result

def tokenize_function_desc2seq_fp(examples, tokenizer, max_length):
    tokenizer_seq_out = tokenizer(
        text=examples["seq"],
        return_attention_mask=False,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

    tokenizer_desc_out = tokenizer(
        text=examples["desc"],
        return_attention_mask=False,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )


    input_ids = tokenizer_desc_out["input_ids"]
    label_ids = tokenizer_seq_out["input_ids"]

    result = {"input_ids": input_ids, "labels": label_ids}

    return result

def tokenize_function_fp(examples, tokenizer, in_length):
    tokenizer_out = tokenizer(
        text=examples["text"],
        return_attention_mask=False,
    )

    input_ids = tokenizer_out["input_ids"]

    concatenated_ids = np.concatenate(input_ids)
    concatenated_fp_ids = np.vstack(examples["molecule_fp"])

    assert concatenated_ids.shape[0] == concatenated_fp_ids.shape[0], 'Incompatible shapes of SELFIES and FP'
    # if concatenated_ids.shape[0] != concatenated_fp_ids.shape[0]:
        # print('Incompatible shapes of SELFIES and FP')
        # logger.info('Incompatible shapes of SELFIES and FP')

    total_length = concatenated_ids.shape[0]
    total_length = (total_length // in_length) * in_length

    concatenated_ids = concatenated_ids[:total_length].reshape(-1, in_length)
    concatenated_fp_ids = concatenated_fp_ids[:total_length].reshape(-1, in_length, len(examples["molecule_fp"][0][0]))

    result = {"input_ids": concatenated_ids, "molecule_fp_ids": concatenated_fp_ids}

    return result

def tokenize_function_fp2selfies(examples, tokenizer, max_length):
    tokenizer_selfies_out = tokenizer(
        text=examples["text"],
        return_attention_mask=False,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

    input_ids = np.ones((len(examples["molecule_fp"]), max_length), dtype=np.int32)
    label_ids = tokenizer_selfies_out["input_ids"]
    
    padded_fp = np.full((len(examples["molecule_fp"]), max_length, len(examples["molecule_fp"][0][0])), -1)
    pend_fp = np.full((1, len(examples["molecule_fp"][0][0])), -1)
    # the eos is also included in the following padding
    for i, fp in enumerate(examples["molecule_fp"]):
        fp = np.array(fp)
        fp = fp[~(np.sum(fp, axis=1) == -4)]
        fp = np.concatenate((pend_fp, fp, pend_fp), axis=0)
        if len(fp) > max_length-1:
            padded_fp[i, :max_length-1] = fp[:max_length-1]
        else:
            padded_fp[i, :len(fp)] = fp

    result = {"input_ids": input_ids, "labels": label_ids, "molecule_fp_ids": padded_fp}

    return result

@dataclass
class DataCollatorForFPT5Finetune:
    pad_token_id: int = 0
    fp_bits: int = 4096
    fp_level: int = 3
    no_fp: bool = False

    def __call__(self, examples: List[Dict[str, np.ndarray]]) -> BatchEncoding:
        # convert list to dict and tensorize input
        # flatten list of dicts in tuple to list of dicts
        if isinstance(examples[0], tuple):
            examples = [sample_dict for sample_tuple in examples for sample_dict in sample_tuple]

        batch = BatchEncoding(
            {
                k: np.array([examples[i][k] for i in range(len(examples))])
                for k, v in examples[0].items()
            }
        )

        batch['labels'][batch['labels'] == self.pad_token_id] = -100

        batch = {k: torch.from_numpy(v) for k, v in batch.items()}

            
        if "molecule_fp_ids" not in batch:
            batch["molecule_fp_ids"] = torch.full((batch["input_ids"].shape[0], batch["input_ids"].shape[1], self.fp_level + 1), -1, dtype=torch.int32)
        if self.no_fp:
            batch["molecule_fp_ids"] = torch.full((batch["input_ids"].shape[0], batch["input_ids"].shape[1], self.fp_level + 1), -1, dtype=torch.int32)
        
        return batch


def tokenize_function_ft(examples, tokenizer, max_source_length, max_target_length, inst_format):
    if inst_format:
        prompt = "Below is an instruction that describes a task, paired with an input molecule. Write a response that appropriately completes the request.\n" \
                 "Instruction: {}\n" \
                 "Input molecule: {}.\n" \
                 "Response: "
        src_prompt = [prompt.format(inst, mol) for inst, mol in zip(examples["instruction"], examples["src"])]
        tokenizer_src_out = tokenizer(
            text=src_prompt,
            return_attention_mask=False,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=max_source_length,
        )

    else:
        tokenizer_src_out = tokenizer(
            text=examples["src"],
            return_attention_mask=False,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=max_source_length,
        )

    tokenizer_tgt_out = tokenizer(
        text=examples["tgt"],
        return_attention_mask=False,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=max_target_length,
    )

    input_ids = tokenizer_src_out["input_ids"]
    label_ids = tokenizer_tgt_out["input_ids"]
    
    padded_fp = np.full((len(examples["molecule_fp"]), max_source_length, len(examples["molecule_fp"][0][0])), -1)
    # the eos is also included in the following padding
    for i, fp in enumerate(examples["molecule_fp"]):
        if inst_format:
            # mol_start_idx = np.where(tokenizer_src_out["input_ids"][i] == 35044)[0][0]
            indices = np.where(tokenizer_src_out["input_ids"][i] == 35044)[0]
            if len(indices) > 0:
                mol_start_idx = indices[0]
            else: # if no molecule token is found
                continue
        else:
            mol_start_idx = 0

        available_space = max_source_length - mol_start_idx - 1
        if len(fp) > available_space:
            padded_fp[i, mol_start_idx:mol_start_idx + available_space] = fp[:available_space]
        else:
            padded_fp[i, mol_start_idx:mol_start_idx + len(fp)] = fp

    result = {"input_ids": input_ids, "labels": label_ids, "molecule_fp_ids": padded_fp}

    return result

@dataclass
class DataCollatorForFPT5FinetuneOnlyEncoderWithHead:
    pad_token_id: int = 0
    fp_bits: int = 4096
    fp_level: int = 3
    no_fp: bool = False

    def __call__(self, examples: List[Dict[str, np.ndarray]]) -> BatchEncoding:
        # convert list to dict and tensorize input
        # flatten list of dicts in tuple to list of dicts
        if isinstance(examples[0], tuple):
            examples = [sample_dict for sample_tuple in examples for sample_dict in sample_tuple]

        batch = BatchEncoding(
            {
                k: np.array([examples[i][k] for i in range(len(examples))])
                for k, v in examples[0].items()
            }
        )
        batch["labels"] = batch["labels"].astype(np.float32)
        batch = {k: torch.from_numpy(v) for k, v in batch.items()}

        if "molecule_fp_ids" not in batch:
            batch["molecule_fp_ids"] = torch.full((batch["input_ids"].shape[0], batch["input_ids"].shape[1], self.fp_level + 1), -1, dtype=torch.int32)
        if self.no_fp:
            batch["molecule_fp_ids"] = torch.full((batch["input_ids"].shape[0], batch["input_ids"].shape[1], self.fp_level + 1), -1, dtype=torch.int32)
        
        return batch

def tokenize_function_ft_no_fp(examples, tokenizer, max_source_length, max_target_length, inst_format):
    if inst_format:
        prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n" \
                 "Instruction: {}\n" \
                 "Input: {}\n" \
                 "Response: "
        src_prompt = [prompt.format(inst, mol) for inst, mol in zip(examples["instruction"], examples["src"])]
        tokenizer_src_out = tokenizer(
            text=src_prompt,
            return_attention_mask=False,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=max_source_length,
        )

    else:
        tokenizer_src_out = tokenizer(
            text=examples["src"],
            return_attention_mask=False,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=max_source_length,
        )

    tokenizer_tgt_out = tokenizer(
        text=examples["tgt"],
        return_attention_mask=False,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=max_target_length,
    )

    input_ids = tokenizer_src_out["input_ids"]
    label_ids = tokenizer_tgt_out["input_ids"]
    
    padded_fp = np.full((len(examples["molecule_fp"]), max_source_length, len(examples["molecule_fp"][0][0])), -1)

    result = {"input_ids": input_ids, "labels": label_ids, "molecule_fp_ids": padded_fp}

    return result
