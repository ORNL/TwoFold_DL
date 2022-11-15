from transformers import DataCollatorWithPadding, default_data_collator, PreTrainedTokenizerBase
from transformers.data.data_collator import default_data_collator, DataCollatorMixin, DataCollatorForLanguageModeling

from transformers import Pipeline

from torch.utils.data import DataLoader

from typing import List, Dict, Any

import os
import collections
from collections.abc import Iterable

import logging
logger = logging.getLogger()

class EnsembleDataCollatorWithPadding:
    def __init__(self,
                 smiles_tokenizer,
                 seq_tokenizer,
                 smiles_padding=True,
                 smiles_max_length=None,
                 seq_padding=True,
                 seq_max_length=None):

        self.smiles_collator = DataCollatorWithPadding(smiles_tokenizer, smiles_padding, smiles_max_length)
        self.seq_collator = DataCollatorWithPadding(seq_tokenizer, seq_padding, seq_max_length)

    def __call__(self, features):
        # individually collate protein and ligand sequences into batches
        batch_1 = self.seq_collator([{'input_ids': b['input_ids_1'], 'attention_mask': b['attention_mask_1']} for b in features])
        batch_2 = self.smiles_collator([{'input_ids': b['input_ids_2'], 'attention_mask': b['attention_mask_2']} for b in features])

        batch_merged = default_data_collator([{k: v for k,v in f.items()
                                              if k not in ('input_ids_1','attention_mask_1','input_ids_2','attention_mask_2')}
                                            for f in features])
        batch_merged['input_ids_1'] = batch_1['input_ids']
        batch_merged['attention_mask_1'] = batch_1['attention_mask']
        batch_merged['input_ids_2'] = batch_2['input_ids']
        batch_merged['attention_mask_2'] = batch_2['attention_mask']
        return batch_merged

class EnsembleTokenizer:
    def __init__(self,
                 smiles_tokenizer,
                 seq_tokenizer,
    ):
        self.smiles_tokenizer = smiles_tokenizer
        self.seq_tokenizer = seq_tokenizer

    def __call__(self, features, **kwargs):
        item = dict(features)

        is_batched = isinstance(features, Iterable) and not isinstance(features, dict)

        seq_args = {}
        smiles_args = {}
        if 'seq_padding' in kwargs:
            seq_args['padding'] = kwargs['seq_padding']
        if 'smiles_padding' in kwargs:
            smiles_args['padding'] = kwargs['smiles_padding']
        if 'seq_max_length' in kwargs:
            seq_args['max_length'] = kwargs['seq_max_length']
        if 'smiles_max_length' in kwargs:
            smiles_args['max_length'] = kwargs['smiles_max_length']
        if 'seq_truncation' in kwargs:
            seq_args['truncation'] = kwargs['seq_truncation']
        if 'smiles_truncation' in kwargs:
            smiles_args['truncation'] = kwargs['smiles_truncation']

        if is_batched:
            seq_encodings = self.seq_tokenizer([f['protein'] for f in features], **seq_args)
        else:
            seq_encodings = self.seq_tokenizer(features['protein'], **seq_args)

        item.pop('protein')
        item['input_ids_1'] = seq_encodings['input_ids']
        item['attention_mask_1'] = seq_encodings['attention_mask']

        if is_batched:
            smiles_encodings = self.smiles_tokenizer([f['ligand'] for f in features], **smiles_args)
        else:
            smiles_encodings = self.smiles_tokenizer(features['ligand'], **smiles_args)

        item.pop('ligand')
        item['input_ids_2'] = smiles_encodings['input_ids']
        item['attention_mask_2'] = smiles_encodings['attention_mask']

        return item

class StructurePredictionPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}

        if 'seq_padding' in kwargs:
            preprocess_kwargs['seq_padding'] = kwargs['seq_padding']
        else:
            preprocess_kwargs['seq_padding'] = True

        if 'smiles_padding' in kwargs:
            preprocess_kwargs['smiles_padding'] = kwargs['smiles_padding']
        else:
            preprocess_kwargs['smiles_padding'] = True

        if 'seq_truncation' in kwargs:
            preprocess_kwargs['seq_truncation'] = kwargs['seq_truncation']
        else:
            preprocess_kwargs['seq_truncation'] = True

        if 'seq_max_length' in kwargs:
            preprocess_kwargs['seq_max_length'] = kwargs['seq_max_length']
        else:
            preprocess_kwargs['seq_max_length'] = None

        if 'smiles_truncation' in kwargs:
            preprocess_kwargs['smiles_truncation'] = kwargs['smiles_truncation']
        else:
            preprocess_kwargs['smiles_truncation'] = True

        if 'smiles_max_length' in kwargs:
            preprocess_kwargs['smiles_max_length'] = kwargs['smiles_max_length']
        else:
            preprocess_kwargs['smiles_max_length'] = None

        return preprocess_kwargs, {}, {}

    def __init__(self,
        model,
        seq_tokenizer,
        smiles_tokenizer,
        output_prediction_scores=False,
        **kwargs
        ):
        self.seq_tokenizer = seq_tokenizer
        self.smiles_tokenizer = smiles_tokenizer
        self.data_collator = EnsembleDataCollatorWithPadding(self.smiles_tokenizer,
                                                             self.seq_tokenizer)
        self.output_prediction_scores = output_prediction_scores
        model.eval()
        super().__init__(model=model,
                         tokenizer=EnsembleTokenizer(self.smiles_tokenizer,
                                                    self.seq_tokenizer),
                         **kwargs)

    def preprocess(self, inputs, **kwargs):
        tokenized_input = self.tokenizer(inputs, **kwargs)
        return tokenized_input

    def _forward(self, model_inputs):
        outputs = self.model(**model_inputs,
                             return_dict=True)
        return outputs

    def postprocess(self, model_outputs):
        if isinstance(model_outputs, dict):
            return {k: v.numpy() for k,v in model_outputs.items()}
        else:
            return model_outputs.numpy()

    def get_iterator(
        self, inputs, num_workers: int, batch_size: int, preprocess_params, forward_params, postprocess_params
    ):
        from transformers.pipelines.pt_utils import PipelineDataset, PipelineIterator
        if isinstance(inputs, collections.abc.Sized):
            dataset = PipelineDataset(inputs, self.preprocess, preprocess_params)
        else:
            if num_workers > 1:
                logger.warning(
                    "For iterable dataset using num_workers>1 is likely to result"
                    " in errors since everything is iterable, setting `num_workers=1`"
                    " to guarantee correctness."
                )
                num_workers = 1
            dataset = PipelineIterator(inputs, self.preprocess, preprocess_params)
        if "TOKENIZERS_PARALLELISM" not in os.environ:
            logger.info("Disabling tokenizer parallelism, we're using DataLoader multithreading already")
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

        collate_fn = self.data_collator
        dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, collate_fn=collate_fn)
        model_iterator = PipelineIterator(dataloader, self.forward, forward_params, loader_batch_size=batch_size)
        final_iterator = PipelineIterator(model_iterator, self.postprocess, postprocess_params)
        return final_iterator

