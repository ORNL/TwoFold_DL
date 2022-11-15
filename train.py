import torch
import logging

import transformers
from transformers import BertModel, BertTokenizer, AutoTokenizer
from transformers import PreTrainedModel, BertConfig
from transformers import Trainer, TrainingArguments
from transformers import BatchEncoding
from transformers import EvalPrediction

from transformers import get_scheduler
from transformers.trainer_pt_utils import get_parameter_names

from tokenizers.pre_tokenizers import BertPreTokenizer
from tokenizers.pre_tokenizers import Digits
from tokenizers.pre_tokenizers import Sequence
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.pre_tokenizers import Split

from tokenizers import Regex
from tokenizers import pre_tokenizers
from tokenizers import normalizers
from tokenizers.normalizers import Replace

from dataclasses import dataclass, field
from enum import Enum

from transformers import HfArgumentParser
from transformers.trainer_utils import get_last_checkpoint
from transformers.modeling_utils import unwrap_model
from transformers.trainer_utils import set_seed

import datasets
from torch.utils.data import IterableDataset

from typing import List

import os
import json
from tqdm.auto import tqdm

import torch.distributed as dist

import numpy as np
import random

from contact_pred.data_utils import EnsembleDataCollatorWithPadding
from contact_pred.models import StructurePrediction
from contact_pred.models import ProteinLigandConfigStructure
from contact_pred.structure import IPAConfig

import webdataset as wd

logger = logging.getLogger(__name__)

def save_json(content, path, indent=4, **json_dump_kwargs):
    with open(path, "w") as f:
        json.dump(content, f, indent=indent, sort_keys=True, **json_dump_kwargs)

def handle_metrics(split, metrics, output_dir):
    """
    Log and save metrics
    Args:
    - split: one of train, val, test
    - metrics: metrics dict
    - output_dir: where to save the metrics
    """

    logger.info(f"***** {split} metrics *****")
    for key in sorted(metrics.keys()):
        logger.info(f"  {key} = {metrics[key]}")
    save_json(metrics, os.path.join(output_dir, f"{split}_results.json"))

@dataclass
class ModelArguments:
    model_type: str = field(
        default='bert',
        metadata = {'choices': ['bert','regex']},
    )

    seq_model_name: str = field(
        default=None
    )

    smiles_model_dir: str = field(
        default=None
    )

    smiles_tokenizer_dir: str = field(
        default=None
    )

    linear_mem_attn: bool = field(
        default=True
    )

    max_seq_length: int = field(
        default=2048
    )

    max_smiles_length: int = field(
        default=512
    )

    n_cross_attn: int = field(
        default=3
    )

    n_ipa: int = field(
        default=8
    )

@dataclass
class DataArguments:
    train_dataset: str = field(
        default=None
    )

    train_size: int = field(
        default=None
    )

    test_dataset: str = field(
        default=None
    )

    pretrained_model: str = field(
        default=None
    )

    freeze_protein: bool = field(
        default=False
    )

    freeze_ligand: bool = field(
        default=False
    )

    enable_cross: bool = field(
        default=True
    )

def main():
    from contact_pred.training_utils import mpi_discovery
    required_env = ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]

    auto_mpi_discovery = False
    try:
        import mpi4py
        auto_mpi_discovery = True
    except:
        logger.info("mpi4py not found, skipping MPI discovery")
        pass

    if auto_mpi_discovery and not all(map(lambda v: v in os.environ, required_env)):
        logger.info("Not using torchrun, attempting to detect MPI environment...")
        mpi_discovery()

    parser = HfArgumentParser([TrainingArguments,ModelArguments, DataArguments])

    training_args, model_args, data_args = parser.parse_args_into_dataclasses()

    if 'LOCAL_RANK' in os.environ:
        training_args.local_rank = int(os.environ["LOCAL_RANK"])

    # error out when there are unused parameters
    training_args.ddp_find_unused_parameters=False

    smiles_tokenizer_directory = model_args.smiles_tokenizer_dir
    smiles_model_directory = model_args.smiles_model_dir
    tokenizer_config = json.load(open(smiles_tokenizer_directory+'/config.json','r'))

    smiles_tokenizer =  AutoTokenizer.from_pretrained(smiles_tokenizer_directory, **tokenizer_config)

    if model_args.model_type == 'regex':
        smiles_tokenizer.backend_tokenizer.pre_tokenizer = Sequence([WhitespaceSplit(),Split(Regex(r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""), behavior='isolated')])

    normalizer = normalizers.Sequence([Replace(Regex('[UZOB]'),'X'),Replace(Regex('\s'),'')])
    pre_tokenizer = pre_tokenizers.Split(Regex(''),behavior='isolated')
    seq_tokenizer = AutoTokenizer.from_pretrained(model_args.seq_model_name, do_lower_case=False)
    seq_tokenizer.backend_tokenizer.normalizer = normalizer
    seq_tokenizer.backend_tokenizer.pre_tokenizer = pre_tokenizer



    max_seq_length = model_args.max_seq_length
    max_smiles_length = min(smiles_tokenizer.model_max_length, model_args.max_smiles_length)

    train_size = data_args.train_size
    import glob
    train = wd.WebDataset(glob.glob(data_args.train_dataset + '/part-*.tar'), resampled=True).shuffle(1000).decode('torch').with_epoch(train_size)
    test = wd.WebDataset(glob.glob(data_args.test_dataset + '/part-*.tar'), nodesplitter=lambda src: (s for s in src)).shuffle(1000).decode('torch')

    # on-the-fly tokenization
    def encode(item):
        if data_args.freeze_protein:
            item['seq.txt'] = ''
            item['receptor_frames_xyz'] = np.empty((0,3))
            item['receptor_frames_rot'] = np.empty((0,9))
            item['receptor_xyz'] = np.empty((0,model.config.num_atoms,3))

        seq_encodings = seq_tokenizer(item['seq.txt'],
                                 return_offsets_mapping=False,
                                 truncation=True,
                                 max_length=max_seq_length)

        item['input_ids_1'] = torch.tensor(seq_encodings['input_ids'])
        item['attention_mask_1'] = torch.tensor(seq_encodings['attention_mask'])

        if data_args.freeze_ligand:
            item['smiles.txt'] = ''
            item['ligand_xyz'] = np.empty((0,3))

        smiles_encodings = smiles_tokenizer(item['smiles.txt'],
                                            max_length=max_smiles_length,
                                            truncation=True)

        item['input_ids_2'] = torch.tensor(smiles_encodings['input_ids'])
        item['attention_mask_2'] = torch.tensor(smiles_encodings['attention_mask'])

        return item

    ensemble_collator = EnsembleDataCollatorWithPadding(smiles_tokenizer, seq_tokenizer)

    def collator_with_label_padding(features):
        if not isinstance(features[0], (dict, BatchEncoding)):
            features = [vars(f) for f in features]

        # Special handling for labels.
        first = list(features[0].keys())

        if 'ligand_xyz_2d' in first:
            for f in features:
                f.pop('ligand_xyz_2d')

        for k in first:
            if k.startswith('labels_'):
                continue
            if 'xyz' in k or 'rot' in k:
                if k.startswith('receptor'):
                    # max len in batch
                    max_len = max([len(f['attention_mask_1']) for f in features])
                else:
                    max_len = max([len(f['attention_mask_2']) for f in features])

                # pad with nan, also account for [CLS] and [SEP]
                for f in features:
                    try:
                        if k == 'receptor_xyz':
                            label = torch.tensor(np.pad(f[k][:max_len-2], ((1,max_len-1-len(f[k][:max_len-2])), (0,0), (0,0)),
                                constant_values=None).astype(np.float64)).type(torch.get_default_dtype())
                        else:
                            label = torch.tensor(np.pad(f[k][:max_len-2], ((1,max_len-1-len(f[k][:max_len-2])), (0,0)),
                                constant_values=None).astype(np.float64)).type(torch.get_default_dtype())
                    except:
                        print('Error padding inputs', k, f[k])
                        raise

                    # keep nan positions for loss calculation
                    # NOTE: a bug in pytorch requires this to be an int64 tensor, otherwise process will hang
                    non_nans = (~torch.any(torch.isnan(label),dim=-1)).type(torch.int64)

                    if k.endswith('_rot'):
                        label = label.reshape(label.shape[:-1] + (3,3))
                        label[non_nans==0,:,:] = torch.eye(3)
                    elif k != 'receptor_xyz':
                        label = torch.nan_to_num(label)

                    if k == 'receptor_frames_xyz':
                        f['labels_receptor_token_mask'] = non_nans
                    elif k == 'ligand_xyz':
                        f['labels_ligand_token_mask'] = non_nans

                        num_feat = model.config.num_atoms
                        feat = label.unsqueeze(1)
                        feat = torch.cat([feat, torch.ones(*(feat.shape[:1] + (num_feat-1,) + feat.shape[2:]),
                                                           device=feat.device, dtype=feat.dtype)], 1)
                        feat[:,1:,:] = float('nan')

                        f['labels_ligand_frames_xyz'] = label
                        f['labels_ligand_frames_rot'] = torch.eye(3, device=label.device, dtype=label.dtype).repeat(label.size()[0], 1, 1)
                        label = feat

                    elif k == 'receptor_frames_rot':
                        pass

                    f['labels_'+k] = label
                    f.pop(k)

        # process the remaining fields
        batch = ensemble_collator(features)

        return batch

    class MyDataset(IterableDataset):
        def __init__(self, dataset):
            self.dataset = dataset

        def __iter__(self):
            class Transform:
                def __init__(self, dataset):
                    self.dataset = dataset

                def __iter__(self):
                    self.iterator = iter(self.dataset)
                    return self

                def __next__(self):
                    item = next(self.iterator)

                    if 'lig_xyz.pyd' in item:
                        item['ligand_xyz'] = item.pop('lig_xyz.pyd')

                    if 'rec_xyz.pyd' in item:
                        item['receptor_frames_xyz'] = item.pop('rec_xyz.pyd')
                        item['receptor_frames_rot'] = item.pop('rec_r.pyd')
                        item['receptor_xyz'] = item.pop('rec_feat.pyd')[..., :-1, :]

                    item = encode(item)

                    if data_args.freeze_ligand and 'xyz.pyd' in item:
                        item['receptor_frames_xyz'] = item.pop('xyz.pyd')
                        item['receptor_frames_rot'] = item.pop('r.pyd')
                        item['receptor_xyz'] = item.pop('feat.pyd')[..., :-1, :]

                    if data_args.freeze_protein and 'xyz.pyd' in item:
                        item['ligand_xyz'] = item.pop('xyz.pyd')

                    return item

            return iter(Transform(self.dataset))

    class FromIterableDataset:
        def __init__(self, iterable_dataset):
            self.dataset = list(iterable_dataset)

        def __getitem__(self, i):
            return self.dataset[i]

        def __len__(self):
            return len(self.dataset)

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)

    seq_config = BertConfig.from_pretrained(model_args.seq_model_name)

    smiles_config = BertConfig.from_pretrained(smiles_model_directory)

    seq_ipa_config = IPAConfig(seq_config,
        num_ipa_heads=seq_config.num_attention_heads,
    )
    smiles_ipa_config = IPAConfig(smiles_config,
        num_ipa_heads=smiles_config.num_attention_heads,
    )

    config = ProteinLigandConfigStructure(
        seq_config=seq_config,
        smiles_config=smiles_config,
        n_cross_attention=model_args.n_cross_attn,
        seq_ipa_config=seq_ipa_config,
        smiles_ipa_config=smiles_ipa_config,
        num_ipa_layers=model_args.n_ipa,
        linear_mem_attn=model_args.linear_mem_attn,
        enable_cross=data_args.enable_cross,
        seq_vocab=seq_tokenizer.get_vocab()
    )

    # uniform seed for model weight initialization
    set_seed(training_args.seed)

    # instantiate model
    model = StructurePrediction(config)

    if not data_args.pretrained_model:
        # only load pretrained sequence embeddings
        model.pair_representation.embedding.load_pretrained(model_args.seq_model_name,
            model_args.smiles_model_dir)
    else:
        if torch.distributed.get_rank() == 0:
            print('Loading pre-trained checkpoint {}'.format(data_args.pretrained_model))
        pretrained_checkpoint = torch.load(data_args.pretrained_model,
                                           torch.device('cuda:{}'.format(training_args.local_rank)))
        model.load_state_dict(pretrained_checkpoint, strict=False)

    if data_args.freeze_protein:
        model.freeze_protein()
    if data_args.freeze_ligand:
        model.freeze_ligand()

    training_args.label_names = ['labels_receptor_frames_xyz',
                                 'labels_receptor_frames_rot',
                                 'labels_receptor_xyz',
                                 'labels_ligand_xyz',
                                 'labels_ligand_frames_xyz',
                                 'labels_ligand_frames_rot',
                                 'labels_ligand_token_mask',
                                 'labels_receptor_token_mask']
    training_args.remove_unused_columns = False

    # create optimizer, only for parameters which require gradients
    forbidden_parameter_names = ['bias']

    # exclude the linear scaling layers producing physical units
    forbidden_parameter_names += ['frame_translation.linear.']

    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if not any([s in name for s in forbidden_parameter_names])]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters and p.requires_grad],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]

    optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args)
    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    lr_scheduler = get_scheduler(
        training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.get_warmup_steps(training_args.max_steps),
        num_training_steps=training_args.max_steps,
    )

    train_dataset = MyDataset(train)
    val_dataset = FromIterableDataset(MyDataset(test))

    trainer = Trainer(
        model=model,
        args=training_args,                   # training arguments, defined above
        train_dataset=train_dataset,          # training dataset
        eval_dataset=val_dataset,             # evaluation dataset
        data_collator=collator_with_label_padding,
        optimizers=(optimizer, lr_scheduler),
    )

    # save model configuration
    if trainer.is_world_process_zero():
        with open(os.path.join(training_args.output_dir,'config.json'),'w') as f:
            json.dump(config.to_dict(), f)

    all_metrics = {}
    logger.info("*** Train ***")
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    trainer.save_model(training_args.output_dir)
    metrics = train_result.metrics

    if trainer.is_world_process_zero():
        handle_metrics("train", metrics, training_args.output_dir)
        all_metrics.update(metrics)

        trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))
        save_json(all_metrics, os.path.join(training_args.output_dir, "all_results.json"))

if __name__ == "__main__":
    main()
