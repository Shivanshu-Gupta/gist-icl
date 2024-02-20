# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Gist training script, adapted from huggingface's run_clm.py example.
"""

import logging
import os

import hydra
import torch  # noqa
from datasets import Dataset, DatasetDict, load_dataset
from omegaconf.dictconfig import DictConfig
from transformers import (
    AutoConfig,
    AutoTokenizer,
    LlamaTokenizer,
    is_torch_tpu_available,
    set_seed, AutoModel, Seq2SeqTrainer, AutoModelForSeq2SeqLM,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from gisting.src import gist_llama, gist_t5
from gisting.src.arguments import Arguments, global_setup
from gisting.src.data import alpaca
from gisting.src.data.utils import nested_select
from gisting.src.gist_llama import DEBUG_LLAMA_CONFIG, GistLlamaForCausalLM
from gisting.src.gist_t5 import GistT5ForConditionalGeneration
from gisting.src.integrations import CustomWandbCallback, EvaluateFirstStepCallback
from gisting.src.metrics import get_compute_metrics_fn
from gisting.src.t5_pe_mixin import T5PEForConditionalGeneration, T5PEConfig
from gisting.src.trainer_seq2seq import GistSeq2SeqTrainer

from constants import Dataset as D

# Will error if the minimal version of Transformers is not installed. Remove at
# your own risks.
check_min_version("4.28.0.dev0")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt",
)

logger = logging.getLogger(__name__)

def load_ckpt(args):
    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(args.training.output_dir)
        and args.training.do_train
        and not args.training.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(args.training.output_dir)
        if last_checkpoint is None and len(os.listdir(args.training.output_dir)) > 0:
            existing_files = os.listdir(args.training.output_dir)
            logger.warning(
                (
                    "Output directory (%s) already exists and "
                    "is not empty. Existing files: %s. "
                    "Training anyways as these may just be output files."
                ),
                args.training.output_dir,
                str(existing_files),
            )
        elif (
            last_checkpoint is not None and args.training.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To "
                "avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from "
                "scratch."
            )
    return last_checkpoint

def get_datasets(args: Arguments, tokenizer):
    if args.data.dataset_name == D.ALPACA:
        lm_datasets = load_dataset(
            "gisting/src/data/alpaca/alpaca.py",
            cache_dir=args.model.cache_dir,
        )
        train_dataset = get_train_dataset(args, lm_datasets) if args.training.do_train else None
        eval_dataset = dict(get_eval_dataset(args, lm_datasets)) if args.training.do_eval else None
    elif args.data.dataset_name == D.FLAN:
        if args.data.flan_dataset_name is None:
            raise ValueError('flan_dataset_name must be specified')
        if args.data.flan_dataset_name.startswith('flan2021'):
            dataset = Dataset.load_from_disk(f'multitask-data/flan2021/{args.data.flan_dataset_name}')
        elif args.data.flan_dataset_name.startswith('flan2022'):
            dataset = Dataset.load_from_disk(f'multitask-data/flan2022/{args.data.flan_dataset_name}')
        elif args.data.flan_dataset_name.startswith('flan_mini'):
            dataset = Dataset.load_from_disk(f'multitask-data/flan_mini/{args.data.flan_dataset_name}')
        else:
            raise ValueError(f'flan_dataset_name {args.data.flan_dataset_name} not supported')

        logger.warning(f'Using {len(dataset)} examples from FLAN')

        lm_datasets = dataset.train_test_split(test_size=0.05, seed=42)
        train_dataset = lm_datasets['train']
        eval_dataset = DatasetDict({'validation': lm_datasets['test']})
    else:
        from data_params import ds2cls
        from pathlib import Path
        DP = ds2cls[args.data.dataset_name]() if args.data.split is None else ds2cls[args.data.dataset_name](split=args.data.split)
        if args.data.dataset_name == D.BREAK:
            DP.qdecomp_path = 'icl-demo-selection/src/third_party/qdecomp_with_dependency_graphs'
        get_splits_kwargs = dict(
            data_root=Path('icl-demo-selection/data'),
            dataloaders_dir=Path('icl-demo-selection/src/data'),
            max_len=500
        )
        train_dataset, _, eval_dataset = DP.get_splits(**get_splits_kwargs, tokenizer=tokenizer)
        eval_dataset = DatasetDict({'validation': eval_dataset})
        if args.evaluate_only:
            DP.n_test = 1000
            if DP.dataset == D.SMCALFLOW_CS:
                DP.split = '8_S'
                _, _, iid_eval = DP.get_splits(**get_splits_kwargs)
                DP.split = '32_C'
                _, _, cg_eval = DP.get_splits(**get_splits_kwargs)
                eval_dataset = DatasetDict({'8_S': iid_eval, '32_C': cg_eval})
            elif DP.dataset == D.COGS:
                DP.test_split = 'dev'
                _, _, iid_eval = DP.get_splits(**get_splits_kwargs)
                DP.test_split = 'gen'
                _, _, cg_eval = DP.get_splits(**get_splits_kwargs)
                eval_dataset = DatasetDict({'dev': iid_eval, 'gen': cg_eval})
            else:
                _, _, eval_dataset = DP.get_splits(**get_splits_kwargs)
                eval_dataset = DatasetDict({'validation': eval_dataset})

    if args.data.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), args.data.max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))
    if not args.evaluate_only:
        # (Deterministically) shuffle eval in case we are truncating.
        eval_dataset = eval_dataset.shuffle(seed=42)
        if args.data.max_eval_samples is not None:
            eval_dataset = nested_select(
                eval_dataset,
                args.data.max_eval_samples,
            )
    logger.warning(f'Train dataset: {train_dataset}')
    logger.warning(f'Eval dataset: {eval_dataset}')
    return train_dataset, eval_dataset

def get_train_dataset(args: Arguments, lm_datasets) -> Dataset | None:
    if "train" not in lm_datasets:
        raise ValueError("--do_train requires a train dataset")
    train_dataset = lm_datasets["train"]
    return train_dataset

def get_eval_dataset(args: Arguments, lm_datasets) -> DatasetDict | Dataset | None:
    validation_splits = [
        split for split in lm_datasets if split.startswith("validation")
    ]
    if not validation_splits:
        raise ValueError(
            "--do_eval requires at least one validation dataset "
            "that starts with `validation`"
        )
    eval_dataset = DatasetDict(
        # Trim "validation-" prefix.
        {split[11:]: lm_datasets[split] for split in validation_splits}
    )
    return eval_dataset

def get_config(args: Arguments, is_t5):
    config_kwargs = {
        "cache_dir": args.model.cache_dir,
        "revision": args.model.model_revision,
        "use_auth_token": True if args.model.use_auth_token else None,
    }
    mode = args.training.mode
    if mode == 'gisting':
        if args.model.llama_debug:
            if args.model.pretrained:
                raise RuntimeError("llama_debug requires pretrained set to False")
            config = DEBUG_LLAMA_CONFIG
        elif args.model.config_name:
            config = AutoConfig.from_pretrained(args.model.config_name, **config_kwargs)
        elif args.model.model_name_or_path:
            config = AutoConfig.from_pretrained(
                args.model.model_name_or_path, **config_kwargs
            )
        else:
            raise ValueError(
                "Unlike run_clm.py, this script does not support specifying a model type "
                "from scratch. Specify args.model.model_name_or_path and set "
                "args.pretrained = False to train from scratch instead."
            )
    elif mode == 'position_bias':
        if is_t5:
            config = T5PEConfig.from_pretrained(args.model.config_name, **config_kwargs)
        else:
            raise ValueError(f"Model type {args.model.model_name_or_path} not supported in gisting mode")
    else:
        raise ValueError(f'training mode "{mode}" is not supported atm')
    return config

def get_tokenizer(args: Arguments, is_llama):
    tokenizer_kwargs = {
        "cache_dir": args.model.cache_dir,
        "use_fast": args.model.use_fast_tokenizer,
        "revision": args.model.model_revision,
        "use_auth_token": True if args.model.use_auth_token else None,
    }
    if args.model.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model.tokenizer_name, **tokenizer_kwargs
        )
    elif args.model.model_name_or_path:
        if is_llama:
            tokenizer = LlamaTokenizer.from_pretrained(
                args.model.model_name_or_path, **tokenizer_kwargs
            )
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model.model_name_or_path, **tokenizer_kwargs
            )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported "
            "by this script."
            "You can do it from another script, save it, and load it from here, using "
            "--tokenizer_name."
        )
    return tokenizer


def get_model(args: Arguments, config, is_t5, is_llama):
    mode = args.training.mode
    if mode == 'gisting':
        if is_t5:
            model_cls = GistT5ForConditionalGeneration
        elif is_llama:
            model_cls = GistLlamaForCausalLM
        else:
            raise ValueError(f"Model type {args.model.model_name_or_path} not supported in gisting mode")
    elif mode == 'position_bias':
        if is_t5:
            model_cls = T5PEForConditionalGeneration
        else:
            raise ValueError(f"Model type {args.model.model_name_or_path} not supported in gisting mode")
    elif mode == 'normal':
        model_cls = AutoModelForSeq2SeqLM
        args.training.gist.condition = 'normal'
    else:
        raise ValueError(f'training mode "{mode}" is not supported atm')

    dtypes = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float,
    }

    if args.model.pretrained:
        model = model_cls.from_pretrained(
            args.model.model_name_or_path,
            from_tf=bool(".ckpt" in args.model.model_name_or_path),
            config=config,
            cache_dir=args.model.cache_dir,
            revision=args.model.model_revision,
            torch_dtype=dtypes[args.model.precision],
            use_auth_token=True if args.model.use_auth_token else None,
        )
    else:
        model = model_cls(config)
    return model

def add_gisting(args: Arguments, model, tokenizer, is_t5, is_llama):
    # Check if gist token has already been added to the model (e.g. because
    # we're resuming from a checkpoint.)
    if is_t5 and len(tokenizer) == gist_t5.PRETRAINED_VOCAB_SIZE + 1:
        assert model.shared.weight.shape[0] == gist_t5.PRETRAINED_VOCAB_SIZE + 1
    elif is_llama and len(tokenizer) == gist_llama.PRETRAINED_VOCAB_SIZE + 1:
        assert (
            model.model.embed_tokens.weight.shape[0]
            == gist_llama.PRETRAINED_VOCAB_SIZE + 1
        )
        assert model.lm_head.weight.shape[0] == gist_llama.PRETRAINED_VOCAB_SIZE + 1
    else:
        # Initialize gist token
        tokenizer.add_special_tokens({"additional_special_tokens": ["<GIST>"]}, replace_additional_special_tokens=False)
        model.resize_token_embeddings(len(tokenizer))
        # Set new word embedding to average of existing word embeddings. For why,
        # see https://nlp.stanford.edu/~johnhew/vocab-expansion.html
        if args.model.pretrained:
            with torch.no_grad():
                if is_t5:
                    model.shared.weight[-1] = model.shared.weight[:-1].mean(0)
                elif is_llama:
                    model.model.embed_tokens.weight[
                        -1
                    ] = model.model.embed_tokens.weight[:-1].mean(0)
                    model.lm_head.weight[-1] = model.lm_head.weight[:-1].mean(0)
                else:
                    raise ValueError(
                        f"Model type {args.model.model_name_or_path} not supported"
                    )

def get_collator(args: Arguments, tokenizer, model, is_t5, is_llama, gist_token):
    if is_t5:
        data_collator = alpaca.collator.DataCollatorForAlpaca(
            args.data.dataset_name,
            args.data.split,
            tokenizer,
            model=model,
            padding="longest",
            # Chosen so that <1% of examples are truncated.
            # See data/alpaca_plus/length_stats.txt for length stats.
            max_source_length=128 if args.data.dataset_name == D.ALPACA else 512,
            max_target_length=256,
            # Human eval examples are longer.
            max_source_length_human=384,
            max_target_length_human=384,
            label_pad_token_id=-100,
            pad_to_multiple_of=8 if args.training.fp16 else None,
            gist_condition=args.training.gist.condition,
            num_gist_tokens=args.training.gist.num_gist_tokens,
            gist_token=gist_token,
            pad_token=tokenizer.pad_token_id,
            add_gist_token=args.training.gist.add_gist_token,
        )
    elif is_llama:
        # This data collator variant does causal language modeling with left
        # padding.
        data_collator = alpaca.collator.DataCollatorForAlpacaCLM(
            tokenizer,
            # Chosen so that <1% of examples are truncated.
            # See data/alpaca_plus/length_stats.txt for length stats.
            max_length=256 + 256,  # source=256; target=256
            # Human eval examples are longer.
            max_length_human=384 + 384,  # source=384; target=384
            gist_condition=args.training.gist.condition,
            num_gist_tokens=args.training.gist.num_gist_tokens,
            gist_token=gist_token,
            pad_token=tokenizer.pad_token_id,
            check_correctness=True,
        )
    else:
        assert False, "should be is_llama or is_t5"
    return data_collator

def get_trainer(args: Arguments, model, tokenizer, train_dataset, eval_dataset, data_collator, compute_metrics):
    # Initialize our Trainer
    custom_callbacks = []
    if args.wandb.log:
        custom_callbacks.append(CustomWandbCallback(args))
    if args.training.evaluate_before_train:
        custom_callbacks.append(EvaluateFirstStepCallback())
    if args.training.early_stopping_patience > 0:
        from transformers import EarlyStoppingCallback
        custom_callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=args.training.early_stopping_patience))

    mode = args.training.mode
    if mode == 'gisting':
        trainer_cls = GistSeq2SeqTrainer
    elif mode == 'position_bias':
        trainer_cls = Seq2SeqTrainer
    elif mode == 'normal':
        trainer_cls = Seq2SeqTrainer
    else:
        raise ValueError(f'training mode "{mode}" is not supported atm')

    trainer = trainer_cls(
        model=model,
        args=args.training,
        train_dataset=train_dataset if args.training.do_train else None,
        eval_dataset=eval_dataset if args.training.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
        if args.training.do_eval and not is_torch_tpu_available()
        else None,
        preprocess_logits_for_metrics=None,
        callbacks=custom_callbacks,
    )
    return trainer

@hydra.main(config_path="conf", config_name="config")
def main(args: DictConfig) -> None:
    args: Arguments = global_setup(args)
    last_checkpoint = load_ckpt(args)

    # Set seed before initializing model.
    set_seed(args.training.seed)

    # config, tokenizer, model
    is_t5 = any(t in args.model.model_name_or_path.lower() or True for t in ("t5", "tk"))
    is_llama = any(t in args.model.model_name_or_path.lower() for t in ("llama",))
    config = get_config(args, is_t5=is_t5)
    tokenizer = get_tokenizer(args, is_llama=is_llama)
    model = get_model(args, config=config, is_t5=is_t5, is_llama=is_llama)
    add_gisting(args, model, tokenizer, is_t5=is_t5, is_llama=is_llama)

    if args.training.lora:
        from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType

        # Define LoRA Config
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q", "v"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )
        # # prepare int-8 model for training
        # model = prepare_model_for_int8_training(model)

        # add LoRA adaptor
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    gist_token = tokenizer.additional_special_tokens_ids[-1]

    # data and metrics
    train_dataset, eval_dataset = get_datasets(args, tokenizer)
    data_collator = get_collator(args, tokenizer, model, is_t5, is_llama, gist_token)
    compute_metrics = get_compute_metrics_fn(
        gist_token=gist_token, tokenizer=tokenizer, args=args
    ) if args.training.do_eval and not is_torch_tpu_available() else None

    # Initialize our Trainer
    trainer = get_trainer(args, model, tokenizer, train_dataset, eval_dataset, data_collator, compute_metrics)

    # Training
    if args.training.do_train:
        checkpoint = None
        if args.training.resume_from_checkpoint is not None:
            checkpoint = args.training.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            args.data.max_train_samples
            if args.data.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if args.training.do_benchmarking:
        if not args.training.do_eval:
            raise RuntimeError("do_benchmarking requires do_eval")
        trainer.benchmark(
            gist_token,
            eval_dataset["human"],
            output_file=args.training.benchmarking_output_file,
        )
        logger.info("Only doing benchmarking. Exiting!")
        return

    # Do evaluation for each dataset.
    if args.training.do_eval:
        all_eval_metrics = {}
        for eval_name, to_eval in eval_dataset.items():
            logger.info(f"*** Evaluate {eval_name} ***")

            metrics = trainer.evaluate(to_eval)

            max_eval_samples = (
                args.data.max_eval_samples
                if args.data.max_eval_samples is not None
                else len(to_eval)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(to_eval))

            metrics = {
                (f"{eval_name}_{k}" if k != "epoch" else k): v
                for k, v in metrics.items()
            }
            all_eval_metrics.update(metrics)
        trainer.log_metrics("eval", all_eval_metrics)
        trainer.save_metrics("eval", all_eval_metrics)


if __name__ == "__main__":
    main()
