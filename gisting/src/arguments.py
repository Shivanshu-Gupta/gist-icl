"""
Huggingface TrainingArguments + hydra = *chefs kiss*
"""


import logging
import os
import os.path as osp
import socket
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union
from rich import print
from rich.logging import RichHandler

import datasets
import torch  # noqa
import transformers
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from transformers import Seq2SeqTrainingArguments, TrainingArguments

import sys
if 'icl-demo-selection/src' not in sys.path:
    sys.path.append('icl-demo-selection/src')
from constants import Dataset as D

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler(rich_tracebacks=True)]
)

logger = logging.getLogger(__name__)


def simple_basename(f):
    return osp.splitext(osp.basename(f))[0]


OmegaConf.register_new_resolver("basename", simple_basename)


@dataclass
class GistArguments:
    """
    Arguments for gist
    """

    condition: str = field(
        default="gist",
        metadata={"help": "One of gist, pos_control (no masking), or neg_control."},
    )
    num_gist_tokens: int = field(
        default=1,
        metadata={
            "help": (
                "number of gist tokens to insert. successive gist tokens can attend "
                "to each other; post-gist-tokens can attend to all gist tokens prior."
            )
        },
    )
    add_gist_token: bool = field(
        default=True,
        metadata={"help": ("Whether to add gist token or not.")},
    )


@dataclass
class WandBArguments:
    log: bool = field(default=False, metadata={"help": "log to wandb"})
    entity: Optional[str] = field(
        default=os.environ.get("USERNAME") or os.environ.get("USER"),
        metadata={"help": "WandB entity"},
    )
    project: Optional[str] = field(
        default="alscratch", metadata={"help": "wandb project"}
    )
    group: Optional[str] = field(default="debug", metadata={"help": "wandb group"})
    name: Optional[str] = field(default="run", metadata={"help": "wandb run name"})
    tag: Optional[str] = field(
        default="run", metadata={"help": "optional tag to prefix wandb group"}
    )


@dataclass
class GistTrainingArguments(TrainingArguments):
    """
    Fix some of the types of TrainingArguments so that OmegaConf typechecks
    okay, and control when _post_init happens.
    """

    gist: GistArguments = GistArguments()

    # if True, benchmark the model with and without gist.
    do_benchmarking: bool = False
    # if True, do sanity checking on gist caching (verifying that
    # e.g. encoder representations look the same with and without gist caching).
    # NOTE: This will fail for very large models on bf16 due to floating point
    # errors.
    do_benchmarking_sanity_checks: bool = False
    # What path to save the benchmarking outputs to.
    benchmarking_output_file: str = "benchmarking-outputs.csv"
    # Use either deepspeed or pytorch profiler. Paper uses pytorch.
    benchmarking_profiler: str = "pytorch"

    # If `benchmarking_prompt_caching` is True, cache prompts rather than gists
    # for decoder-only models.
    benchmarking_prompt_caching: bool = False

    max_benchmarking_samples: Optional[int] = 256

    # Change these types to strs so that they typecheck with str configs
    debug: str = ""
    fsdp: bool | str = False
    evaluation_strategy: Optional[str] = "no"
    lr_scheduler_type: Optional[str] = "linear"
    logging_strategy: Optional[str] = "steps"
    save_strategy: Optional[str] = "steps"
    optim: Optional[str] = "adamw_hf"
    hub_strategy: Optional[str] = "every_save"
    report_to: str = "none"
    remove_unused_columns: bool = False
    include_inputs_for_metrics: bool = True

    write_outputs: bool = field(
        default=True,
        metadata={"help": ("If True, save outputs to .csv file in output dir.")},
    )
    evaluate_before_train: bool = False

    # Change these types to optional
    data_seed: Optional[int] = None
    tf32: Optional[bool] = None
    xpu_backend: Optional[str] = None
    eval_steps: Optional[int] = None
    hub_model_id: Optional[str] = None
    hub_token: Optional[str] = None
    push_to_hub_model_id: Optional[str] = None
    push_to_hub_organization: Optional[str] = None
    push_to_hub_token: Optional[str] = None
    gradient_checkpointing_kwargs: Optional[dict] = None
    neftune_noise_alpha: Optional[float] = None

    early_stopping_patience: Optional[int] = -1

    _run_post_init: bool = False

    def __post_init__(self):
        # Don't run post-init until ready to convert to TrainingArgs
        if self._run_post_init:
            super().__post_init__()


@dataclass
class GistSeq2SeqTrainingArguments(GistTrainingArguments, Seq2SeqTrainingArguments):
    generation_config: Optional[Union[str, Path]] = field(
        default=None,
        metadata={
            "help": "Model id, file path or url pointing to a GenerationConfig json file, to use during prediction."
        },
    )
    mode: str = field(
        default='gisting',
        metadata={
            'help': 'only valid choices: {gisting, position_bias, normal}'
        }
    )

    lora: Optional[bool] = field(
        default=False,
        metadata={
            'help': 'whether to use lora'
        }
    )



@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to
    fine-tune, or train from scratch.
    """

    model_name_or_path: str = field(
        default="google/flan-t5-base",
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you "
                "want to train a model from scratch."
            )
        },
    )
    llama_debug: bool = field(
        default=False,
        metadata={"help": "If True, load a tiny LLama Model for debugging."},
    )
    pretrained: bool = field(
        default=True,
        metadata={
            "help": (
                "Use pretrained model. This replaces old run_clm.py script "
                "`model_type` argument."
            )
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Where do you want to store the pretrained models downloaded from "
                "huggingface.co"
            )
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to use one of the fast tokenizer (backed by the tokenizers "
                "library) or not."
            )
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": (
                "The specific model version to use (can be a branch name, tag name "
                "or commit id)."
            )
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` "
                "(necessary to use this script with private models)."
            )
        },
    )
    position_bias: Optional[str] = field(
        default=None,
        metadata={
            'help': 'the position bias used for the experiment'
        }
    )

    precision: Optional[str] = field(
        default='fp32',
        metadata={
            'help': 'model parameters precision used for the experiment'
        }
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for
    training and eval.
    """

    dataset_name: Optional[D] = field(
        default=D.ALPACA,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )

    split: Optional[str] = field(
        default=None,
        metadata={"help": "The split of the dataset to use."},
    )

    flan_dataset_name: Optional[str] = field(
        default=None
    )

    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The configuration name of the dataset to use (via the datasets "
                "library)."
            )
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity "
            "on (a text file)."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of "
                "training examples to this value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of "
                "evaluation examples to this value if set."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
        ):
            raise ValueError(
                "Need either a dataset name or a training/validation file."
            )
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                    "txt",
                ], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                    "txt",
                ], "`validation_file` should be a csv, a json or a txt file."


@dataclass
class Arguments:
    model: ModelArguments = ModelArguments()
    data: DataTrainingArguments = DataTrainingArguments()
    wandb: WandBArguments = WandBArguments()
    training: GistSeq2SeqTrainingArguments = GistSeq2SeqTrainingArguments("dummy")
    evaluate_only: bool = field(
        default=False,
        metadata={"help": "If true, evaluate the model on the ICL test splits."},
    )

cs = ConfigStore.instance()
cs.store(name="base_config", node=Arguments)


def global_setup(args: DictConfig) -> Arguments:
    """Global setup of arguments."""
    hostname = socket.gethostname()
    logger.info(f"Running on {hostname}")

    if args.training.report_to != "none":
        raise ValueError(
            "report_to is disabled; use training.wandb settings to "
            "configure wandb logging."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Convert args to the actual dataclass object, to enable methods.  Need to
    # delete _n_gpu, a property that TrainingArgs init doesn't expect.
    del args.training._n_gpu
    # Dirty hack: only run post init when we're ready to convert to TrainingArgs
    args.training._run_post_init = True
    args: Arguments = OmegaConf.to_object(args)
    if args.data.dataset_name == D.FLAN and args.data.flan_dataset_name is not None:
        args.training.output_dir = args.training.output_dir.replace(D.FLAN.value, args.data.flan_dataset_name)
        args.wandb.group = args.wandb.group.replace(D.FLAN.value, args.data.flan_dataset_name)
        args.wandb.name = args.wandb.name.replace(D.FLAN.value, args.data.flan_dataset_name)

    log_level = args.training.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {args.training.local_rank}, device: {args.training.device}, "
        f"n_gpu: {args.training.n_gpu}"
        f" distributed training: {bool(args.training.local_rank != -1)}, 16-bits "
        f"training: {args.training.fp16}, bf16 training: {args.training.bf16}"
    )
    logger.warning(f"output_dir: {args.training.output_dir}")
    import pprint
    logger.warning(f"All parameters {pprint.pformat(args)}")
    # print(args)

    return args
