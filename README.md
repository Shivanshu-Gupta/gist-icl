# Gisting for In-Context Example Selection

This repository implements the in-context example-selection approach described in the paper [GistScore: Learning Better Representations for In-Context Example Selection with Gist Bottlenecks](https://arxiv.org/abs/2311.09606).

- [Gisting for In-Context Example Selection](#gisting-for-in-context-example-selection)
  - [Setup](#setup)
  - [Organization](#organization)
  - [Workflows](#workflows)
    - [Finetuning Gist Models on Individual Datasets](#finetuning-gist-models-on-individual-datasets)
    - [Multi-task Training](#multi-task-training)
    - [Running ICL Evaluations](#running-icl-evaluations)
    - [Adding a new dataset](#adding-a-new-dataset)
  - [Command Line Tips](#command-line-tips)
  - [Citation](#citation)


## Setup

1. Clone this repository along with the `icl` submodule: `git clone --recurse-submodules https://github.com/Shivanshu-Gupta/gist-icl/`
2. Install Python dependencies:

    ```bash
    pip install -r gisting/requirements.txt
    pip install -r icl/requirements.txt -U
    ```

3. Set up for ICL evaluations as described in [`icl/README.md`](https://github.com/Shivanshu-Gupta/in-context-learning#setup).
4. Download the finetuned and multi-task trained gist model checkpoints from [here][gistlms] and store in `gistlms/finetunes` and `gistlms/pretrains`, respectively.

## Organization

This repository is organized as follows:

```plaintext
gist-icl
├── gisting                 (code for training/evaluating gist models forked from https://github.com/jayelm/gisting)
├── gistlms                 (gist training logs and models)
│   ├── finetunes           (finetuned gist lms)
│   └── pretrains           (multi-task trained gist lms)
├── multitask-data          (multi-task data collections)
└── icl                     (code for ICL evaluations -- https://github.com/Shivanshu-Gupta/icl)
```

For details of the ICL repository, see [`icl/README.md`](icl/README.md#organization).

## Workflows

### Finetuning Gist Models on Individual Datasets

Gist LMs are trained using [`gisting/src/train.py`](gisting/src/train.py). It is a hydra script and can be run directly as `python -m gisting.src.train` with the parameters defined in [`gisting/src/arguments.py`](gisting/src/arguments.py). It outputs to a directory in `gistlms/` at a path configured [here](gisting/src/conf/config.yaml).

[`gist-train.py`](gist-train.py) is a convenience wrapper of `gisting/src/train.py` for finetuning example gisting models on individual datasets. It defines default hyperparameters for the various datasets, constructs the command to run [`gisting/src/train.py`](gisting/src/train.py) and can initiate multiple runs in parallel. The process to finetune 1 and 3 token gist LMs for all the datasets used in the paper is:

1. Output complete commands and write the parameters for all the experiments to `params.jsonl`:

    ```bash
    python gist-train.py finetune \
    --lm 'flan-t5-base' \
    --datasets "QNLI;MNLI;RTE;SST2;MRPC;QQP;PAWS;CMSQA;COLA;SST5;AGNEWS;SMCALFLOW_CS;MTOP;COGS;GSM8K;DROP;BOOLQ;WANLI;XNLI;MEDNLI;TWEET;PAWSX;ROTTEN_TOMATOES" \
    --initckpts 'vanilla' \
    --n-gists '1,3' \
    --tag 'v3' --paramsfile "params.jsonl" --preview "commands"
    ```

2. Run all the experiments in parallel on multiple GPUs:

    ```bash
    python icl-demo-selection/src/gist-train.py run-expsfile-parallel --paramsfile params.jsonl --gpus 0,1,2,3,4,5,6,7
    ```

The finetuned gist LMs and training logs will be stored in `gistlms/finetunes` at a path configured by `gist-train.py:Experiment.output_dir`. See `gist-train.py:finetune` for detailed usage. The finetuned models for all the datasets used in the paper are provided [here][finetuned-lms-all] and for individual datasets [here][finetuned-lms].

To do ICL evaluation of any new gist models, update `icl/src/exp_utils.py:ds2gistlms`.

### Multi-task Training

To train multi-task models, we first need a multi-task collection. `flan_multi_task.py` is used to create different subsamples of the Flan 2021, 2022 and Flan-mini collections. The subsamples are dumped in `multitask-data/`. The subsample used in the paper is `flan2022_zs_len256_max10K` which comprises up to 10K zero-shot prompts of length < 256 for each task in the [Flan 2022 collection](https://github.com/google-research/FLAN/tree/main/flan/v2). This and many other subsamples can be downloaded from [here][multitask-collections].

The training itself is done using [gisting/src/train.py](gisting/src/train.py) directly. To train training `flan-t5-large` on the `flan2022_zs_len256_max10K` collection with 1 gist and with gradient accumulation over 64 batches of size 4 for an effective batch size of 256:

```bash
python -m gisting.src.train +model=flan-t5-large \
data.dataset_name=FLAN \
data.flan_dataset_name=flan2022_zs_len256_max10K \
training.gist.num_gist_tokens=3 \
training.gist.condition='gist' \
training.num_train_epochs=12 \
training.max_steps=-1 \
training.metric_for_best_model='eval_validation_rougeL' \
training.eval_steps=500 \
training.save_steps=500 \
data.max_eval_samples=1000 \
wandb.tag='adafactor-256-bs256' \
training.bf16=False \
training.bf16_full_eval=False \
training.lora=False \
training.per_device_train_batch_size=4 \
training.per_device_eval_batch_size=4 \
training.gradient_accumulation_steps=64 \
training.lr_scheduler_type='constant' \
training.learning_rate=5e-4 \
training.overwrite_output_dir=False \
training.optim='adafactor' training.logging_steps=50
```

The output directory and wandb tags for these are configured in `config.yaml` and `gisting/src/arguments.py:global_setup()`. The above will write to `gistlms/adafactor-256-bs256-gist-3tok-flan-t5-large-flan2022_zs_len256_max10K`.  For more details see [gisting/README.md](gisting/README.md) or the original [gisting repository](https://github.com/jayelm/gisting).

The trained gist LMs can be directly used to gist and select in-context examples. However, to avoid dealing with the long names, the checkpoints are **copied** to `gistlms/pretrains` and then referenced in `icl/src/exp_utils.py:multitask_pretrained_gistlms`.

The `large` and `xl`-size gist models trained on `flan2022_zs_len256_max10K` with 1, 3, 6 and 15 tokens for the paper are provided [here][multitask-lms-all].

### Running ICL Evaluations

As described above, to run ICL evaluations with the gist models, the paths to the checkpoints need to be configured in `icl/src/exp_utils.py:ds2gistlms` and `icl/src/exp_utils.py:multitask_pretrained_gistlms`. For details of how to run ICL evaluations, see [icl/README.md](https://github.com/Shivanshu-Gupta/in-context-learning#running-icl-evaluations).

### Adding a new dataset

1. Follow the steps in [icl/README.md](https://github.com/Shivanshu-Gupta/in-context-learning#adding-a-new-dataset).
2. For finetuning gist LMs some updates are necessary in `gist-train.py`
   1. Add it to `finetune_datasets`
   2. Add its `TrainingParams` to `ds2params`. Typically only `bs` and `eval_steps` need setting.
   3. Update `get_metric` if needed.

## Command Line Tips

There are two different types of command lines in this repository:
1. [Typer](https://typer.tiangolo.com/) - this one is used for non-nested parameterization. Allows multiple commands in a single script run as `python <script> <command> <arguments>`. The `<command>` only needs to be specified if there are more than one commands (eg. `icl/src/data_params.py`). The `<arguments>` are specified a bit differently so try running with `--help` to see them.
   1. `gist-train.py`
   2. `icl/src/experiments.py`:
   3. `icl/src/run.py`
   4. `icl/src/data_params.py`
2. [Hydra](hydra.cc/) - this one is used for more nested parameterization.
   1. `gisting/src/train.py`: parameters defined in (`gisting/src/arguments.py`). Used to train gist LMs. Only use directly when doing [multi-task training](#multi-task-training). When [fine-tuning](#finetuning-gist-models-on-individual-datasets) use `gist-train.py`.
   2. `icl/src/driver.py`: parameters defined in (`icl/src/params.py:AllParams`)

[gistlms]: https://bbbe-128-195-10-172.ngrok-free.app/gistlms/
[finetuned-lms]: https://bbbe-128-195-10-172.ngrok-free.app/gistlms/finetunes
[finetuned-lms-all]: https://bbbe-128-195-10-172.ngrok-free.app/gistlms/finetunes.tar
[multitask-lms]: https://bbbe-128-195-10-172.ngrok-free.app/gistlms/pretrains
[multitask-lms-all]: https://bbbe-128-195-10-172.ngrok-free.app/gistlms/pretrains.tar
[multitask-collections]: https://bbbe-128-195-10-172.ngrok-free.app/multittask-data/
[icl-datasets]: https://1drv.ms/u/s!AqJNiE6C-nXuoawBxh-3rfUsSf4-8A?e=3o1YDK
[icl-repo]: https://github.com/Shivanshu-Gupta/in-context-learning

## Citation

If you found this repository useful, please cite the following paper:

```bibtex
@article{gupta2023gistscore,
   title={GistScore: Learning Better Representations for In-Context Example Selection with Gist Bottlenecks},
   author={Shivanshu Gupta and Clemens Rosenbaum and Ethan R. Elenberg},
   year={2023},
   eprint={2311.09606},
   archivePrefix={arXiv},
   primaryClass={cs.CL}
}
```
