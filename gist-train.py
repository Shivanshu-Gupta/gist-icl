from __future__ import annotations
import os
import sys
import attr
import time
import queue
import typer
import torch
import jsonlines
from rich import print
from pathlib import Path
from functools import partial
from typing import Optional
from collections import defaultdict
from joblib import Parallel, delayed, parallel_backend

if 'icl-demo-selection/src' not in sys.path:
    sys.path.append('icl-demo-selection/src')
from constants import Dataset as D
from tools.param_impl import Parameters, DictDataClass
from tools.typer_dataclass import dataclass_cli


app = typer.Typer()
q = queue.Queue()

get_ints = lambda s, sep=';': [int(x) for x in s.split(sep)]
get_strings = lambda s, sep=';': [x for x in s.split(sep)]
get_datasets = lambda s, sep=';': [D[x] for x in s.split(sep)]

finetune_datasets = [
    D.SMCALFLOW_CS, D.BREAK, D.MTOP, D.CFQ, D.COGS,
    D.QNLI, D.MNLI, D.RTE,
    D.SST2, D.YELP,
    D.MRPC, D.QQP, D.PAWS,
    D.COMMONGEN, D.E2ENLG, D.DART,
    D.WINOGRANDE, D.WSC,
    D.AESLC, D.AGNEWS,
    D.COLA,
]

@attr.s(auto_attribs=True)
class TrainingParams(DictDataClass):
    bs: str = '9x4'        # batchsize x gradient accumulation steps. Currently all datasets have the effective batch size of 36. Single step batch size depends on the length of inputs.
    epochs: int = 4         # number of epochs. Ignored if max_steps != -1. (huggingface TrainingArguments)
    lr: str = 'constant;5e-5'   # lr scheduler and learning rate
    early_stopping: int = 8 # Number of eval steps to early stop after. See huggingface TrainingArguments.
    max_steps: int = 40000  # maximum number of gradient steps. See huggingface TrainingArguments.
    eval_steps: int = 200   # number of steps to validate after. Currently set to 200 for most datasets and 500 for those with longer training (like Semantic Parsing). See huggingface TrainingArguments.

# QNLI;MNLI;RTE;SST2;YELP;MRPC;QQP;PAWS;COPA;PIQA;WINOGRANDE;WSC;CMSQA;COLA;COMMONGEN;E2ENLG;DART;SST5;AGNEWS;SMCALFLOW_CS;BREAK;MTOP;COGS;
# NL2BASH;WANLI;XNLI;MEDNLI;CONDAQA;GSM8K;DROP;BOOLQ

ds2params: dict[D, TrainingParams] = defaultdict(TrainingParams, {
    D.ALPACA: TrainingParams(bs='10x2', epochs=10, max_steps=-1, eval_steps=1000),
    D.FLAN: TrainingParams(bs='10x4', epochs=4, max_steps=-1, eval_steps=1000),

    D.OVERNIGHT: TrainingParams(bs='6x6', epochs=30, eval_steps=200),
    D.SMCALFLOW_CS: TrainingParams(bs='4x9', epochs=10, eval_steps=500, early_stopping=6),
    D.BREAK: TrainingParams(bs='9x4', epochs=10, eval_steps=500, early_stopping=6),
    D.MTOP: TrainingParams(bs='9x4', epochs=10, eval_steps=500, early_stopping=6),
    D.CFQ: TrainingParams(bs='6x6', epochs=10, eval_steps=500, early_stopping=6),
    D.SPIDER: TrainingParams(bs='6x6', epochs=10, eval_steps=500, early_stopping=6),
    D.COGS: TrainingParams(bs='9x4', epochs=4, eval_steps=500, early_stopping=6),
    D.NL2BASH: TrainingParams(bs='9x4', epochs=10, eval_steps=500, early_stopping=6),

    D.COMMONGEN: TrainingParams(bs='9x4', epochs=2, eval_steps=500, early_stopping=6),
    D.E2ENLG: TrainingParams(bs='9x4', epochs=2, eval_steps=500, early_stopping=6),
    D.DART: TrainingParams(bs='4x9', epochs=2, eval_steps=500, early_stopping=6),

    D.QNLI: TrainingParams(bs='6x6', epochs=2, eval_steps=200),
    D.MNLI: TrainingParams(bs='4x9', epochs=1, eval_steps=200),
    D.RTE: TrainingParams(bs='6x6', epochs=1, eval_steps=200),
    D.WANLI: TrainingParams(bs='6x6', epochs=2, eval_steps=200),
    D.XNLI: TrainingParams(bs='6x6', epochs=1, eval_steps=500),
    D.MEDNLI: TrainingParams(bs='6x6', epochs=4, eval_steps=500),

    D.CONDAQA: TrainingParams(bs='4x9', epochs=4, eval_steps=500),
    D.DROP: TrainingParams(bs='4x9', epochs=4, eval_steps=500),
    D.BOOLQ: TrainingParams(bs='4x9', epochs=2, eval_steps=200),
    D.GSM8K: TrainingParams(bs='4x9', epochs=4, eval_steps=500),

    D.SST2: TrainingParams(bs='9x4', epochs=2, eval_steps=200),
    D.YELP: TrainingParams(bs='3x12', epochs=2, eval_steps=200),
    D.SST5: TrainingParams(bs='6x6', epochs=2, eval_steps=200),
    D.TWEET: TrainingParams(bs='6x6', epochs=5, eval_steps=200),
    D.ROTTEN_TOMATOES: TrainingParams(bs='4x9', epochs=5, eval_steps=200),

    D.MRPC: TrainingParams(bs='9x4', epochs=2, eval_steps=200),
    D.QQP: TrainingParams(bs='4x9', epochs=2, eval_steps=200),
    D.PAWS: TrainingParams(bs='9x4', epochs=2, eval_steps=200),
    D.PAWSX: TrainingParams(bs='9x4', epochs=2, eval_steps=200),

    D.CMSQA: TrainingParams(bs='9x4', epochs=5, eval_steps=200),
    D.COPA: TrainingParams(bs='9x4', max_steps=-1, epochs=20, eval_steps=10, lr='constant;1e-6'),
    D.PIQA: TrainingParams(bs='4x9', epochs=2, eval_steps=200),

    D.WINOGRANDE: TrainingParams(bs='9x4', epochs=2, eval_steps=200),
    D.WSC: TrainingParams(bs='9x4', max_steps=-1, epochs=20, eval_steps=10, lr='constant;1e-6'),

    D.AESLC: TrainingParams(bs='4x9', epochs=2, eval_steps=200),
    D.AGNEWS: TrainingParams(bs='4x9', epochs=2, eval_steps=200),

    D.COLA: TrainingParams(bs='9x4', epochs=20, eval_steps=200),
})
# OVERNIGHT;SMCALFLOW_CS;BREAK;MTOP;CFQ;COGS;QNLI;MNLI;RTE;SST2;YELP;MRPC;QQP;PAWS;COMMONGEN;E2ENLG;DART;WINOGRANDE;WSC;AESLC;AGNEWS;COLA
# SMCALFLOW_CS;BREAK;MTOP;COGS;QNLI;MNLI;RTE;SST2;YELP;MRPC;QQP;PAWS;COMMONGEN;E2ENLG;DART;WINOGRANDE;WSC;AGNEWS;COLA
# SMCALFLOW_CS;BREAK;MTOP;COGS;QNLI;MNLI;RTE;SST2;YELP;MRPC;QQP;PAWS;WINOGRANDE;WSC;AGNEWS;COLA;COMMONGEN;E2ENLG;DART;SST5;CMSQA;COPA;PIQA;AESLC

# QNLI;MNLI;RTE;SST2;YELP;MRPC;QQP;PAWS;COPA;PIQA;WINOGRANDE;WSC;CMSQA;COLA;COMMONGEN;E2ENLG;DART;SST5;AGNEWS;AESLC;SMCALFLOW_CS;BREAK;MTOP;COGS

# SST5;CMSQA
# COPA;PIQA;AESLC
# COMMONGEN;E2ENLG;DART
# COMMONGEN;E2ENLG;DART;SST5;CMSQA;COPA;PIQA;AESLC;
# SENT140;SNLI;SQUAD;MULTIRC;BOOLQ;OBQA;NQ;ARCC

def get_metric(dataset):
    # metric to use for model selection
    if dataset == D.ALPACA:
        return 'unseen_rougeL'
    elif dataset.startswith('flan') or dataset in [
        D.GSM8K, D.DROP, D.FLAN, D.COMMONGEN, D.E2ENLG, D.DART, D.AESLC, D.GIGAWORD
    ]:
        return 'eval_validation_rougeL'
    else:
        return 'eval_validation_accuracy'

pretrained_ckpt_formats = {
    'alpaca': 'gistlms/dist-bf16-gist-{n_gist}tok-flan-t5-large-alpaca-plus/dist-bf16-gist-{n_gist}tok-flan-t5-large-alpaca-plus-run-42',
    'flan_zs_max30K_sel1': 'gistlms/adafactor-256-bs256-gist-{n_gist}tok-flan-t5-large-flan_zs_len{seqlen}_max30K_sel1_train/adafactor-256-bs256-gist-{n_gist}tok-flan-t5-large-flan_zs_len{seqlen}_max30K_sel1_train-run-42'
}
ckptname2dirfn = {
    'alpaca': pretrained_ckpt_formats['alpaca'].format,
    'flan_zs_len256_max20K_sel1': partial(pretrained_ckpt_formats['flan_zs_max30K_sel1'].format, seqlen=256),
    'flan_zs_len512_max20K_sel1': partial(pretrained_ckpt_formats['flan_zs_max30K_sel1'].format, seqlen=512),
}

@attr.s(auto_attribs=True)
class Experiment(Parameters):
    dataset: D
    split: Optional[str] = None
    flan_ds: Optional[str] = None
    lm: str = 'flan-t5-large'
    initckpt: Optional[str] = 'vanilla'  # key from ckptname2dirfn (eg. alpaca) or vanilla
    bf16: bool = False
    lora: bool = False
    n_gist: int = 3
    condition: str = 'gist'
    metric: Optional[str] = None
    eval_steps: Optional[int] = None
    eval_samples: int = 1000
    train_samples: Optional[int] = 72000
    initeval: bool = False
    bs: Optional[str] = None
    tag: str = 'test'
    epochs: Optional[int] = None
    max_steps: Optional[int] = None
    early_stopping: Optional[int] = None
    deepspeed: bool = False
    maxlen: Optional[int] = None
    extra: str = ''
    optim: str = 'adafactor'
    lr: str = 'constant;5e-5'
    overwrite: bool = False
    multiline: bool = False
    gpus: Optional[str] = None      # eg. '0,1,2,3'
    run: bool = False
    evaluate_only: bool = True

    def __attrs_post_init__(self: Experiment):
        if self.is_settings_grid(): return
        if 'test' in self.tag:
            self.overwrite = True

        self.metric = self.metric or get_metric(self.dataset)

        evolve = lambda d, o: attr.evolve(d, **{k: v for k, v in o.items() if v is not None and k in TrainingParams.__annotations__})
        TP = evolve(ds2params[self.dataset], self.to_dict())
        for k, v in TP.to_dict().items():
            setattr(self, k, v)

    def get_init_ckpt(P):
        return ckptname2dirfn[P.initckpt](n_gist=P.n_gist)

    @property
    def output_dir_old(P):
        model = P.lm.split('/')[-1] if P.initckpt == 'vanilla' else P.get_init_ckpt().split('/')[-1]
        group = f'{P.tag}-{P.condition}-{P.n_gist}tok-{model}-{P.dataset.value}'
        name = f'{group}-run-42'
        return Path('exp') / group / name, group, name

    @property
    def output_dir(P):
        if P.dataset in [D.FLAN, D.ALPACA] or P.condition != 'gist':
            return P.output_dir_old
        else:
            lm = P.lm.split('/')[-1]
            name_parts = []
            if P.tag: name_parts.append(P.tag)
            if P.deepspeed: name_parts.append('ds')
            if P.bf16: name_parts.append('bf16')
            name_parts.append(f'{P.initckpt}-{P.n_gist}tok-{lm}')
            name = '-'.join(name_parts)
            output_dir: Path = Path('gistlms/finetunes') / P.dataset.name
            if P.split:
                output_dir /= P.split
            output_dir /= name
            wandb_group = P.dataset.name
            wandb_name = f'{name}-{P.dataset.value}'
            return output_dir, wandb_group, wandb_name

    @property
    def outfile(P):
        return P.output_dir[0] / 'output.log'

    @property
    def completed(P):
        resultsfile = P.output_dir[0] / 'eval_results.json'
        return resultsfile.exists()

    def completed_after(P, timestamp: float) -> bool:
        resultsfile = P.output_dir[0] / 'eval_results.json'
        return resultsfile.exists() and resultsfile.stat().st_mtime > timestamp

    @property
    def cmd(P):
        cmd = []
        if P.deepspeed:
            cmd.append(f'deepspeed --num_gpus=4 --no_local_rank --module gisting.src.train +model={P.lm}')
            cmd.append('training.deepspeed=gisting/ds_configs/stage3.json')
        else:
            cmd.append(f'python -m gisting.src.train +model={P.lm}')

        if P.initckpt != 'vanilla':
            cmd.append(f"model.model_name_or_path='{P.get_init_ckpt()}'")

        cmd.append(f'data.dataset_name={P.dataset.name}')
        if P.split:
            cmd.append(f'data.split={P.split}')
        if P.dataset == D.FLAN and P.flan_ds:
            cmd.append(f'data.flan_dataset_name={P.flan_ds}')

        cmd.append(f"training.gist.num_gist_tokens={P.n_gist}")
        cmd.append(f"training.gist.condition='{P.condition}'")

        if P.early_stopping:
            cmd.append(f"training.early_stopping_patience={P.early_stopping}")
        cmd.append(f"training.metric_for_best_model='{P.metric}'")
        cmd.append(f"training.eval_steps={P.eval_steps} training.save_steps={P.eval_steps}")
        cmd.append(f"data.max_eval_samples={P.eval_samples}")

        if P.dataset in finetune_datasets and P.train_samples:
            cmd.append(f"data.max_train_samples={P.train_samples}")

        cmd.append(f"training.bf16={True if P.bf16 else False}")
        cmd.append(f"training.bf16_full_eval={True if P.bf16 else False}")
        if P.bf16: cmd.append('model.precision="bf16"')

        if P.lora: cmd.append(f"training.lora=True")
        if P.maxlen: cmd.append(f"training.generation_max_length={P.maxlen}")

        bs, accum = P.bs.split('x')
        cmd.append(f"training.per_device_train_batch_size={bs}")
        cmd.append(f"training.per_device_eval_batch_size={bs}")
        cmd.append(f'training.gradient_accumulation_steps={accum}')


        if P.optim != 'adam':
            cmd.append(f"training.optim='{P.optim}'")
        lr_sched, lr = P.lr.split(';')
        cmd.append(f"training.lr_scheduler_type='{lr_sched}'")
        if float(lr) != 5e-5:
            cmd.append(f"training.learning_rate={lr}")

        output_dir, wandb_group, wandb_name = P.output_dir
        if not P.evaluate_only:
            cmd.append(f"training.output_dir='{output_dir}'")
            if P.max_steps == -1:
                cmd.append(f"training.num_train_epochs={P.epochs}")
            else:
                cmd.append(f"training.max_steps={P.max_steps}")
            if P.initeval: cmd.append(f'training.evaluate_before_train=True')
        else:
            cmd.append(f"training.output_dir='{output_dir}/eval/'")
            cmd.append(f"model.model_name_or_path='{output_dir}'")
            cmd.append(f"training.max_steps=1")
            cmd.append(f'training.evaluate_before_train=True')
            cmd.append(f"evaluate_only=True")

        if (P.tag and 'test' in P.tag) or P.evaluate_only:
                cmd.append(f"wandb.log=False")
        else:
            cmd.append(f'wandb.group={wandb_group}')
            cmd.append(f'wandb.name={wandb_name}')
            cmd.append(f"wandb.tag='{P.tag}'")

        if P.overwrite: cmd.append(f"training.overwrite_output_dir=True")
        if P.extra: cmd.append(P.extra)

        if P.multiline:
            return ' \ \n'.join(cmd)
        else:
            return ' '.join(cmd)

def run_cmd(
    cmd, env: dict[str, str] = None, outfile: Path = None,
    tee_output=False, verbose=False, debug=False
):
    import os, shlex, subprocess
    if verbose:
        print(cmd)
        print(f'Logging to: {outfile}')
    if debug: return

    args = shlex.split(cmd)
    env = os.environ | (env or {})
    os.makedirs(outfile.parent, exist_ok=True)
    if outfile:
        if tee_output:
            process = subprocess.Popen(
                args, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            tee = subprocess.Popen(['tee', outfile], stdin=process.stdout)
            process.stdout.close()
            tee.communicate()
        else:
            process = subprocess.Popen(
                args, env=env, stdout=outfile.open('w'), stderr=subprocess.STDOUT)
    else:
        process = subprocess.Popen(args, env=env)
    ret = process.wait()
    return ret

def run_exps_parallel(
    params_l: list[Experiment],
    gpus: str = '0,1,2,3',
    debug: bool = False,
):
    gpus = get_ints(gpus, sep=',')
    # if 'CUDA_VISIBLE_DEVICES' in os.environ:
    #     print(os.environ['CUDA_VISIBLE_DEVICES'])
    #     gpus = [get_ints(os.environ['CUDA_VISIBLE_DEVICES'], ',')[i] for i in gpus]
    print(gpus)
    for gpu in gpus:
        q.put(gpu)

    n_jobs = len(params_l)
    n_concurrent = len(gpus)

    def run_wrapper(i, exp: Experiment):
        print(f'  > {i+1}/{n_jobs} {exp.outfile}')
        gpu = q.get(block=True)
        cmd = exp.cmd
        env = dict(CUDA_VISIBLE_DEVICES=str(gpu))
        print(i + 1, cmd)
        run_cmd(cmd, env, exp.outfile, tee_output=n_jobs == 1, debug=debug)
        q.put(gpu)
        torch.cuda.empty_cache()
        print(f'  < {i+1}/{n_jobs} {exp.outfile}')

    print(f'Running {len(params_l)} jobs...')
    start_time = time.time()
    while params_l:
        with Parallel(n_jobs=n_concurrent, require='sharedmem', verbose=True) as parallel:
            parallel(delayed(run_wrapper)(i, params)
                     for i, params in enumerate(params_l))
        if debug: break
        completion = [p.completed_after(start_time) for p in params_l]
        if not any(completion):
            print('all jobs failed')
            break
        else:
            print(f'Completed {sum(completion)}/{len(params_l)} jobs')
        params_l = [p for p in params_l if not p.completed_after(start_time)]
        if params_l:
            print(f'Rerunning {len(params_l)} failed jobs...')
        else:
            print('All jobs completed')

@app.command()
def run_expsfile_parallel(
    paramsfile: Path = typer.Option('params.jsonl', help='Path to the params file.'),
    gpus: str = '0,1,2,3',
    debug: bool = False,
):
    if not paramsfile.exists():
        print('Params file does not exist...')
        return

    with jsonlines.open(paramsfile, mode='r') as reader:
        params_l = [Experiment.from_dict(p) for p in reader]

    run_exps_parallel(params_l, gpus=gpus, debug=debug)

@app.command()
@dataclass_cli
def main(exp: Experiment):
    """
    Run training for a single dataset with parameters are defined in Experiment class.
    Creates a command to run `gisting/src/train.py`.

    Example ussage: python gist-train.py main SST2 --lm 'flan-t5-large' --n-gist 3 ...
    """
    cmd = exp.cmd
    if exp.run:
        env = dict(CUDA_VISIBLE_DEVICES=exp.gpus) if exp.gpus else None
        run_cmd(' '.join(cmd), env, exp.outfile, tee_output=True)
    else:
        print(cmd)
        print(f'Logging to: {exp.outfile}')
        return cmd, exp.outfile

@app.command()
def pretrain_flan():
    pass

@app.command()
def finetune(
    lm: str = 'flan-t5-large',
    datasets: str = ';'.join([d.name for d in finetune_datasets]),
    initckpts: str = 'vanilla',
    n_gists: str = '3',
    tag: str = '',
    evaluate_only: bool = False,
    only_incomplete: bool = False,
    tiny: bool = False,
    preview: str | None = None, # used in `process_params`
    run: bool = True,  # used in `process_params`
    paramsfile: Path = Path('gistlms/params.jsonl'),    # used in `process_params`
):
    """
    Run finetuning for multiple datasets, number of gist tokens, base LM, etc. on multiple gpus in parallel.
    Internally calls `main()` which runs `gisting/src/train.py`.

    Example Usage:
    ```bash
        python icl-demo-selection/src/gist-train.py finetune --gpus '0,1,2,3' --datasets 'SMCALFLOW_CS;BREAK;MTOP;COGS;QNLI;MNLI;RTE;SST2;YELP;MRPC;QQP;PAWS;COMMONGEN;E2ENLG;DART;WINOGRANDE;WSC;AGNEWS;COLA' --initckpts 'vanilla;alpaca' --n-gists '1;3' --tag 'v2' --only-incomplete --debug
    ```

    Args:
        datasets: list of names from `constants.Dataset` as a ';' separated string
        initckpts: LM checkpoint to start training from. list of keys from `ckptname2dirfn` or "vanilla" for vanilla flan-t5-large, as a ';' separated string
        n_gists: different number of gist tokens to train for. list of integers as a ';' separated string
        gpus: ','-separated list of GPUs to use. This will index into `CUDA_VISIBLE_DEVICES` if set.
        tag:
        only_incomplete: only process the incomplete experiments. completedness tested by `Experiment.completed`
        debug: just print the list of experiments and quit
        tiny: small batch size etc. for testing
    """
    ds2splits = {
        # D.PAWSX: ['fr', 'es', 'de', 'zh'][:-1],
        # D.XNLI: ['fr', 'de', 'ru'],
        # D.TWEET: ['emotion', 'sentiment', 'offensive', 'irony', 'stance'],
        D.PAWSX: ['fr', 'es'],
        D.XNLI: ['de', 'ru'],
        D.TWEET: ['emotion', 'offensive', 'irony', 'stance'],
        D.CFQ: ['mcd1', 'random_split'],
    }
    exp_l: list[Experiment] = []
    for ds in get_datasets(datasets):
        exp_l += Experiment(
            dataset=ds,
            split=ds2splits.get(ds, None),
            n_gist=get_ints(n_gists),
            initckpt=get_strings(initckpts),
            lm=lm,
            tag=tag,
            train_samples=None,
            evaluate_only=evaluate_only,
        ).get_settings()
    print(f'Total {len(exp_l)} experiments...')
    exp_l = [exp for exp in exp_l if not only_incomplete or not exp.completed]
    print(f'Running {len(exp_l)} experiments...')
    if tiny:
        for exp in exp_l:
            exp.tag = 'tiny-test'
            exp.overwrite = True
            # exp.eval_samples = 50
            # exp.logging_steps = 5
            # exp.max_steps = 10
            # exp.eval_steps = 10
            exp.max_steps = 500
            exp.logging_steps = 50
            exp.eval_steps = 100
            exp.eval_samples = 500
    process_params(exp_l, only_incomplete, preview=preview, run=run, paramsfile=paramsfile)

def process_params(
    params_l: list[Experiment], only_incomplete: bool, preview: str,
    run: bool, paramsfile: str
):
    """Process the list of experiment parameters `params_l`.

    Args:
        params_l: list of experiment parameters
        only_prompts: only select demos and generate ICL prompts. (WIP)
        only_incomplete: only experiments that are not completed yet
        preview: just output a property of the experiment parameter (acceptable values: params, exp_path, commands, logfiles).
        run: dump all parameters to a jsonl file to be run using `run.run_exps_parallel`
        paramsfile: jsonl file to dump the parameters
    """
    import jsonlines
    print(f'Total {len(params_l)} experiments...')
    params_to_run: list[Experiment] = []
    for i, params in enumerate(params_l):
        if only_incomplete:
            if params.completed:
                print(f'Skipping experiment {i+1}/{len(params_l)}: {params.exp_path} ...')
                continue
        params_to_run.append(params)

    print(f'Running {len(params_to_run)} experiments...')
    if preview:
        for i, params in enumerate(params_to_run):
            if preview == 'params':
                print(f'\n{i+1}/{len(params_to_run)}:', params)
            elif preview == 'commands':
                print(f'\n{i+1}/{len(params_to_run)}:', params.cmd)
            elif preview == 'outfiles':
                print(f'{i+1}/{len(params_to_run)}:', params.outfile)
            else:
                print(f'Invalid preview option: {preview}')
    if run:
        with jsonlines.open(paramsfile, mode='w') as writer:
            # breakpoint()
            writer.write_all([p.to_dict() for p in params_to_run])

if __name__ == '__main__':
    app()
