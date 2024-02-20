# {
#     "Reading Comprehension": [SQuADv1, BoolQ, MultiRC, OBQA],
#     "Closed-book QA": [ARC-c/e, NQ],
#     "Paraphrase Detection": [MRPC, QQP, Paws Wiki],
#     "Natural Language Inference": [MNLIm/mm, QNLI, SNLI, RTE],
#     "Sentiment Analysis": [SST-2, Yelp, Sentiment140],
#     "Commonsense Reasoning": [COPA, HellaSwag, PIQA],
#     "Coreferenece Resolution": [Winogrande, DPR, WSC273],
#     "Structure to Text": [CommonGen, E2ENLG, DART],
#     "Summarization": [AESLC, AGNews, Gigaword],
#     "Misc": ['COLA', 'QUAC', 'CoQA']
# }

sel1 = [
    "ai2_arc/ARC-Challenge:1.0.0",
    "ai2_arc/ARC-Easy:1.0.0",
    "natural_questions_open:1.0.0",
    "hellaswag:1.1.0",
    "piqa:1.0.0",
    "super_glue/copa:1.0.2",
    "super_glue/wsc.fixed:1.0.2",
    "winogrande:1.1.0",
    "glue/mnli:2.0.0",
    "glue/qnli:2.0.0",
    "snli:1.1.0",
    "super_glue/rte:1.0.2",
    "glue/mrpc:2.0.0",
    "glue/qqp:2.0.0",
    "paws_wiki:1.1.0",
    "bool_q:1.0.0",
    "openbookqa:0.1.0",
    "squad/v1.1:3.0.0",
    "super_glue/multirc:1.0.2",
    "glue/sst2:2.0.0",
    "sentiment140:1.0.0",
    "yelp_polarity_reviews:0.2.0",
    "gem/common_gen:1.1.0",
    "gem/dart:1.1.0",
    "gem/e2e_nlg:1.1.0",
    "aeslc:1.0.0",
    "ag_news_subset:1.0.0",
    "gigaword:1.2.0",
]

sel1_eval = [
    "glue/qnli:2.0.0",
    "yelp_polarity_reviews:0.2.0",
    "piqa:1.0.0",
    "bool_q:1.0.0",
]
sel1_train = set(sel1) - set(sel1_eval)

translate_tasks = [
    'para_crawl_enes',
    'wmt14_translate/fr-en:1.0.0',
    'wmt16_translate/cs-en:1.0.0',
    'wmt16_translate/de-en:1.0.0',
    'wmt16_translate/fi-en:1.0.0',
    'wmt16_translate/ro-en:1.0.0',
    'wmt16_translate/ru-en:1.0.0',
    'wmt16_translate/tr-en:1.0.0',
]

from collections import Counter, defaultdict
from functools import partial
from pathlib import Path
from more_itertools import chunked
from datasets import concatenate_datasets
from tqdm import tqdm
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

def generate_and_save(outputpath, gen_func, overwrite=False):
    if Path(outputpath).exists() and not overwrite:
        return Dataset.load_from_disk(outputpath)
    else:
        output_ds = gen_func()
        output_ds.save_to_disk(outputpath)
        return output_ds

def subsample_task(dataset, size=3e4, version='v0'):
    import numpy.random as npr
    npr.seed(42)
    size = int(size)
    tasks = dataset['task_name']
    task2idxs = defaultdict(list)
    for i, t in enumerate(tasks):
        task2idxs[t].append(i)
    if version == 'v0':
        dataset = concatenate_datasets([
            dataset.select(task2idxs[task]).shuffle(seed=42).select(range(min(len(task2idxs[task]), size)))
            for task in task2idxs
        ])
    else: # this is faster -- use it if ever regenerating the datasets
        task2idxs = {
            task: npr.choice(idxs, size=size, replace=False) if len(idxs) > size else idxs
            for task, idxs in task2idxs.items()
        }
        dataset = concatenate_datasets([dataset.select(idxs) for idxs in tqdm(task2idxs.values())])
    return dataset

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large", use_fast=True)
def get_enc_len(examples):
    inputs = examples['inputs']
    lens = [len(enc) for enc in tokenizer(inputs).input_ids]
    return {'length': lens}

def get_enc_len_targets(examples):
    inputs = examples['targets']
    lens = [len(enc) for enc in tokenizer(inputs).input_ids]
    return {'target_length': lens}

# FLAN MINI
if not Path('data/flan_mini/flan_mini').exists():
    import datasets
    import json
    flan_mini = json.load(open('data/flan_mini.json/flan_mini.json'))
    flan_mini = [dict(id=e['id'], source=e['source'], inputs=e['conversations'][0]['value'], targets=e['conversations'][1]['value'])
                    for e in flan_mini if len(e['conversations']) == 2 and e['id'].startswith('identity')]
    features = datasets.Features({
        "id": datasets.Value("string"),
        "source": datasets.Value("string"),
        "inputs": datasets.Value("string"),
        "targets": datasets.Value("string"),
    })
    flan_mini = Dataset.from_list(flan_mini, features=features)
    flan2022: Dataset = flan_mini.map(get_enc_len, batched=True, batch_size=2048)
    flan_mini.save_to_disk('data/flan_mini/flan_mini')
else:
    flan_mini = Dataset.load_from_disk('data/flan_mini/flan_mini')
flan_mini_len256 = generate_and_save('data/flan_mini/flan_mini_len256',
    lambda: flan_mini.filter(lambda x: len(x['inputs']) <= 256))
flan_mini_len512 = generate_and_save('data/flan_mini/flan_mini_len512',
    lambda: flan_mini.filter(lambda x: len(x['inputs']) <= 512))


# FLAN 2022
if not Path('data/flan2022/flan2022').exists():
    flan2022: Dataset = load_dataset('conceptofmind/FLAN_2022')['train']
    flan2022: Dataset = flan2022.map(get_enc_len, batched=True, batch_size=2048)
    flan2022.save_to_disk('data/flan2022/flan2022')
else:
    flan2022: Dataset = Dataset.load_from_disk('data/flan2022/flan2022')
flan2022_zs = generate_and_save('data/flan2022/flan2022_zs',
    lambda: flan2022.filter(lambda x: x['template_type'].startswith('zs')))

flan2022_len256 = generate_and_save('data/flan2022/flan2022_len256',
    lambda: flan2022.filter(lambda x: x['length'] <= 256))
flan2022_len256_max30K = generate_and_save('data/flan2022/flan2022_len256_max30K',
    partial(subsample_task, flan2022_len256, version='v0'))
flan2022_zs_len256 = generate_and_save('data/flan2022/flan2022_zs_len256',
    lambda: flan2022_zs.filter(lambda x: x['length'] <= 256))
flan2022_zs_len256 = generate_and_save('data/flan2022/flan2022_zs_len256',
    lambda: flan2022_zs.filter(lambda x: x['length'] <= 256))
flan2022_zs_len256 = flan2022_zs_len256.map(get_enc_len_targets, batched=True, batch_size=2048)
flan2022_zs_len256_max30K = generate_and_save('data/flan2022/flan2022_zs_len256_max30K',
    partial(subsample_task, flan2022_zs_len256, version='v0'))
flan2022_zs_len256_max10K = generate_and_save('data/flan2022/flan2022_zs_len256_max10K',
    partial(subsample_task, flan2022_zs_len256, size=10000, version='v1'))

flan2022_len512 = generate_and_save('data/flan2022/flan2022_len512',
    lambda: flan2022.filter(lambda x: x['length'] <= 512))
flan2022_len512_max30K = generate_and_save('data/flan2022/flan2022_len512_max30K',
    partial(subsample_task, flan2022_len512, version='v0'))
flan2022_zs_len512 = generate_and_save('data/flan2022/flan2022_zs_len512',
    lambda: flan2022_zs.filter(lambda x: x['length'] <= 512))
flan2022_zs_len512_max30K = generate_and_save('data/flan2022/flan2022_zs_len512_max30K',
    partial(subsample_task, flan2022_zs_len512, version='v0'))
flan2022_zs_len512_max10K = generate_and_save('data/flan2022/flan2022_zs_len512_max10K',
    partial(subsample_task, flan2022_zs_len512, size=10000, version='v1'))


niv2: Dataset = load_dataset('conceptofmind/niv2_submix_original')['train']
niv2: Dataset = niv2.map(get_enc_len, batched=True, batch_size=2048)
niv2.save_to_disk('data/niv2')


# FLAN 2021
if not Path('data/flan2021/flan').exists():
    flan: Dataset = load_dataset('conceptofmind/flan2021_submix_original')['train']
    tasks = flan['task_name']
    inputs = flan['inputs']
    lens = []
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large", use_fast=True)
    for batch in tqdm(chunked(inputs, 2048), total=len(inputs) // 2048):
        lens.extend([len(enc) for enc in tokenizer(batch).input_ids])
    flan = flan.add_column('length', lens)
    flan.save_to_disk('data/flan2021/flan')
else:
    flan: Dataset = Dataset.load_from_disk('data/flan2021/flan')

flan_zs = generate_and_save('data/flan2021/flan_zs',
    lambda: flan.filter(lambda x: x['template_type'].startswith('zs')))
flan_len256 = generate_and_save('data/flan2021/flan_len256',
    lambda: flan.filter(lambda x: x['length'] <= 256))
flan_len256_max30K = generate_and_save('data/flan2021/flan_len256_max30K',
    partial(subsample_task, flan_len256, version='v0'))
flan_zs_len256 = generate_and_save('data/flan2021/flan_zs_len256',
    lambda: flan_zs.filter(lambda x: x['length'] <= 256))
flan_zs_len256_max30K = generate_and_save('data/flan2021/flan_zs_len256_max30K',
    partial(subsample_task, flan_zs_len256, version='v0'))
flan_zs_len256_max30K_notranslate = generate_and_save('data/flan2021/flan_zs_len256_max30K_notranslate',
    lambda: flan_zs_len256_max30K.filter(lambda x: x['task_name'] not in translate_tasks))
flan_zs_len256_max30K_sel1 = generate_and_save('data/flan2021/flan_zs_len256_max30K_sel1',
    lambda: flan_zs_len256_max30K.filter(lambda x: x['task_name'] in sel1))
flan_zs_len256_max30K_sel1_train = generate_and_save('data/flan2021/flan_zs_len256_max30K_sel1_train',
    lambda: flan_zs_len256_max30K.filter(lambda x: x['task_name'] in sel1_train))
flan_zs_len256_max30K_sel1_eval = generate_and_save('data/flan2021/flan_zs_len256_max30K_sel1_eval',
    lambda: flan_zs_len256_max30K.filter(lambda x: x['task_name'] in sel1_eval))

flan_len512 = generate_and_save('data/flan2021/flan_len512',
    lambda: flan.filter(lambda x: x['length'] <= 512))
flan_len512_max30K = generate_and_save('data/flan2021/flan_len512_max30K',
    partial(subsample_task, flan_len512, version='v0'))
flan_zs_len512 = generate_and_save('data/flan2021/flan_zs_len512',
    lambda: flan_zs.filter(lambda x: x['length'] <= 512))
flan_zs_len512_max30K = generate_and_save('data/flan2021/flan_zs_len512_max30K',
    partial(subsample_task, flan_zs_len512, version='v0'))
flan_zs_len512_max30K_notranslate = generate_and_save('data/flan2021/flan_zs_len512_max30K_notranslate',
    lambda: flan_zs_len512_max30K.filter(lambda x: x['task_name'] not in translate_tasks))
flan_zs_len512_max30K_sel1 = generate_and_save('data/flan2021/flan_zs_len512_max30K_sel1',
    lambda: flan_zs_len512_max30K.filter(lambda x: x['task_name'] in sel1))
flan_zs_len512_max30K_sel1_train = generate_and_save('data/flan2021/flan_zs_len512_max30K_sel1_train',
    lambda: flan_zs_len512_max30K.filter(lambda x: x['task_name'] in sel1_train))
flan_zs_len512_max30K_sel1_eval = generate_and_save('data/flan2021/flan_zs_len512_max30K_sel1_eval',
    lambda: flan_zs_len512_max30K.filter(lambda x: x['task_name'] in sel1_eval))
