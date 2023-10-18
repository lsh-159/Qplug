# Replug + Soft Contrastive Learning

This repository contains pre-trained models, corpora, indices, and code for pre-training, finetuning, retrieving and evaluating for the paper [Atlas: Few-shot Learning with Retrieval Augmented Language Models](https://arxiv.org/pdf/2208.03299.pdf) and [REPLUG: Retrieval-Augmented Black-Box Language Models](https://arxiv.org/pdf/2301.12652.pdf), and [Contriever : Unsupervised Dense Information Retrieval with Contrastive Learning](https://arxiv.org/pdf/2112.09118.pdf)  for reproducing the results and doing advanced experiments.


We perform evaluations on a wide range of tasks, including MMLU, KILT and NaturalQuestions, and
study the impact of the content of the document index, showing that it can easily be updated.


This repository supports pretraining and finetuning, for *both* large and small datasets. This repository can be supports the following features:

* Performing end-to-end retrieval-augmented training over a user-supplied corpus of passages (tested with up to 400M passages, ~40B words) with retrieval-in-the-training-loop
* Support for training on Masked-Language modelling, prefix-language modelling, wikipedia section generation, Open-Domain Question Answering, Multiple Choice Question Answering, Fact checking, and KILT (arbitrary seq2seq tasks can also be supported)
  
* A fast, parallel distributed GPU-based exact and approximate maximum inner product search for dense vector retrieval
* Support for fast in-place index refreshes 
* Various memory optimizations and methods for maintaining fast and accurate retrieval while training retrievers in-the-loop.


## Table of Contents

* [Installation](#installation)
* [Getting Started and Codebase at a Glance](#getting-started-and-codebase-at-a-glance)
* [Available Data and Models for download](#available-data-and-Models-for-download)
  * [Corpora](#corpora)
  * [Models](#models)
  * [Pre-built Indices](#prebuilt-indices)
* [Tasks](#tasks)
  * [Basic](#base-task)
  * [Masked Language Modelling](#mlm-task)
  * [Wikipedia Section Generation](#section-task)
  * [Open-Domain Question Answering (e.g. NaturalQuestions, TriviaQA, TempLama)](#qa-task)
  * [Multiple Choice Question Answering (e.g. MMLU)](#mcqa-task)
  * [Fact Checking](#fever-task)
  * [KILT](#kilt-task)
* [Retrieval and Index Details](#retrieval-and-index-details)
  * [Flat vs Faiss](#flat-vs-faiss)
  * [Index Saving and Loading](#index-saving-and-loading)
  * [Strategies for dealing with stale indices](#strategies-for-dealing-with-stale-indices)
    * [Index Refresh](#strategies-for-dealing-with-stale-indices)
    * [Over-Retrieve with Reranking](#strategies-for-dealing-with-stale-indices)
    * [Query-Side Finetuning](#strategies-for-dealing-with-stale-indices)
  * [Retrieve-only mode](#retrieve-only-mode)
  * [Using pre-retrieved or cached passages](#using-pre-retrieved-or-cached-passages)
* [Other features](#other-features)
  * [Closed book mode](#closed-book-mode)
  * [Specifying formats](#specifying-formats)
  * [Implementing your own task](#implementing-your-own-task)
* [Full list of command line flags](#full-list-of-command-line-flags)
* [Citing](#citing)
* [LICENSE](#license)
  * [Code License:](#code-license)
  * [Data License:](#data-license)


## Installation

The Atlas codebase uses the following dependencies:

* python 3 (tested with 3.8)
* fairscale (tested with 0.4.6)
* transformers (tested with 4.18.0)
* numpy (tested with 1.22.4)
* faiss (tested with 1.7.2)

We recommend installing using conda. The following will install all dependencies:
```
git clone https://github.com/facebookresearch/atlas.git
cd atlas
conda create --name atlas-env python=3.8
conda activate atlas-env
conda install pytorch==1.11.0 cudatoolkit=11.3 -c pytorch
conda install -c pytorch faiss-gpu=1.7.2 cudatoolkit=11.3
pip install -r requirements.txt
```

## Getting Started and Codebase at a Glance


The biggest difference to most standard NLP training codebases is that This code (based on facebookai/atlas) performs retrieval on-the-fly, and can refresh its retrieval embeddings index in-place.
This is achieved using a custom-designed distributed GPU index, which automatically handles fast and scale-able retrieval.

Notice : 
* All data files (retriever passages and train/dev/test data) should be supplied in the form of [jsonlines](https://jsonlines.org/) ("jsonl") files.
* Passages to retrieve from should consist of json-serialized objects with `text` and `title` text fields, one passage per line.
* Example passage files are available for wikipedia (see [corpora](#corpora)).
* Train/dev/test data files should be json-serialized objects, one instance per line. The name of the fields is task dependent (covered in detail in [Tasks](#tasks)), but e.g. for NaturalQuestions, the required fields are `question` (a question string) and `answers` (a list of reference answer strings)


실험 코드		
* assumes 4 nodes, each with 8 GPUs

```bash
DATA_DIR=./project_data  			# 중요(수정필요) : 데이터 폴더 이름

# 1. prepare_qa.py : nq, tqa 를 다운해서 train,val,test set({query : str, answer : str}) 을 {DATA_DIR}/data/nq_data/train.jsonl 경로에 저장
python preprocessing/prepare_qa.py --output_directory ${DATA_DIR}/data/

# 2. download_corpus.py : wiki 2018 corpus를 다운해서 {DATA_DIR} 경로에 저장
python preprocessing/download_corpus.py --corpus corpora/wiki/enwiki-dec2018 --output_directory ${DATA_DIR} 

port=$(shuf -i 15000-16000 -n 1)
TRAIN_FILE="${DATA_DIR}/data/nq_data/train.64-shot.jsonl"
EVAL_FILES="${DATA_DIR}/data/nq_data/dev.jsonl"
SAVE_DIR=${DATA_DIR}/experiments/
EXPERIMENT_NAME=my-nq-64-shot-example		# 중요(수정필요) : 실험 이름
TRAIN_STEPS=30

# 3. experiment_1_checkstat.py : top k passage들의 soft contrastive loss 통계 값 확인, {SAVE_DIR}/${EXPERIMENT_NAME}/run.log 경로에 로그 저장
#
#   3.1 retriever_model =  huggingface.from_pretrained(Contriever) 를 불러와서 wiki 2018 passage들을 전부 임베딩
#	time_cost : TBU
#   3.2 임베딩들을 DistributedFaissIndex 인스턴스에 저장, 인스턴스는 {SAVE_DIR}/${EXPERIMENT_NAME}에 백업 (더이상 3.1, 3.2 시간 소모 X)
#	time_cost : TBU
#   3.3 llamaforcpp 로 llama2 q5 인퍼런스 logit 배치 통계량 확인
# 	time_cost : TBU

```bash
srun python evaluate.py \
    --name 'my-nq-64-shot-example-evaluation' \
    --generation_max_length 16 \
    --gold_score_mode "pdist" \
    --precision fp32 \
    --reader_model_type google/t5-${size}-lm-adapt \
    --text_maxlength 512 \
    --model_path ${SAVE_DIR}/${EXPERIMENT_NAME}/checkpoint/step-30 \ #now, we point this to the model we just trained
    --eval_data "${DATA_DIR}/data/nq_data/dev.jsonl ${DATA_DIR}/data/nq_data/test.jsonl" \ # lets evaluate on the dev data and the test data this time
    --per_gpu_batch_size 1 \
    --n_context 40 --retriever_n_context 40 \
    --checkpoint_dir ${SAVE_DIR} \
    --main_port $port \
    --index_mode "flat"  \
    --task "qa" \
    --load_index_path ${SAVE_DIR}/${EXPERIMENT_NAME}/saved_index\ # rather than re-embed all the wikipedia passages again, lets load them from the index we just saved above
    --write_results # write the inference results
```
This script will load the model, and since we specified to load a saved index via `--load_index_path`, it will load an index rather than embed from passages as before. 
It will then evaluate the development and test sets.
Inspecting the saved logs at `${SAVE_DIR}/my-nq-64-shot-example-evaluation/run.log`, we will see the same exact match score for the dev set that we got before, and a test score of ~38 (in our case 38.8 EM).

The rest of this readme describes data, code and functionality in detail.

## Available Data and Models for download

Atlas's wikipedia corpora, the pretrained models and pre-built wikipedia indices are available for download at this time. 

Click to expand:
<details>
<summary>
<h4 name="corpora">Corpora</h4>
</summary>

The preprocessed wikipedia dumps we use for retrieving and pretraining Atlas can be downloaded as follows:

```bash
python preprocessing/download_corpus.py --corpus {corpus download key} --output_directory ${DATA_DIR} 
```
The above string will download a corpus and unzip it to `${DATA_DIR}/{corpus download key}` 

The available corpora are given below:

| Corpus Name      | Corpus Download Key | Description | Size |
| ----------- | ----------- | --------|  ---- |
| enwiki-dec2017      | `corpora/wiki/enwiki-dec2017` | Wikipedia dump from Dec 2017, preprocessed into passages       |  30.4M (26.9M text, 2.7M  infobox)| 
| enwiki-dec2018      | `corpora/wiki/enwiki-dec2018` | Wikipedia dump from Dec 2018, preprocessed into passages (recommended for NQ, TriviaQA) | 32.1M (28.4M text, 3.7M infobox) |
| enwiki-aug2019      | `corpora/wiki/enwiki-aug2019` |  Wikipedia dump from August 2019, preprocessed into passages       | 33.1M (29.4M text, 3.8M infobox)  |
| enwiki-dec2020      | `corpora/wiki/enwiki-dec2020` |  Wikipedia dump from Dec 2020, preprocessed into passages       | 35.6M (31.5M text, 4.1M infobox) |
| enwiki-dec2021      | `corpora/wiki/enwiki-dec2021` | Wikipedia dump from Dec 2021, preprocessed into passages       | 37.5M (33.1M text, 4.3M infobox) |

Passage files are jsonl formatted, with one passage serialized as a json object per line. By default, each passage should be formatted as follows:

```python
{
    "id": "0", # passages should have a unique id
    "title": "Orchid", # should specify the title of the page passage comes from (can be empty string if there's no good title)
    "text": "Orchids are easily distinguished from other plants, as they share some very evident derived characteristics or synapomorphies. Among these are: bilateral symmetry of the flower (zygomorphism), many resupinate flowers, a nearly always highly modified petal (labellum), fused stamens and carpels, and extremely small seeds.", # main text of passage
    "section": "Description" # Optional, section title, if non empty this field is appended to the title as {title}: {section} by default
    ... # you can have other fields you want to keep around for ease of analysis, but they wont actually be used
}
```

Creating your own passage files to use with Atlas should be straightforward if you follow the above formatting.

We cannot open-source the common-crawl indices used in the paper at this time.
</details>

<details>
<summary>
<h4 name="models">Models</h4>
</summary>

We are open-sourcing pretrained Atlas models at base, large, xl and xxl sizes. These include both the pretrained retriever and reader weights.
In addition, we're open-sourcing our strongest-performing fully-finetuned NaturalQuestions Atlas models, for users who want to perform state-of-the-art QA inference (or finetune them on other QA tasks).
Models can be downloaded as follows:

```bash
python preprocessing/download_model.py --model {model download key} --output_directory ${DATA_DIR} 
```

This will download the requested model to `${DATA_DIR}/{model download key}`, and it can then be used in scripts by passing `${DATA_DIR}/{model download key}` to `--model_path`.
The following table details the available models:

| Model | Model Download Key | Description | Parameters (reader / retriever) |
| ----------- | ----------- | --------| ----|
| Atlas-xxl | `models/atlas/xxl` | Pretrained Atlas XXL model | 11B / 110M |
| Atlas-xl | `models/atlas/xl` | Pretrained Atlas XL model | 3B / 110M |
| Atlas-large | `models/atlas/large` | Pretrained Atlas large model | 770M / 110M |
| Atlas-base | `models/atlas/base` | Pretrained Atlas base model | 220M / 110M |
| NQ-finetuned Atlas-xxl | `models/atlas_nq/xxl` |Atlas XXL model, finetuned on Natural Question | 11B / 110M |
| NQ-finetuned Atlas-xl | `models/atlas_nq/xl` | Atlas XL model, finetuned on Natural Question | 3B / 110M |
| NQ-finetuned Atlas-large | `models/atlas_nq/large` | Atlas large model, finetuned on Natural Question | 770M / 110M |
| NQ-finetuned Atlas-base | `models/atlas_nq/base` |Atlas base model, finetuned on Natural Question| 220M / 110M |
</details>

<details>
<summary>
<h4 name="prebuilt-indices">Pre-built Indices</h4>
</summary>

Atlas will automatically build an index if none is provided. This is convenient, but can take a long time, especially with fewer GPU workers, or if the index is very large.

We have therefore made precomputed indices available for download for the wiki-dec2018 corpus for the pretrained Atlas checkpoints, and for nq-finetuned Atlas checkpoints

These can be downloaded as follows :
```bash
python preprocessing/download_index.py --index {index download key} --output_directory ${DATA_DIR} 
```

The above script will download the requested pretrained index and save them to `${DATA_DIR}/{index download key}`. 
They can then be used in training or evaluation by passing them to `--load_index_path`. 
More details on index saving and loading are given in [Retrieval and Index Details](#retrieval-and-index-details). 
The following indices are available for download:

| Index  | Index Download Key | Corresponding Model |  Description |
| --------| ------| --------| ------|
| Atlas XXL wiki-dec2018 index | `indices/atlas/wiki/xxl` | `models/atlas/xxl` | Precomputed index for the wiki-dec2018 corpus for the pretrained Atlas-xxl model |
| Atlas XL wiki-dec2018 index | `indices/atlas/wiki/xl` | `models/atlas/xl` | Precomputed index for the wiki-dec2018 corpus for the pretrained Atlas-xl model |
| Atlas large wiki-dec2018 index | `indices/atlas/wiki/large` | `models/atlas/large` | Precomputed index for the wiki-dec2018 corpus for the pretrained Atlas-large model |
| Atlas base wiki-dec2018 index | `indices/atlas/wiki/base` | `models/atlas/base` | Precomputed index for the wiki-dec2018 corpus for the pretrained Atlas-base model |
| Atlas-nq XXL wiki-dec2018 index | `indices/atlas_nq/wiki/xxl` | `models/atlas_nq/xxl` | Precomputed index for the wiki-dec2018 corpus for the natural-questions-finetuned Atlas xxl model |
| Atlas-nq XL wiki-dec2018 index | `indices/atlas_nq/wiki/xl` | `models/atlas/xl` | Precomputed index for the wiki-dec2018 corpus for the natural-questions-finetuned Atlas xl model |
| Atlas-nq large wiki-dec2018 index | `indices/atlas_nq/wiki/large` | `models/atlas/large` | Precomputed index for the wiki-dec2018 corpus for the natural-questions-finetuned Atlas large model |
| Atlas-nq base wiki-dec2018 index | `indices/atlas_nq/wiki/base` | `models/atlas/base` | Precomputed index for the wiki-dec2018 corpus for the natural-questions-finetuned Atlas base model |
</details>

## Tasks

Atlas can train (or evaluate) on any supervised learning task which can be formulated in a "seq2seq" format, where there is a sequence of 1 or more tokens comprising an input *query* and a sequence of 1 or more tokens comprising an output *target*.
For example, a query might be a question, `Where is the Bermuda Triangle?`, and a target might be the answer to that question, `Western part of the North Atlantic Ocean`.
This way of modelling will be familiar to users of models like T5 or BART. Anywhere these models could be used, Atlas can be used too, using the exact same data: Atlas will learn to retrieve passages from its retrieval index by itself - annotations for associating passages to (`query`, `target`) pairs are not used.

The Atlas codebase configures what task it is doing, and what evaluation metrics to call using the `--task` command line argument. 
We have implemented a `base` task, with only the most basic support for seq2seq training, but provide more fully-featured functionality for Masked Language Modelling (`mlm`), Language Modelling (`lm`), Wikipedia section generation (`section`), Open-domain QA (`QA`), Multiple choice QA (`multiple_choice`), fact checking (`fever`), and the KILT suite (`kilt`), 
All tasks expect input data formatted as jsonl format, but the specific field names are task specific. Some tasks have additional command line args, and specialized evaluation.
Adding new tasks is straightforward, and described [here](#defining-your-own-task).

The tasks are described in more detail below, and most have example commands in `examples/{task}/` (click to expand).

<details>
<summary>
<h4 name="base-task">Base Task</h4>
</summary>


This is the most basic task available, and is probably not the best option for you, especially if your task closely resembles one the other implemented tasks.

Specify this task by passing `--task base` to either `train.py` or `evaluate.py` 

Train/validation/test data for this task should consist of jsonl files, which should be passed to `train.py` or `evaluate.py` space-separated lists to `--train_data train_file_1.jsonl train_file_2.jsonl`, and `--eval_data eval_file_1.jsonl eval_file_2.jsonl` etc.
This task expects input files to have a `query` field with the input query string and a `target` field with the output query string, e.g.:

```json
{"query": "input to Atlas", "target": "desired generation from Atlas"}
```

The evaluation loop will calculate evaluation loss and the fraction of eval data examples where Atlas generates an output that exactly matches the target.
If you pass `--write_results` to the script, Atlas predictions on the eval data will be written to the save checkpoint directory with the following format:

```json
{"query": "input to Atlas", "answers": ["desired generation from Atlas"], "generation": "Atlas's prediction for the query", "passages": ["list of retrieved passages"]}
```

</details>

<details>
<summary>
<h4 name="mlm-task">Masked Language Modelling</h4>
</summary>

The Masked Language modelling task implements the Masked Language Modelling pretraining task as introduced by [T5](https://arxiv.org/abs/1910.10683).
This is the task we use to pretrain the main Atlas in the paper.

Specify this task by passing `--task mlm` to `train.py`.

Train/validation/test data for this task should consist of jsonl files, which should be passed to `train.py`  as `--train_data train_file_1.jsonl train_file_2.jsonl`, and `--eval_data eval_file_1.jsonl eval_file_2.jsonl` etc.
These files should be comprised of JSON objects with the following format:
```python
{
  "text": "text passage to apply noise to and train to de-noise",
  "id": "unique id of text passage"
  ... # you can have other fields you want to keep around for ease of analysis, but they wont actually be used
}
```
The intention is that the same files that you use for the retrieval corpus, (passed to `--passages`) can be used as training data.
The task will apply the T5 noise function to `text` field, to automatically create inputs and target generations.

The MLM task will prevent Atlas from retrieving the passage that it is trying to de-noise. It does this by filtering out any passage from retrieved results which have same `id` field as the instance Atlas is de-noising. 
This functionality is important if the de-noising training data and the passages Atlas is retrieving from are the same corpus.

This task has the following task specific args:
```
  --mlm_noise_density MLM_NOISE_DENSITY
      how much of an input text should be masked by masking spans (default: 0.15)
  --mlm_mean_noise_span_length MLM_MEAN_NOISE_SPAN_LENGTH
      average length of an MLM masking span (default: 3)
  --min_words_per_lm_instance MIN_WORDS_PER_LM_INSTANCE
      Instances with fewer than min_words_per_lm_instance instances will be skipped for MLM/LM/Section generation (default: None)
```

If you pass `--write_results`, Atlas will write its mask-filling predictions to file.

Atlas will log the following evaluation metrics for MLM during its evaluation loop: 
* `eval_loss`: evaluation reader loss of generated mlm mask-fill spans
* `accuracy`: fraction of perfectly de-noised mask-fill spans
* `f1`: token f1 fraction of correct de-noised mask-fill spans
* `rouge_1`: rouge 1 score of generated mask-fill spans relative to the gold reference masked spans
* `rouge_2`: rouge 2 score of generated mask-fill spans relative to the gold reference masked spans
* `rouge_L`: rouge L score of generated mask-fill spans relative to the gold reference masked spans

</details>

<details>
<summary>
<h4 name="lm-task">Language Modelling</h4>
</summary>

Atlas can be trained to do Left-to-Right Language Modeling by passing `--task lm` to `train.py`.

Train/validation/test data for this task should consist of jsonl files, which should be passed to `train.py`  as `--train_data train_file_1.jsonl train_file_2.jsonl`, and `--eval_data eval_file_1.jsonl eval_file_2.jsonl` etc.
These files should be comprised of JSON objects with the following format:
```python
{
  "text": "text passage to train Atlas to generate",
  "id": "unique id of text passage"
  ... # you can have other fields you want to keep around for ease of analysis, but they wont actually be used
}
```
The intention is that the same files that you use for the retrieval corpus, (passed to `--passages`) can be used as training data.
The task will preprocess the `text` field automatically, dividing it into two random segments - the left part serves as conditioning context, and the right part is the text the Atlas model will be trained to generate as a continuation.

The LM task will prevent Atlas from retrieving the same passage that it is trying to generate. It does this by filtering out any passage from retrieved results which have same `id` field as the instance Atlas is generating. 
This functionality is important if the de-noising training data and the passages Atlas is retrieving from are the same corpus.

This task has the following task specific args:
```
  --min_words_per_lm_instance MIN_WORDS_PER_LM_INSTANCE
      Instances with fewer than min_words_per_lm_instance instances will be skipped for  MLM/LM/Section generation (default: None)
  --min_lm_context_ratio MIN_LM_CONTEXT_RATIO
      Splits text into two segments for language modelling.' 'Left segment is conditioning context, right segment is for generating.' 'The left segment must be more than min_lm_context_ratio of
      the right segment (default: 0.5)
  --max_lm_context_ratio MAX_LM_CONTEXT_RATIO
      Splits text into two segments for language modelling.' 'Left segment is conditioning context, right segment is for generating.' 'The left segment must be less than max_lm_context_ratio
      of the right segment (default: 0.5)
```

If you pass `--write_results`, Atlas will write its lm predictions to file.

Atlas will log the following evaluation metrics for LM during its evaluation loop: 
* `eval_loss`: evaluation reader loss of continuations for the reference data
* `accuracy`: fraction of perfectly predicted continuations
* `f1`: token f1 fraction of correct generated continuations
* `rouge_1`: rouge 1 score of generated continuations relative to the gold reference continuations
* `rouge_2`: rouge 2 score of generated continuations relative to the gold reference continuations
* `rouge_L`: rouge L score of generated continuations relative to the gold reference continuations

</details>

<details>
<summary>
<h4 name="section-task">Wikipedia Section Generation</h4>
</summary>

Atlas can be trained to generate the text of a wikipedia passage given its title and section title, by passing  `--task section` to `train.py`.

Train/validation/test data for this task should consist of jsonl files, which should have the form of the `text-list-100-sec.jsonl` files in the wikipedia dumps.
These can be obtained by following the instructions in [Available Data and Models for download](#available-data-and-Models-for-download), for example the training file: `enwiki-dec2018/text-list-100-sec.jsonl`.
These files should be comprised of JSON objects, one per line, with the following format:
```json
{
  "id": "3793043", 
  "title": "Bermuda Triangle",
  "section": "Compass variations",
  "text": " Compass problems are one of the cited phrases in many Triangle incidents. While some have theorized that unusual local magnetic anomalies may exist in the area, such anomalies have not been found. Compasses have natural magnetic variations in relation to the magnetic poles, a fact which navigators have known for centuries."
}
```
The task will automatically format the input query to the model as "{Title}, {Section}" - e.g. in this example, the input to Atlas will be constructed as `Bermuda Triangle, Compass Variations`. The output will be the `text` field of the example.
The `section` task will prevent Atlas from retrieving the same passage that it is trying to generate. It does this by filtering out any passage from retrieved results which have same `id` field as the instance Atlas is generating. 

This task has the following task specific args:
```
  --min_words_per_lm_instance MIN_WORDS_PER_LM_INSTANCE
      Instances with fewer than min_words_per_lm_instance instances will be skipped for MLM/LM/Section generation (default: None)
```
If you pass `--write_results`, Atlas will write its generated predictions for the text for Wikipedia sections to file.

Atlas will log the following evaluation metrics for `section` during its evaluation loop: 
* `eval_loss`: evaluation reader loss of continuations for the reference data
* `accuracy`: fraction of perfectly predicted continuations
* `f1`: token f1 fraction of correct generated continuations
* `rouge_1`: rouge 1 score of generated continuations relative to the gold reference continuations
* `rouge_2`: rouge 2 score of generated continuations relative to the gold reference continuations
* `rouge_L`: rouge L score of generated continuations relative to the gold reference continuations

</details>


<details>
<summary>
<h4 name="qa-task">Open-Domain Question Answering (e.g. NaturalQuestions, TriviaQA, TempLama)</h4>
</summary>

Atlas can be trained to answer open-domain QA questions by passing `--task qa` to `train.py` or `evaluate.py`.
There is a worked example of QA in the [Getting Started and Codebase at a Glance](#getting-started-and-codebase-at-a-glance) section.
We use this task for the NaturalQuestions, TriviaQA and TempLama datasets in the paper.

Train/validation/test data for this task should consist of jsonl files, which should be passed to `train.py`  as `--train_data train_file_1.jsonl train_file_2.jsonl`, and `--eval_data eval_file_1.jsonl eval_file_2.jsonl` etc.
Files should have one JSON instance per line with the following format:
```python
{
  "question": "where is the bermuda triangle",
  "answers": ["Western part of the North Atlantic Ocean"],
   ... # you can have other fields you want to keep around for ease of analysis, but they wont actually be used
}
```
The question will be formatted according to the task specific argument `--qa_prompt_format`, which defaults to `question: {question} answer: <extra_id_0>`.
For example above, the question would be automatically formatted as input queries to Atlas as `question: where is the bermuda triangle answer: <extra_id_0>`.
The supervision target is obtained from the `target` field. If this field does not exist, the supervision target will get selected at random from the available answers in the `answers` field, and formatted as `<extra_id_0> {answer}`.

If you pass `--write_results`, Atlas will write its predicted answers to file.

Atlas will log the following evaluation metrics for open domain QA during its evaluation loop: 
* `eval_loss`: evaluation reader loss of evaluation answers.
* `exact_match`: Open-domain QA exact match score of generated answers
* `f1`: Open-domain QA F1 score of generated answers

#### Natural Questions & TriviaQA

You can download the NaturalQuestions and TriviaQA data by calling:

```bash
python preprocessing/prepare_qa.py --output_directory ${DATA_DIR} 
```

which will download `train.jsonl`, `train.64-shot.jsonl` (the fewshot training dataset we use), `dev.jsonl` and `test.jsonl` to `${DATA_DIR}/data/nq_data` and `${DATA_DIR}/data/triviaqa_data`.

Example scripts for running fewshot and standard finetuning and evaluation with a wikipedia index for NQ can be found in `examples/nq`. This script can be used for TriviaQA by swapping the train/dev/test files.

#### TempLama

We defined a cloze-question answering task for assessing index faithfulness and temporal transfer, derived from the TempLAMA dataset.

You can download the TempLAMA data and create and format our derived dataset by calling the following script:

```bash
python preprocessing/prepare_templama.py --output_directory ${DATA_DIR} 
```

which will create the files  `temp_lama.train.2017.jsonl`, `temp_lama.valid.2017.jsonl`, `temp_lama.test.2017.jsonl`, `temp_lama.train.2020.jsonl`, `temp_lama.valid.2020.jsonl`, `temp_lama.test.2020.jsonl` under `${DATA_DIR}/data/templama_data/`.
These files will contain cloze questions, with answers specific to that year. 

Example scripts for running training and evaluation for TempLama can be found at `examples/templama`. (note the use of `qa_prompt_format {question}`, which switches off the automatic QA prompt formatting used for TriviaQA and NQ)

</details>

<details>
<summary>
<h4 name="mcqa-task">Multiple Choice Question Answering (e.g. MMLU)</h4>
</summary>

Atlas can be trained to answer multiple choice questions by passing `--task multiple_choice` to `train.py` or `evaluate.py`.
We use this task for our experiments with MMLU.

Train/validation/test data for this task should consist of jsonl files, which should be passed to `train.py`  as `--train_data train_file_1.jsonl train_file_2.jsonl`, and `--eval_data eval_file_1.jsonl eval_file_2.jsonl` etc.
Files should have one JSON instance per line with the following format:
```python
{
  "question": "Which of the following is the body cavity that contains the pituitary gland?", 
  "options": {
    "A": "Abdominal",
    "B": "Cranial",
    "C": "Pleural", 
    "D": "Spinal"
    ... # you can have more (or fewer) answer options as long as they have alphabetically consecutive upper case letter keys, starting at A
  }, 
  "answer": "B",
  ... # you can have other fields you want to keep around for ease of analysis, but they wont actually be used
}
```
These will get automatically formatted into input queries for Atlas of the form `question: {question} answers: (A) {options['A']} (B) {options['B']} (C) {options['C']} (D) {options['D']} Answer: <extra_id_0>`, with target generations of the format `<extra_id_0> {answer letter}`.
The example above would get formatted to: `question: {Which of the following is the body cavity that contains the pituitary gland? answers: (A) Abdominal (B) Cranial (C) Pleural (D) Spinal Answer: <extra_id_0>`, with the target generation `{extra_id_0} B`.


Multiple-Choice QA has the following task specific args:
```
  --multiple_choice_num_options
      How many choice options for multiple choice QA (MMLU is 4) (default: 4)
  --multiple_choice_train_permutations {single,cyclic,all}
      Whether to train with answer order permutations When training on multiple choice (e.g. MMLU). Can improve results by de-biasing models's preferences for arbitrary answer orderings. Recommend
      training with 'all'. single: no permutations. cyclic: cyclic permutations. all: all possible answer order permutations' (default: single)
  --multiple_choice_eval_permutations {single,cyclic,all}
      Whether to evaluate with answer order permutations for multiple choice (e.g. MMLU). Can improve results by de-biasing models's preferences for arbitrary answer orderings. Best results with
      'all' but very slow. 'cyclic' is a good compromise. single: no permutations. cyclic: cyclic permutations. all: all possible answer order permutations' (default: single)
```

The permutation options will automatically duplicate the inputs, but with the answer orders permuted (e.g. With "A" now being "cranial", "B" being "pleural" etc.)
This improves results for when we have very small amounts of supervised data (or zeroshot). 
The code will automatically marginalize across results for evaluation permutations for you, in the case you use --multiple_choice_eval_permutations option `cyclic` or `all`.
More details on the permutation de-biasing can be found in the appendix of [Atlas: Few-shot Learning with Retrieval Augmented Language Models](https://arxiv.org/pdf/2208.03299.pdf).

If you pass `--write_results`, Atlas will write its predicted answers to file, with the following format:

```json
{
  "question": "the prompt-template applied input",
  "generation": "answer letter choice with highest probability after marginalizing across permutations",
  "choice_probs": "the probability of each answer choice (normalized over total answer options)",
  "all_probs": "the un-marginalized answer probabilities from all the answer order permutations",
  "permutations": ["the list of prediction objects for each permutation of the answer ordering"]
}
```

#### MMLU

A dedicated ReadMe is available for running MMLU experiments [here](./example_scripts/mmlu/README_MMLU.md). 
There is a tool to download and preprocess the MMLU data, and example scripts for running each of the experimental settings that we explore with MMLU are available `examples/mmlu`.
These are documented in detail in the MMLU Dedicated Readme.

</details>


<details>
<summary>
<h4 name="fever-task">FEVER Fact Verification</h4>
</summary>

Atlas can be trained to classify textual claims as "SUPPORTED", "REFUTED" or "NOT_ENOUGH_INFO" by a corpus, such as for the FEVER task  by using `--task fever` to `train.py` or `evaluate.py`.
	
You can download the FEVER data by calling the following script:

```bash
python preprocessing/prepare_fever.py --output_directory ${DATA_DIR} 
```

Train/validation/test data for this task should consist of jsonl files, which should be passed to `train.py`  as `--train_data train_file_1.jsonl train_file_2.jsonl`, and `--eval_data eval_file_1.jsonl eval_file_2.jsonl` etc.
Files should have one JSON instance per line with the following format:

```python
{
  "claim": "the claim to assess", 
  "label": "either 'SUPPORTS', 'REFUTES' or 'NOT ENOUGH INFO'",
   ... # you can have other fields you want to keep around for ease of analysis, but they wont actually be used
}
```
Atlas will automatically process these instances, and format them for input as `question: {claim} answer: <extra_id_0>` and the output as `<extra_id_0> {true, false or maybe}`.
If you pass `--write_results`, Atlas will write its predicted labels to file.
Atlas will log the following evaluation metrics for open domain QA during its evaluation loop: 

* `accuracy`:  how many claims were correctly classified by the model.

</details>

<details>
<summary>
<h4 name="kilt-task">KILT</h4>
</summary>

Atlas can be trained to perform KILT tasks by using `--task kilt` to `train.py` or `evaluate.py`.

KILT data can be obtained from [here](https://github.com/facebookresearch/KILT)

Train/validation/test data for this task should consist of jsonl files, which should be passed to `train.py`  as `--train_data train_file_1.jsonl train_file_2.jsonl`, and `--eval_data eval_file_1.jsonl eval_file_2.jsonl` etc.
Files should have one JSON instance per line with the following format (i.e. the codebase will accept the KILT format directly):
```python
{'id': # original data point id if available otherwise unique id
 'input': # question / claim / sentence / etc
 'output': [ # each element might contain an answer, a provenance or both
    {
    'answer': # answer in textual form
    'provenance': [
        # evidence set for the answer from the KILT ks
        {
            'wikipedia_id':  # *mandatory* 
            'title': 
            'section': 
            'start_paragraph_id': 
            'start_character': 
            'end_paragraph_id':
            'end_character': 
            'bleu_score': # wrt original evidence
            'meta': # dataset/task specific
        }
        ] 
      }
    ]
 'meta': # dataset/task specific
 }
```
Atlas will automatically process these instances appropriately, into Atlas] query inputs based on the `input` field and target generations based on the `answer` fields

If you pass `--write_results`, Atlas will write its predicted labels to file.

Atlas will log the following evaluation metrics for open domain QA during its evaluation loop: 
* `accuracy`:  how often generations exactly match the reference
* `exact_match`:  how often generations exactly match the reference, with open-domain QA normalization applied
* `f1`:  the token level f1 score overlap between the generation and reference

</details>



### Flat vs Faiss

There are two index modes implemented for Atlas. 
By default, we perform retrieval using an exact search ('Flat') index, where retrieval is performed on GPU using pure pytorch.
We also support a [FAISS](https://github.com/facebookresearch/faiss) mode, which is useful for saving GPU memory for extremely large indices, or where GPU memory is very restricted.
FAISS is a library for fast approximate nearest neighbor search. Our retrieval is on GPU, so we do not usually require further search acceleration, but faiss can be used for compressing the size of an index in memory, which may be of use for very large indices.

The mode to use is specified by `--index_mode {"flat"|"faiss"}`. 
For most use cases, the `flat` index will be sufficient and likely preferable. 

If using the faiss index, users should specify what kind of faiss index to use, using the following options:

```
  --faiss_index_type {ivfflat,flat,ivfsq,ivfpq,pq}
      IVFFlat, IndexFlatIP, IVFScalarQuantizer, IndexPQ or IndexIVFPQ with faiss-gpu (default: flat)
  --faiss_code_size FAISS_CODE_SIZE
      Parameter for PQ/SQ quantization (default: None)
```

A good default if using a faiss index is to use `--faiss_index_type ivfpq --faiss_code_size 16`. This will use an IVF-PQ index with the number of IVF clusters set to the square root of the number of embeddings per shard, and PQ code size of 16. More details on this index structure can be found in the faiss documentation [FAISS](https://github.com/facebookresearch/faiss).

### Index Saving and Loading

Indices (passage and embeddings shards) can be saved to disk and loaded in, to avoid recomputing them.
See [above](#prebuilt-indices) for some downloadable indices.

Index saving can be switched on using `--save_index_path {path/to/directory/save/index/in}`, which will create a directory,  
and save each worker's embedding shard to index (as a pytorch tensor on disk) and passages shard (as a pickle file).

To load an index, pass `--load_index_path {path}`, which will load the index at the specified path.

Saving and loading works with both `flat` and `faiss` modes.

In order to easily load an index when using a different number of workers from the index that created it, we can configure `--save_index_n_shards N`, which will save the index into N shards (for example if we have 32 workers, we can pass `--save_index_n_shards 128` to save the index as 128 shards to disk). 
When we try to load the index again, for example with 64 workers, the code will figure out it should load 2 saved files per worker. (Note: this functionality only works with `flat` indices - for faiss indices, you can only load indices where the number of workers is the same as when it was saved to disk).

### Strategies for dealing with stale indices

As the retriever is trained, the passage embeddings stored in memory become stale. 
This affects the accuracy of retrieval, and, over long periods of time, may lead to suboptimal training or instability.
Atlas has three methods that can combat this

1. <b name="#index-refresh">Index Refresh</b>: The simplest and most expensive option is to recompute the embeddings using the up-to-date retriever embedder. The index refresh rate schedule is controlled by the `--refresh_index` argument. format: `startstep-endstep:refreshrate,` e.g. `--refresh_index 0-1000:500,1000-10000:1000` will refresh the index every 500 steps for the first 1000 steps, and then every 1000 steps from step 1000 to 10000. You can also just pass in a single number e.g. `--refresh_index 100` will refresh the index every 100 steps. Pass `--refresh_index -1` to never refresh. We use this setting for large datasets and pretraining. 
2. <b name="#overretrieve-with-reranking">Over-Retrieve with Reranking</b>: Here, instead of refreshing the index, we can retrieve the top L passages (where L > K), and then, rerank these L passages using the up-to-date embedder on-the-fly, and pass the top K of these. This works well if the true top K are indeed contained in the stale top L. To use this pass `--retrieve_with_rerank` and specify `--n_to_rerank_with_retrieve_with_rerank L`. This method can be used in conjunction with index refreshing, to reduce staleness between refreshes.
3.  <b name="#query-Side-finetuning">Query-Side Finetuning</b>: To avoid stale-ness, we can keep the passage embedder of the retriever fixed, and only train the query embedder. This method will sacrifice retriever performance if there is lots of training data, but works well in few-shot settings. To enable this mode, pass `--query_side_retriever_training`. Note: usually we use parameter sharing for the passage and query encoder of the retriever - this mode is the exception, where we break the parameter tying to keep the passage encoder fixed.

### Retrieve-only mode

Atlas can be used purely in a retrieval mode at evaluation time. 
This can be useful for users who want a fast, scalable, easy to launch GPU-enabled dense retriever.

In this mode, (which only works with `evaluate.py`) no reader language model gets loaded, and the script will perform retrieval, and then write retrieval results to file if the `--write_results` flag has been passed.

To use this mode, pass `--retrieve_only` to `evaluate.py`.
There is an example of NaturalQuestions retrieval using this mode in `examples/nq/retrieve_only.sh`.

### Using pre-retrieved or cached passages

In some cases, users may have already performed retrieval and want to cache the retrieved results for their dataset, or know a priori the most relevant passages, and thus do not need to perform retrieval.

In these cases, Atlas can be forced to use user-specified passages per input instance, rather than retrieve, by 1) passing the `--use_file_passages` flag and 2) including a json field `passages` in the train/eval files they pass in, with the following format (e.g for the `qa` task)

<details>
<summary>
(click to expand to see example)
</summary>

```python
{
  "question": "where is the bermuda triangle",
  "answers": ["Western part of the North Atlantic Ocean"],
  "passages": [
    {
      "text": "text of first passage",
      "title": "title of  first passage",
      "id": "id of first passage"
      ... # other fields can be here but wont be used
    },
    {
      "text": "text of second passage",
      "title": "title of  second passage",
      "id": "id of second passage"
    },
    ... # more passages if you like
  ]
}
```

</details>

## Other features

The following are other features that Atlas provides for advanced users:

### Closed book mode

Atlas can be run as a standard non-retrieval-augmented T5 model, often referred to as "closed-book" in the literature. This is useful for running baseline experiments, and checking that your model does indeed benefit from retrieval-augmentation for your task. Pass the `--closed_book` argument to do closed-book training and ignore the retrieved passages.

### Specifying formats

Format strings can be injected for greater formatting control of how the inputs get presented to the Atlas model:

```
  --encoder_format ENCODER_FORMAT
    format string for reader's encoder preprocessing (default: "{query} title: {title} context: {text}")
  --retriever_format RETRIEVER_FORMAT
    format string for retriever's encoder preprocessing (default: "{title} {text}")
```

For example, passing `--encoder_format "{query} text: {text}"` wouldn't pass the retrieved passages' titles to the reader model.


### Implementing your own task

To implement a new task for Atlas, there are two options: the easiest is to preprocess or format your task to be compatible using one of the already implemented tasks (the `base` task should support almost all potential use cases).

The other is to implement your own task under `src/tasks/your_task_name.py` and import it under `src/tasks/__init__.py`.

See the `src/tasks/qa.py` for an example. 

The `process` function takes the raw parsed, jsonl-objects passed to --train_data or --eval_data, and should return a dict with `{query: "query to pass to Atlas", "target": "target string", "passages": [list of gold retrieved passages, can be empty]}`

The `evaluate` function takes a predicted generation and references for a task, and return a dict of task-specific evaluation scores, which the codebase will average across evaluation instances.


## Citing

To cite this work, please use the following bibtex:
```
@article{izacard_few-shot_2022,
	title = {Few-shot {Learning} with {Retrieval} {Augmented} {Language} {Models}},
	url = {http://arxiv.org/abs/2208.03299},
	publisher = {arXiv},
	author = {Izacard, Gautier and Lewis, Patrick and Lomeli, Maria and Hosseini, Lucas and Petroni, Fabio and Schick, Timo and Dwivedi-Yu, Jane and Joulin, Armand and Riedel, Sebastian and Grave, Edouard},
	year = {2022},
}
```

## License

### Code License:

The majority of the Atlas code is licensed under [CC-BY-NC](./LICENSE), however portions of the project are available under separate license terms: huggingface transformers is licensed under the [Apache 2.0 license](https://raw.githubusercontent.com/huggingface/transformers/main/LICENSE), which covers `src/modeling_bert.py` and `src/modeling_t5.py`.

### Data License:

The wikipedia-derived data used in the repository, such as the corpora and indices available from `download_corpus.py` and `download_index.py` are licensed according to [CC-BY-SA](https://creativecommons.org/licenses/by-sa/3.0/). 

