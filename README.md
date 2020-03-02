# **BERT (Base) Sequence Tagging on OpenKP** (Pytorch)

This repository provides the code of the model named **BERT (Base) Sequence Tagging** , which outperforms the Baselines (MSMARCO Team) on the [**OpenKP Leaderboard**](https://microsoft.github.io/msmarco/#kp).

OpenKP (OpenKeyPhrase) is a large scale, open-domain keyphrase extraction dataset, which was first released in the paper [Open Domain Web Keyphrase Extraction Beyond Language Modeling](https://www.emnlp-ijcnlp2019.org/program/) at EMNLP-IJCNLP 2019. Now it is a part of the [MSMARCO](http://www.msmarco.org/) dataset family .



#### Official Evaluation Results (ranked by F1 @3 on Eval)

| Rank | Model                                                        | Submission Date    | Precision @1,@3,@5  |   Recall @1,@3,@5   |     F1 @1,**@3**,@5     |
| :--- | :----------------------------------------------------------- | :----------------- | :-----------------: | :-----------------: | :---------------------: |
| 1    | **BERT (Base) Sequence Tagging** Si Sun (Tsinghua University), Chenyan Xiong (MSR AI), Zhiyuan Liu (Tsinghua University) | November 5th, 2019 | 0.484, 0.312, 0.227 | 0.255, 0.469, 0.563 | 0.321, **0.361**, 0.314 |
| 2    | **Baseline finetuned on Bing Queries** MSMARCO Team          | October 19th, 2019 | 0.397, 0.249, 0.149 | 0.215, 0.391, 0.391 | 0.267, **0.292**, 0.209 |
| 3    | **Baseline** MSMARCO Team                                    | October 19th, 2019 | 0.365, 0.237, 0.142 | 0.196, 0.367, 0.367 | 0.244, **0.277**, 0.198 |

## Attention
```
- Our model is trained on 2 Tesla T4 GPUs (2 * 16G), so we can set per_gpu_train_batch_size = 12.
- If you change the batch_size on your GPUs, we suggested that you adjust the gradient_accumulation_steps, max_train_epochs, and warmup_proportion parameters to better train your own model.
- We also provide the training loss for our best checkpoint : **Epoch Mean Loss=0.0471 (epoch=4)**
```

## Quickstart

```
python 3.5
Pytorch 1.3.0
Tensorflow (tested on 1.14.0, only for tensorboardX)
```


You should first download the [**DATA**](https://drive.google.com/file/d/1aqPl8eUHKR1yTr4CX9lpzzmogp1mc3I3/view?usp=sharing) folder, which includes preprocess data, checkpoint and extracted keyphrase candidates for our result on the leardbord. Default directory structure should be setted as following :

```
Bert2Tag
  |— DATA
    |— cached_features (saved preprocess data)
    |  |— openkp.train.json (OpenKP train dataset)
    |  |— openkp.valid.json (OpenKP Dev dataset)
    |  |— openkp.eval_public.json (OpenKP Valid dataset)
    |  |— Dev_reference.json (OpenKP Dev ground-truth keyphrases for test)
    |
    |— pretrain_model
    |  |— bert-base-cased
    |  |  |— vocab.txt
    |  |  |— config.json
    |  |  |— pytorch_model.bin
    |  |
    |  |— output (our best checkpoint)
    |     |— epoch_4.checkpoint
    |
    |— Pred (extracted keyphrase candidates)
       |— Dev_candidate.json
       |— EvalPublic_candidate.json (submitted to the leardbord)
```

P.S. `bert-base-cased` can also be download from [Huggingface's Pytorch-Transformers](https://github.com/huggingface/pytorch-transformers)



#### * Re-produce evaluation result on Dev using our checkpoint

- **Get evaluation result using our generated keyphrase candidates**

  ```
  python evaluate.py ./DATA/Pred/Dev_candidate.json ./DATA/cached_features/Dev_reference.json
  ```

  the`evaluate.py ` script is cloned from [official evaluation script](https://github.com/microsoft/OpenKP/blob/master/evaluate.py)  (we can only evaluate dev candidates because we don't know the ground-truths of Eval) , then the evaluation result on Dev can be shown as below :

  ```
  ########################
  Metrics

  @1
  F1:0.35075642965204235
  P:0.47745839636913767
  R:0.2980584972264246

  @3
  F1:0.36837115481593546
  P:0.3011094301563288
  R:0.5318204740292486

  @5
  F1:0.31765903032922704
  P:0.22160363086232984
  R:0.6389056984367121
  #########################
  ```

- **Generate keyphrase candidates using our checkpoint model**

  ```
  source test.sh
  ```

  The new generated keyphrases for Dev and Eval can be found in `./Pred`  folder .



#### * Re-train a new model from scratch

- **Train a new model using our preprocess data**

  ```
  source train.sh
  ```

  Several new files will be generated：

  - `logging.txt` and `viso` folder (if use tensorboardX) : track the train & valid losses saved in `./Log` ;

  - `epoch_i.checkpoint`  saved in `./output` after each i epoch.



#### * Re-preprocess source OpenKP datasets

- Download the OpenKP dataset from [MS MARCO](http://www.msmarco.org/dataset.aspx) website to your own directory

- Preprocess the dataset using our `preprocess.py` script (it might take 2~3 hours)

  ```
  python preprocess.py --source_dataset_dir "your own directory" --output_path "your save directory"
  ```



## Overview

#### * Data Characteristics

The documents of the dataset come from real world webs , with a diversified topic domain distribution differs from previous keyphrase extraction datasets (focuse on a single sciencific-field).

For each document, 1-3 most relevant keyphrase labels have been generated by expert annotators, **they have to appear in the document**.



#### * Model Architectures

Upon the Characteristics of the data, we formulated keyphrase extraction as a soft-select sequence tagging task, and first introduced BERT into open-domian keyprhase extraction (as we know). We describe our model's workflow as the following 3 stages :

- **Word-Level Representations :**   We encode an input document into a sequence of WordPiece tokens' vectors with a pretrained [BERT](https://www.aclweb.org/anthology/N19-1423.pdf) (base), and then we pick up the first sub-token vector of each word to represent the input in word-level.

- **Phrase-Level Representations :** We perform a **soft-select** method to decode phrase from word-level vector instead of hard-select used in the standard sequence tagging task .

  The word-level representation is feed into an classification layer to obtain the tag probabilities of each word on 5 classes  (O, B, I, E, U) , and then we employ different tag patterns for extracting different n-grams ( 1 ≤ n ≤ 5 ) over the whole sequence.

  Last there are a collect of n-gram candidates, each word of the n-gram just has one score.

  **Soft-select Example :** considering all 3-grams (B I E) on the L-length document, we can extract (L-3+1)  3-grams sequentially like sliding window. In each 3-gram, we only keep **B score** for the first word, **I score** for the middle word, and **E score** for the last word, etc.

  > **O** : Non Keyphrase ;  **B** : Begin word of the keyprase ;  **I** : Middle word of the keyphrase ;  **E** : End word of keyprhase ;  **U** : Uni-word keyphrase

- **Document-Level Keyphrase :** At the Last stage, the recovering from phrase-level n-grams to document-level keyphrases can be naturally formulated as a ranking task.

  Incorporating with term frequency, we employ **Min Pooling** to get the final score of each n-gram (we tested Min / Mean / LogMean Pooling , Min pooling is the best) . Based on the final scores, we extract 5 top ranked keyprhase candidates for each document.



## Code Reference

[1] https://github.com/huggingface/transformers

[2] https://github.com/kamalkraj/BERT-NER/tree/experiment

[3]  https://github.com/facebookresearch/DrQA

[4] https://github.com/microsoft/OpenKP



## Contact

For any question, please contact **Si Sun** by email s-sun17@mails.tsinghua.edu.cn , we will try our best to solve.
