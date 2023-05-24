# ASTormer: AST Structure-aware Transformer Decoder for Text-to-SQL

This is the project containing source code for the ACL2022 paper [*ASTormer: An AST Structure-aware Transformer Decoder for Text-to-SQL*](https://to-be-realized). If you find it useful, please cite our work.


## Create environment and download dependencies
The following commands are also provided in `setup.sh`.

1. Firstly, create conda environment `text2sql`:
    
        conda create -n text2sql python=3.7
        source activate text2sql
        pip install torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
        pip install -r requirements.txt

2. Next, download dependencies:

        python -c "import stanza; stanza.download('en')"
        python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt');"

3. Download pre-trained language models from [`Hugging Face Model Hub`](https://huggingface.co/models), such as `bert-large-whole-word-masking` and `electra-large-discriminator`, into the `pretrained_models` directory: (please ensure that `Git LFS` is installed)

        mkdir -p pretrained_models && cd pretrained_models
        git lfs install
        git clone https://huggingface.co/bert-large-uncased-whole-word-masking
        git clone https://huggingface.co/google/electra-large-discriminator

## Download and preprocess dataset

1. Create a new directory `data` to store all text-to-SQL data. Next, download, unzip and rename the [spider.zip](https://drive.google.com/uc?export=download&id=1_AckYkinAnhqmRQtGsQgUKAnTHxxX5J0), [sparc.zip](https://drive.google.com/uc?export=download&id=1Uu7NMHTR1tdQw1t7bAuM7OPU4LElVKfg) and [cosql_dataset.zip](https://drive.google.com/uc?export=download&id=1Y3ydpFiQQ3FC0bzdfy3groV95O_f1nXF), into the directory `data`. The testsuite database, for [Test Suite Accuracy](https://arxiv.org/abs/2010.02840), is also [downloaded](https://drive.google.com/file/d/1mkCx2GOFIqNesD4y8TDAO1yX1QZORP5w/view) and renamed into `data/spider/database-testsuite`. These default paths can be changed in `nsts/transition_system.py`. The directory `data` should be organized as follows:

        - data:
                - spider:
                        - database:
                                - ... all databases
                        - database-testsuite:
                                - ... all databases for Test Suite Accuracy
                        - train_spider.json
                        - train_others.json
                        - dev.json
                        - tables.json
                        - ... other files
                - sparc:
                        - database:
                                - ... all databases
                        - train.json
                        - dev.json
                        - tables.json
                        - ... other files
                - cosql:
                        - database:
                                - ... all databases
                        - sql_state_tracking:
                                - cosql_train.json
                                - cosql_dev.json
                        - tables.json
                        - ... other directories or files
                - dusql:
                        - db_content.json
                        - train.json
                        - dev.json
                        - test.json
                        - db_schema.json
                        - ... other files
                - chase:
                        - database:
                                - ... all .sqlite files
                        - chase_tables.json
                        - chase_train.json
                        - chase_dev.json
                        - chase_test.json

2. Merge `data/spider/train_spider.json` and `data/spider/train_others.json` into one single dataset `data/spider/train.json`, and preprocess all datasets:
  - we also fix some annotation errors in the following script
  - it takes roughly 10 minutes to preprocess each dataset (including training and validation set)

        ./run/run_preprocessing.sh


## Training

Train ASTormer with small/base/large series pre-trained language models respectively:
- `dataset_name` can be chosen from `['spider', 'sparc', 'cosql']`

        ./run/run_train_and_eval_small.sh [dataset_name]
        ./run/run_train_and_eval_base.sh [dataset_name]
        ./run/run_train_and_eval_large.sh [dataset_name]

## Evaluation and submission

For evaluation, see `run/run_eval.sh` (evaluation on the dev dataset) and `run/run_eval_from_scratch.sh` (evaluation from scratch, for submission) for reference.

## Acknowledgements

We would like to thank Tao Yu, Yusen Zhang and Bo Pang for running evaluations on our submitted models. We are also grateful to the flexible semantic parser [TranX](https://github.com/pcyin/tranX) that inspires our works.