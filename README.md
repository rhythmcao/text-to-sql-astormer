# ASTormer: AST Structure-aware Transformer Decoder for Text-to-SQL

This is the project containing source code for the paper [*ASTormer: An AST Structure-aware Transformer Decoder for Text-to-SQL*](https://to-be-realized). If you find it useful, please cite our work.

```bibtex
@inproceedings{Scholak2021:PICARD,
  author = {Torsten Scholak and Nathan Schucher and Dzmitry Bahdanau},
  title = "ASTormer: An AST Structure-aware Transformer Decoder for Text-to-SQL",
  booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
  month = nov,
  year = "2021",
  publisher = "Association for Computational Linguistics",
  url = "https://aclanthology.org/2021.emnlp-main.779",
  pages = "9895--9901",
}
```

> **Note that:** This work focuses on leveraging small-sized pre-trained models and abundant labeled training data to train a specialized, interpretable and efficient local text-to-SQL parser in low-resource scenarios, instead of chasing SOTA performances. For better performances, please try LLM with in-context learning (like [DINSQL](https://github.com/MohammadrezaPourreza/Few-shot-NL2SQL-with-prompting) and [ACTSQL](https://github.com/X-LANCE/text2sql-GPT)), or resort to larger encoder-decoder architectures containing billion parameters (like [Picard-3B](https://github.com/ServiceNow/picard) and [RESDSQL-3B](https://github.com/RUCKBReasoning/RESDSQL)). Due to a shift in the author's research focus in the LLM era, this project will no longer be maintained.


## Create environment
The following commands are also provided in `setup.sh`.

1. Firstly, create conda environment `astormer`:
    
        conda create -n astormer python=3.8.8
        conda activate astormer
        pip install torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
        pip install -r requirements.txt

2. Next, download dependencies:

        python -c "import stanza; stanza.download('en')"
        python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt');"

3. Download the desired pre-trained language models from [`Hugging Face Model Hub`](https://huggingface.co/models), such as `electra-small-discriminator`, into the `pretrained_models` directory: (please ensure that `Git LFS` is installed)

        mkdir -p pretrained_models && cd pretrained_models
        git lfs install
        git clone https://huggingface.co/google/electra-small-discriminator


## Download and preprocess dataset

1. Create a new directory `data` to store all text-to-SQL data. Next, download, unzip and rename the [spider.zip](https://drive.google.com/uc?export=download&id=1_AckYkinAnhqmRQtGsQgUKAnTHxxX5J0), [sparc.zip](https://drive.google.com/uc?export=download&id=1Uu7NMHTR1tdQw1t7bAuM7OPU4LElVKfg) and [cosql_dataset.zip](https://drive.google.com/uc?export=download&id=1Y3ydpFiQQ3FC0bzdfy3groV95O_f1nXF), into the directory `data`. The testsuite database, for [Test Suite Accuracy](https://arxiv.org/abs/2010.02840), is also [downloaded](https://drive.google.com/file/d/1mkCx2GOFIqNesD4y8TDAO1yX1QZORP5w/view) and renamed as `data/spider/database-testsuite`. Chinese benchmarks such as [DuSQL](https://dataset-bj.cdn.bcebos.com/qianyan/DuSQL.zip) and [Chase](https://github.com/xjtu-intsoft/chase/tree/page/data) are also integrated in this project. These default paths can be changed by modifying the dictionary `CONFIG_PATHS` in `nsts/transition_system.py`. For variants of dev dataset on Spider, e.g., [SpiderSyn](https://github.com/ygan/Spider-Syn/tree/main/Spider-Syn), [SpiderDK](https://github.com/ygan/Spider-DK), [SpiderRealistic](https://zenodo.org/records/5205322#.YTts_o5Kgab), they are also included at the evaluation stage. The directory `data` should be organized as follows:

        - data:
                - spider:
                        - database:
                                - ... # all databases
                        - database-testsuite:
                                - ... # all databases for Test Suite Accuracy
                        - train_spider.json
                        - train_others.json
                        - dev.json
                        - tables.json
                        - ... other files such as dev_syn.json, dev_dk.json, dev_realistic.json
                - sparc:
                        - database:
                                - ... # all databases
                        - train.json
                        - dev.json
                        - tables.json
                        - ... # other files
                - cosql:
                        - database:
                                - ... all databases
                        - sql_state_tracking:
                                - cosql_train.json
                                - cosql_dev.json
                        - tables.json
                        - ... #other directories or files
                - dusql:
                        - db_content.json
                        - train.json
                        - dev.json
                        - test.json
                        - db_schema.json
                        - ... # other files
                - chase:
                        - database:
                                - ... # all .sqlite files
                        - chase_tables.json
                        - chase_train.json
                        - chase_dev.json
                        - chase_test.json

2. Datasets preprocessing, including:
  - Merge `data/spider/train_spider.json` and `data/spider/train_others.json` into one single dataset `data/spider/train.json`
  - Transform the dataset, tables and database format of Chinese benchmarks DuSQL and Chase
  - Fix some annotation errors of SQLs and type errors in database schema
  - Re-parse the SQL query into a unified JSON format for all benchmarks. We modify and unify the format of `sql` field, including: (see `nsts/parse_sql_to_json.py` for details)
      - For a single condition, the parsed tuple is changed from `(not_op, op_id, val_unit, val1, val2)` into `(agg_id, op_id, val_unit, val1, val2)`. The `not_op` is removed and integrated into `op_id`, such as `not in` and `not like`
      - For FROM conditions where the value is a column id, the target `val1` must be a column list `(agg_id, col_id, isDistinct(bool))` to distinguish from int values
      - For ORDER BY clause, the parsed tuple is changed from `('asc'/'desc', [val_unit1, val_unit2, ...])` to `('asc'/'desc', [(agg_id, val_unit1), （agg_id, val_unit2), ...])`
  - It takes less than 10 minutes to preprocess each dataset (tokenization, schema linking and value linking)

        ./run/run_preprocessing.sh


## Training

To train ASTormer with `small`/`base`/`large` series pre-trained language models respectively:
- `dataset_name` can be chosen from `['spider', 'sparc', 'cosql', 'dusql', 'chase']`
- `plm` is the name of pre-trained language models under the directory `pretrained_models`. Please conform to the choice in preprocessing script (`run/run_preprocessing.sh`)

        # swv means utilizing static word embeddings, extracted from small-series models such as electra-small-discriminator
        ./run/run_train_and_eval_swv.sh [dataset_name] [plm]
        
        # DDP is not needed, a single 2080Ti GPU is enough
        ./run/run_train_and_eval_small.sh [dataset_name] [plm]
        
        # if DDP used, please specify the three environment variables
        GPU_PER_NODE=2 NUM_NODES=1 NODE_RANK=0 ./run/run_train_and_eval_base.sh [dataset_name] [plm]
        
        # if DDP used, please specify the three environment variables
        GPU_PER_NODE=4 NUM_NODES=1 NODE_RANK=0 ./run/run_train_and_eval_large.sh [dataset_name] [plm]


## Inference and Submission

For inference, see `run/run_eval.sh` (evaluation on the dev dataset) and `run/run_eval_from_scratch.sh` (evaluation from scratch, for testset submission) for reference.

        ./run/run_eval.sh [saved_model_dir]
        ./run/run_eval_from_scratch.sh [saved_model_dir]

For both training and inference, you can also use the prepared Docker environment from [rhythmcao/astormer:v0.3](https://hub.docker.com/layers/rhythmcao/astormer/v0.3/images/sha256-fcc35a6d4422d7283f23427301b51f7236aa55054c5a85a60c35cca7b1b276a3?context=repo):

        docker pull rhythmcao/astormer:v0.3
        docker run -it -v $PWD:/workspace rhythmcao/astormer:v0.3 /bin/bash

## Acknowledgements

We are grateful to the flexible semantic parser [TranX](https://github.com/pcyin/tranX) that inspires our works.