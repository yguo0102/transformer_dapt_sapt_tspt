# transformer_dapt_sapt_tapt
This repository is the source code for performing masked language model pretraining and fine-truning on transformer-based models with domain adaptive pretraining (DAPT), source adaptive pretraining (SAPT), and topic specific pretraining (TSPT).

## Installation
The code is implemented in Python 3.6.12. To install the Python environment, run the `pip` instruction below.
```sh
pip install -r requirements.txt
```

## Data prepration
For collecting topic-specific pretraining data, we implemented three filters `breast_cancer_filter.py`, `covid_filter.py`,
and `nmpu_filter.py` for three health-related topics `breast cancer`, `NPMU`, and `COVID-19`.
For each filter, the input is a text file where each line is a text sequence,
which would be written to the standard output if it satisfies the filter.
For example, a text file `a.txt` is as below:
```
hello world
covid 19
```
Run the instrction `python covid_filter.py < a.txt > b.txt`, and the output file `b.txt` is as below:
```
covid 19
```
See the folder `data_preprocess` for details.

We provided data samples in the folder `datasets` to test our code.
The full datasets can be obtained from their original creators/authors.


## MLM pretraining
To perform masked language model pretraining, run the script:
```sh
sh scripts/run_lm_training_apt_one_gpu.sh
```
The parameters in this script are the same as those mentioned in the paper.
It is worth noting that if the script works for single GPU.
If it is run in a machine with multiple GPUs, the parameter `--gradient_accumulation_steps` or `--per_device_train_batch_size` need to be changed accordingly, because the valid batch size is computed by `gradient_accumulation_steps*per_device_train_batch_size*n_npu`.


## Classification
To perform classification, run the script:
```sh
sh scripts/run_classification_batch.sh
```
The script implements the learning rate search and three random initialzations.
