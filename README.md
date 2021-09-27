# transformer_dapt_sapt_tapt

## Python environment
```sh
# Python 3.6.12
pip install -r requirements.txt
```

## Data preprocessing
For collecting topic-specific pretraining data, we implemented three filters for `breast cancer`,
`NPMU`, and `COVID-19`. See the folder `data_preprocess` for details.

## MLM pretraining
```sh
sh scripts/run_lm_training_apt_one_gpu.sh
```

## Classification
```sh
sh scripts/run_classification_batch.sh
```
