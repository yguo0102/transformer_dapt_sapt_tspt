# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""


import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    RobertaForSequenceClassification
)
from transformers.data.metrics import simple_accuracy, pearson_and_spearman
from sklearn.metrics import f1_score

from custom_dataset import CustomDataset
from custom_trainer import CustomTrainer

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


def conv_ex_args(remaining_args):
    ex_args = {}
    ex_args['train_file'] = remaining_args[1]
    ex_args['dev_file'] = remaining_args[3]
    ex_args['test_file'] = remaining_args[5]
    ex_args['metric'] = remaining_args[7]
    return ex_args

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, remaining_args = parser.parse_args_into_dataclasses(args=None, 
                return_remaining_strings=True, look_for_args_file=True)
        ex_args = conv_ex_args(remaining_args)

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    if not os.path.exists(training_args.output_dir):
        os.mkdir(training_args.output_dir)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Get datasets
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    train_dataset = CustomDataset(data_args, ex_args, tokenizer=tokenizer)
    eval_dataset =  CustomDataset(data_args, ex_args, tokenizer=tokenizer, mode="dev")
    test_dataset =  CustomDataset(data_args, ex_args, tokenizer=tokenizer, mode="test")

    try:
        num_labels = len(train_dataset.get_labels())
        output_mode = 'classification'
        logger.info("num labels: {}".format(num_labels))
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    def compute_metrics(p: EvalPrediction) -> Dict:
        preds = np.argmax(p.predictions, axis=1)
        labels = p.label_ids
        #print(preds)
        #print(p.label_ids)

        metrics = {
            'acc' : simple_accuracy(preds, labels),
            'pearson' : pearson_and_spearman(preds, labels)['pearson'],
            'f1_macro_weighted' : f1_score(y_true=labels, y_pred=preds, average='weighted'),
            'f1_macro' : f1_score(y_true=labels, y_pred=preds, average='macro'),
            'f1_micro' : f1_score(y_true=labels, y_pred=preds, average='micro'),
            'pos_class_f1' : f1_score(y_true=labels, y_pred=preds) if ex_args['metric'] == 'pos_class_f1' else 0,
        }
        return metrics

    # Initialize our Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        eval_metric = ex_args['metric'],
    )

    # Training
    if training_args.do_train:
        _, max_output_dir = trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )

        tokenizer.save_pretrained(max_output_dir)

        trainer.remove_checkpoints(keep_ckpt=max_output_dir)

        # re-load selected model
        model = AutoModelForSequenceClassification.from_pretrained(
            max_output_dir,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        eval_metric = ex_args['metric'],
    )

    # Evaluation
    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        eval_datasets = [eval_dataset]

        for eval_dataset in eval_datasets:
            eval_result = trainer.evaluate(eval_dataset=eval_dataset)

            output_eval_file = os.path.join(
                training_args.output_dir, f"eval_results_{eval_dataset.args.task_name}.txt"
            )
            if trainer.is_world_master():
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Eval results {} *****".format(eval_dataset.args.task_name))
                    for key, value in eval_result.items():
                        logger.info("%s\t%s\t%s", model_args.model_name_or_path, key, value)
                        writer.write("%s = %s\n" % (key, value))

            eval_results.update(eval_result)

    if training_args.do_predict:
        logging.info("*** Test ***")
        test_datasets = [test_dataset]

        for test_dataset in test_datasets:
            predictions = trainer.predict(test_dataset=test_dataset).predictions
            predicted_probs = predictions
            if output_mode == "classification":
                predictions = np.argmax(predictions, axis=1)

            output_test_file = os.path.join(
                training_args.output_dir, f"test_results_{test_dataset.args.task_name}.txt"
            )
            if trainer.is_world_master():
                with open(output_test_file, "w") as writer:
                    logger.info("***** Test results {} *****".format(test_dataset.args.task_name))
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if output_mode == "regression":
                            writer.write("%d\t%3.3f\n" % (index, item))
                        else:
                            item = test_dataset.get_labels()[item]
                            writer.write("%d\t%s\n" % (index, item))

            output_test_prob_file = os.path.join(
                training_args.output_dir, f"test_results_{test_dataset.args.task_name}_prob.txt"
            )
            if trainer.is_world_master():
                with open(output_test_prob_file, "w") as writer:
                    logger.info("***** Test prob results {} *****".format(test_dataset.args.task_name))
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predicted_probs):
                        if output_mode == "regression":
                            writer.write("%d\t%3.3f\n" % (index, item))
                        else:
                            fmt_values = ','.join([str(val) for val in item])
                            writer.write("%d\t%s\n" % (index, fmt_values))


    return eval_results



def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
