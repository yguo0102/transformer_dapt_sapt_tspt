import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Union
import csv
import pandas as pd

import torch
from filelock import FileLock
from torch.utils.data.dataset import Dataset

#from ...tokenization_roberta import RobertaTokenizer, RobertaTokenizerFast
#from ...tokenization_utils import PreTrainedTokenizer
#from ...tokenization_xlm_roberta import XLMRobertaTokenizer
#from ..processors.glue import glue_convert_examples_to_features, glue_output_modes, glue_processors
#from ..processors.utils import InputFeatures
from transformers import (
    DataProcessor,
    RobertaTokenizer, RobertaTokenizerFast,
    PreTrainedTokenizer,
    XLMRobertaTokenizer,
    glue_convert_examples_to_features,
    InputExample,
    InputFeatures
)



logger = logging.getLogger(__name__)


@dataclass
class CustomDataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    train_file: str = field(metadata={"help": "The name of training set"})
    dev_file: str = field(metadata={"help": "The name of dev set"})
    test_file: str = field(metadata={"help": "The name of testing set"})
    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    task_name: str = field(default='social_media')

    def __post_init__(self):
        self.task_name = self.task_name.lower()


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


class CustomProcessor(DataProcessor):

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    #@classmethod
    #def _read_csv(cls, input_file, quotechar=None):
    #    with open(input_file, "r", encoding="utf-8-sig") as f:
    #        return list(csv.reader(f, delimiter=",", quotechar=quotechar))

    @classmethod
    def _read_csv(cls, input_file):
        df = pd.read_csv(open(input_file, 'r'), lineterminator='\n')
        #df = pd.read_csv(input_file, lineterminator='\n')
        #try:
        #    df = pd.read_csv(input_file)
        #except:
        #    try:
        #        df = pd.read_csv(input_file, lineterminator='\n')
        #    except:
        #        raise
        return df

    def get_train_examples(self, data_dir, filename):
        """See base class."""
        #return self._create_examples(self._read_csv(os.path.join(data_dir, filename)), "train")
        return self._create_examples(os.path.join(data_dir, filename), "train")

    def get_dev_examples(self, data_dir, filename):
        """See base class."""
        #return self._create_examples(self._read_tsv(os.path.join(data_dir, filename)), "dev")
        return self._create_examples(os.path.join(data_dir, filename), "dev")

    def get_test_examples(self, data_dir, filename):
        """See base class."""
        #return self._create_examples(self._read_tsv(data_dir, filename), "test")
        return self._create_examples(os.path.join(data_dir, filename), "test")

    def get_labels(self, data_dir, filename):
        df = self._read_csv(os.path.join(data_dir, filename))
        labels = list(set([str(label) for label in df['label']]))
        labels.sort()
        logger.info("Load labels {}".format(labels))
        return labels

    def _create_examples(self, data_file, set_type):
        """Creates examples for the training, dev and test sets."""
        df = self._read_csv(data_file)
        texts = df.text.values.tolist() if 'text' in df.columns else df[df.columns[0]].values.tolist()
        labels = df.label.values.tolist() if 'label' in df.columns else [0 for _ in range(len(texts))]

        examples = []
        for (i, (text, label)) in enumerate(zip(texts, labels)):
            if pd.isna(label): # skip NaN
                continue

            text = str(text)
            label = str(label)
            guid = "%s-%s" % (set_type, i)
            text_a = text
            label = None if set_type == "test" else label
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples




class CustomDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    args: CustomDataTrainingArguments
    output_mode: str
    features: List[InputFeatures]

    def __init__(
        self,
        args: CustomDataTrainingArguments,
        ex_args: dict,
        tokenizer: PreTrainedTokenizer,
        limit_length: Optional[int] = None,
        mode: Union[str, Split] = Split.train,
    ):
        self.args = args
        self.processor = CustomProcessor()
        self.output_mode = "classification"
        if isinstance(mode, str):
            try:
                mode = Split[mode]
            except KeyError:
                raise KeyError("mode is not a valid split name")

        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            args.data_dir,
            "cached_{}_{}_{}_{}".format(
                mode.value, tokenizer.__class__.__name__, str(args.max_seq_length), 'social_media',
            ),
        )

        label_list = self.processor.get_labels(args.data_dir, ex_args['train_file'])
        self.label_list = label_list

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                start = time.time()
                self.features = torch.load(cached_features_file)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"Creating features from dataset file at {args.data_dir}")

                if mode == Split.dev:
                    examples = self.processor.get_dev_examples(args.data_dir, ex_args['dev_file'])
                elif mode == Split.test:
                    examples = self.processor.get_test_examples(args.data_dir, ex_args['test_file'])
                else:
                    examples = self.processor.get_train_examples(args.data_dir, ex_args['train_file'])
                if limit_length is not None:
                    examples = examples[:limit_length]
                self.features = glue_convert_examples_to_features(
                    examples,
                    tokenizer,
                    max_length=args.max_seq_length,
                    label_list=label_list,
                    output_mode=self.output_mode,
                )
                start = time.time()
                torch.save(self.features, cached_features_file)
                # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

    def get_labels(self):
        return self.label_list

