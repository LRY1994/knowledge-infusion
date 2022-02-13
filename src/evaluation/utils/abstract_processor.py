
"""
EDIT NOTICE

File edited from original in https://github.com/castorini/hedwig
by Bernal Jimenez Gutierrez (jimenezgutierrez.1@osu.edu)
in May 2020
"""

import csv
import sys

from nltk.tokenize import sent_tokenize

from transformers import BartTokenizer, BartForConditionalGeneration
class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, input_text, target_text=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        
        self.input_text = input_text
        self.target_text = target_text
       


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, source_ids, source_mask, target_ids):
        self.source_ids=source_ids
        self.source_mask=source_mask
        self.target_ids=target_ids



class BertProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """
        Gets a collection of `InputExample`s for the train set
        :param data_dir:
        :return:
        """
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """
        Gets a collection of `InputExample`s for the dev set
        :param data_dir:
        :return:
        """
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """
        Gets a collection of `InputExample`s for the test set
        :param data_dir:
        :return:
        """
        raise NotImplementedError()

    def get_labels(self):
        """
        Gets a list of possible labels in the dataset
        :return:
        """
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """
        Reads a Tab Separated Values (TSV) file
        :param input_file:
        :param quotechar:
        :return:
        """

        csv.field_size_limit(sys.maxsize)

        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(str(cell, "utf-8") for cell in line)
                lines.append(line)
            return lines

def preprocess_data_bart(data,tokenizer,max_seq_length):
    input_text = data.input_text
    target_text = data.target_text

    input_ids = tokenizer.batch_encode_plus(
        [input_text], max_length=max_seq_length, padding='max_length', return_tensors="pt", truncation=True
    )

    target_ids = tokenizer.batch_encode_plus(
        [target_text], max_length=max_seq_length, padding='max_length', return_tensors="pt", truncation=True
    )

    return {
        "source_ids": input_ids["input_ids"].squeeze(),
        "source_mask": input_ids["attention_mask"].squeeze(),
        "target_ids": target_ids["input_ids"].squeeze(),
    }
    
def convert_examples_to_features(
    examples, max_seq_length, tokenizer, print_examples=False
):
    """
    Loads a data file into a list of InputBatch objects
    :param examples:
    :param max_seq_length:
    :param tokenizer:
    :param print_examples:
    :return: a list of InputBatch objects
    """

    features = []
    
    for (ex_index, example) in enumerate(examples):
        tmp = preprocess_data_bart(example,tokenizer,max_seq_length)
        features.append(
            InputFeatures(
                source_ids=tmp['source_ids'],
                source_mask=tmp['source_mask'],
                target_ids=tmp['target_ids'],
            )
        )
    return features


