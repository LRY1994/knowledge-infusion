
"""
EDIT NOTICE

File edited from original in https://github.com/castorini/hedwig
by Bernal Jimenez Gutierrez (jimenezgutierrez.1@osu.edu)
in May 2020
"""

import csv
from os import truncate
import sys
import json
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

    def __init__(self, input_ids, attention_mask, decoder_input_ids,decoder_attention_mask):
        self.input_ids=input_ids
        self.attention_mask=attention_mask
        self.decoder_input_ids=decoder_input_ids
        self.decoder_attention_mask=decoder_attention_mask
        



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
    print ("Start tokenizing...")

    questions = [d.input_text.replace('\\n','')  for d in examples]
  
    answers = [d.target_text.replace('\\n','') for d in examples]

    # print(len(answers))
    # answers, metadata = self.flatten(answers)
    # if self.args.do_lowercase:
    #     questions = [question.lower() for question in questions]
    #     answers = [answer.lower() for answer in answers]
    # if self.args.append_another_bos:
    #     questions = ["<s> "+question for question in questions]
    #     answers = ["<s> " +answer for answer in answers]
    question_input = tokenizer.batch_encode_plus(questions,
                                                padding='max_length',
                                                max_length=32,
                                                truncation=True
                                                )
    answer_input = tokenizer.batch_encode_plus(answers,                                           
                                                padding=True,  
                                                max_length=36,                                          
                                                truncation=True
                                                )

    input_ids, attention_mask = question_input["input_ids"], question_input["attention_mask"]
    decoder_input_ids, decoder_attention_mask = answer_input["input_ids"], answer_input["attention_mask"]

    preprocessed_data = [input_ids, attention_mask,
                                     decoder_input_ids, decoder_attention_mask,
                                     ]
    with open('train-barttokenized.json', "w") as f:
        json.dump([input_ids, attention_mask,
                    decoder_input_ids, decoder_attention_mask
                    ], f)


    return{
        'input_ids':input_ids,
        'attention_mask':attention_mask,
        'decoder_input_ids':decoder_input_ids,
        'decoder_attention_mask':decoder_attention_mask
    }




