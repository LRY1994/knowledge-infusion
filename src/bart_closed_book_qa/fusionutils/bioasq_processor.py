import pandas as pd

from .abstract_processor import BertProcessor, InputExample
from data_reader.data_reader import read_data_source_target

def load_bioasq(args):
    train_df = read_data_source_target(args.data_dir + "train.source", args.data_dir + "train.target")
    
    dev_df = read_data_source_target(args.data_dir + "valid.source", args.data_dir + "valid.target")

    if args.predict_during_training == True:
        if args.predict_on_valid == True:
            test_df = read_data_source_target(args.data_dir + "valid.source", args.data_dir + "valid.target")
        else:
            test_df = read_data_source_target(args.data_dir + "test.source", args.data_dir + "test.target")
    else:
        test_df = None

    test_df =  read_data_source_target(args.data_dir + "test.source", args.data_dir+ "test.target")
    return train_df, dev_df, test_df


class BioAsqProcessor(BertProcessor):
    NAME = "BioAsq"
    

    def __init__(self, args):
        self.train_df, self.dev_df, self.test_df = load_bioasq(args)

    def get_train_examples(self):
        return self.train_df

    def get_dev_examples(self):
        return self.dev_df

    def get_test_examples(self):
        return self.test_df

    