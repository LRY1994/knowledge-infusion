import pandas as pd

from .abstract_processor import BertProcessor, InputExample

def read_data_source_target(file_name_source, file_name_target):
    file_source = open(file_name_source, 'r', encoding='utf8')
    file_target = open(file_name_target, 'r', encoding='utf8')

    source = file_source.readlines()
    target = file_target.readlines()

    if len(source) != len(target):
        raise ValueError(
            "The length of the source file should be equal to target file"
        )
    source_target_pair = [[ source[i], target[i]] for i in range(len(source))] # "" for "prefix" used in t5_util.py
    data_df = pd.DataFrame(source_target_pair, columns=[ "input_text", "target_text"])
    return data_df

def load_bioasq(data_dir):
    train_df = read_data_source_target(data_dir + "train.source", data_dir + "train.target")   
    dev_df = read_data_source_target(data_dir + "dev.source", data_dir + "dev.target")
    test_df =  read_data_source_target(data_dir + "test.source", data_dir+ "test.target")
    return train_df, dev_df, test_df


class BioAsqProcessor(BertProcessor):
    NAME = "webquestion"
    
    def __init__(self, data_dir):
        self.train_df, self.dev_df, self.test_df = load_bioasq(data_dir)

    def get_train_examples(self):
        return self._create_examples(self.train_df, set_type="train")

    def get_dev_examples(self):
        return self._create_examples(self.dev_df, set_type="dev")

    def get_test_examples(self):
        return self._create_examples(self.test_df, set_type="test")

    def _create_examples(self, data_df, set_type):
        examples = []
        
        for (i, row) in data_df.iterrows():           
            input_text = row["input_text"]
            target_text = row["target_text"]
            
            examples.append(
                InputExample(input_text=input_text, target_text=target_text)
            )
        print(
            f"Get {len(examples)} examples of {self.NAME} datasets for {set_type} set"
        )
        return examples
