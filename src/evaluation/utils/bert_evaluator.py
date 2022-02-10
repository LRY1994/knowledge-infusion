import warnings
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn import metrics
from tabulate import tabulate
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm

from .abstract_processor import convert_examples_to_features

# Suppress warnings from sklearn.metrics
warnings.filterwarnings("ignore")


def print_dict_as_table(dic, tag=None, columns=["keys", "values"]):
    """Print a dictionary as table.
    Args:
        dic (dict): dict object to be formatted.
        tag (str): A name for this dictionary.
        columns ([str,str]):  default ["keys", "values"]. columns name for keys and values.
    Returns:
        None
    """
    print("-" * 80)
    if tag is not None:
        print(tag)
    df = pd.DataFrame(dic.items(), columns=columns)
    print(tabulate(df, headers=columns, tablefmt="psql"))
    print("-" * 80)
    return tabulate(df, headers=columns, tablefmt="psql")




class BertEvaluator(object):
    def __init__(
        self, model, processor, tokenizer, args, split="dev", dump_predictions=False
    ):
        self.args = args
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.split = split
        self.dump_predictions = dump_predictions

        if split == "train":
            self.eval_examples = self.processor.get_train_examples(
                args.data_dir, args.train_file
            )
            if args.train_ratio < 1:
                keep_num = int(len(self.eval_examples) * args.train_ratio) + 1
                self.eval_examples = self.eval_examples[:keep_num]
                print(f"Reduce Training example number to {keep_num}")
        elif split == "dev":
            self.eval_examples = self.processor.get_dev_examples(
                args.data_dir, args.dev_file
            )
        elif split == "test":
            self.eval_examples = self.processor.get_test_examples(
                args.data_dir, args.test_file
            )
        self.examples_ids = [example.guid for example in self.eval_examples]

    def _get_inputs_dict(self, batch):
        device = self.device
        pad_token_id = self.tokenizer.pad_token_id
        source_ids, source_mask, y = batch["source_ids"], batch["source_mask"], batch["target_ids"]
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone()
        lm_labels[y[:, 1:] == pad_token_id] = -100

        inputs = {
            "input_ids": source_ids.to(device),
            "attention_mask": source_mask.to(device),
            "decoder_input_ids": y_ids.to(device),
            "labels": lm_labels.to(device),
        }
        return inputs

    def get_scores(self, silent=False):
        """
        Evaluates the model on eval_dataset.

        Utility function to be used by the eval_model() method. Not intended to be used directly.
        """
      
        eval_features = convert_examples_to_features(
            self.eval_examples, self.args.max_seq_length, self.tokenizer
        )

        source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
        source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)       
        target_ids = torch.tensor([f.target_ids for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(
            source_ids, source_mask, target_ids
        )

        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(
            eval_data, sampler=eval_sampler, batch_size=self.args.batch_size
        )

        self.model.eval()

        eval_loss = 0
        nb_eval_steps = 0
        results = {}
       
        for batch in tqdm( eval_dataloader, desc="Evaluating", disable=silent): 

            inputs = self._get_inputs_dict(batch)
          
            with torch.no_grad():
                outputs = self.model(**inputs)
                loss = outputs[0]
                eval_loss += loss.mean().item()
                
            loss = outputs[0]

            if self.args.n_gpu > 1:
                loss = loss.mean()
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps
            eval_loss += loss.item()         
            nb_eval_steps += 1

        
        eval_loss = eval_loss / nb_eval_steps
        results["eval_loss"] = eval_loss

        output_eval_file = os.path.join(self.args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))

        return results
            
        
   