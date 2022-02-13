import datetime
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, TensorDataset, Dataset
from tqdm import tqdm
import pandas as pd

from .abstract_processor import convert_examples_to_features
from .bert_evaluator import BertEvaluator

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, idx):
        return self.data[idx]
    def __len__(self):
        return(len(self.data))

class BertTrainer(object):
    def __init__(self, model, optimizer, processor, scheduler, tokenizer, args):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.processor = processor
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.device = 'cuda'
        self.train_examples = self.processor.get_train_examples()
        if args.train_ratio < 1:
            keep_num = int(len(self.train_examples) * args.train_ratio) + 1
            self.train_examples = self.train_examples[:keep_num]
            print(f"Reduce Training example number to {keep_num}")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_str = self.args.model
        if "/" in model_str:
            model_str = model_str.split("/")[1]#bart-base

        self.num_train_optimization_steps = (
            int(
                len(self.train_examples)
                / args.batch_size
                / args.gradient_accumulation_steps
            )
            * args.epochs
        )

        self.log_header = (
            "Epoch Iteration Progress   Dev/Acc.  Dev/Pr.  Dev/Re.   Dev/F1   Dev/Loss"
        )
        self.log_template = " ".join(
            "{:>5.0f},{:>9.0f},{:>6.0f}/{:<5.0f} {:>6.4f},{:>8.4f},{:8.4f},{:8.4f},{:10.4f}".split(
                ","
            )
        )

        self.iterations, self.nb_tr_steps, self.tr_loss = 0, 0, 0
        self.best_dev_acc, self.unimproved_iters = 0, 0
        self.early_stop = False

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
 
    def train_epoch(self, train_dataloader,epoch_number):
        self.tr_loss = 0

        batch_iterator = tqdm(
            train_dataloader,
            desc=f"Running Epoch {epoch_number} of {self.args.epochs} ",
            mininterval=0,
        )

        for step, batch in enumerate(batch_iterator):
            self.model.train()
           
            inputs = self._get_inputs_dict(batch)
            outputs = self.model(**inputs)
            loss = outputs[0]
            current_loss = loss.item()
            
            batch_iterator.set_description(
                f"Epochs {epoch_number}/{self.args.epochs}. Running Loss: {current_loss:9.4f}"
            )


            if self.args.n_gpu > 1:
                loss = loss.mean()
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps

            loss.backward()

            self.tr_loss += loss.item()
            self.nb_tr_steps += 1
            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.iterations += 1

        print(self.tr_loss)

    def train(self):
        train_features = convert_examples_to_features(
            self.train_examples, self.args.max_seq_length, self.tokenizer
        )
      
        print("Number of examples: ", len(self.train_examples))
        print("Batch size:", self.args.batch_size)
        print("Num of steps:", self.num_train_optimization_steps)

        source_ids = torch.cat([f.source_ids for f in train_features],dim=0)
        # source_ids = [f.source_ids for f in train_features]

        source_mask = torch.cat([f.source_mask for f in train_features], dim=0)       
        target_ids = torch.cat([f.target_ids for f in train_features], dim=0)

        # train_data = TensorDataset(
        #     source_ids, source_mask, target_ids
        # )
        train_df=pd.DataFrame({'source_ids':source_ids, 'source_mask':source_mask, 'target_ids':target_ids})
        train_data=MyDataset(train_df)

        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(
            train_data, 
            sampler=train_sampler, 
            batch_size=self.args.batch_size
        )

        print("Start Training")
      
        for epoch in tqdm(range(self.args.epochs), file=sys.stdout, desc="Epoch"):
            self.train_epoch(train_dataloader,self.args.epochs)
            dev_evaluator = BertEvaluator(
                self.model, self.processor, self.tokenizer, self.args, split="dev"
            )
            result = dev_evaluator.get_scores()[0]

            # Update validation results
            if result["correct_ratio"] > self.best_dev_acc:
                self.unimproved_iters = 0
                self.best_dev_acc = result["correct_ratio"]
                torch.save(self.model, self.args.best_model_dir + "model.bin")
            else:
                self.unimproved_iters += 1
                if self.unimproved_iters >= self.args.patience:
                    self.early_stop = True
                    tqdm.write(
                        "Early Stopping. Epoch: {}, Best Dev performance: {}".format(
                            epoch, self.best_dev_acc
                        )
                    )
                    break
    
