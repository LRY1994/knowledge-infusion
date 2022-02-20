from unittest import result
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
from torch import Tensor, nn
from multiprocessing import Pool, cpu_count
from .abstract_processor import convert_examples_to_features
import logging
# Suppress warnings from sklearn.metrics
warnings.filterwarnings("ignore")

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
            self.eval_examples = self.processor.get_dev_examples()
        elif split == "test":
            self.eval_examples = self.processor.get_test_examples()
        self.examples_ids = [example.guid for example in self.eval_examples]

    def _get_inputs_dict(self, batch):
        device = self.device
        # pad_token_id = self.tokenizer.pad_token_id
        # source_ids, source_mask, y = batch["source_ids"], batch["source_mask"], batch["target_ids"]
        # y_ids = y[:, :-1].contiguous()
        # lm_labels = y[:, 1:].clone()
        # lm_labels[y[:, 1:] == pad_token_id] = -100

        # inputs = {
        #     "input_ids": source_ids.to(device),
        #     "attention_mask": source_mask.to(device),
        #     "decoder_input_ids": y_ids.to(device),
        #     "labels": lm_labels.to(device),
        # }
        inputs = {
            # "input_ids": batch[0].to(device),
            # "decoder_input_ids": lm_labels.to(device),
            # "labels": lm_labels_masked.to(device),
            'input_ids':batch[0].to(device), 
            'attention_mask':batch[1].to(device),
            'decoder_input_ids':batch[2].to(device),
            'decoder_attention_mask':batch[3].to(device),
        }
        return inputs

    def get_scores(self, silent=False,verbose=True,**kwargs):
        """
        Evaluates the model on eval_dataset.

        Utility function to be used by the eval_model() method. Not intended to be used directly.
        """
      
        eval_features = convert_examples_to_features(
            self.eval_examples, self.args.max_seq_length, self.tokenizer
        )

        input_ids = [f.input_ids for f in eval_features]
        attention_mask = [f.attention_mask for f in eval_features]      
        decoder_input_ids = [f.decoder_input_ids for f in eval_features]
        decoder_attention_mask = [f.decoder_attention_mask for f in eval_features]

        padded_input_ids = torch.tensor(input_ids, dtype=torch.long)
        padded_attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        padded_decoder_input_ids= torch.tensor(decoder_input_ids, dtype=torch.long)
        padded_decoder_attention_mask= torch.tensor(decoder_attention_mask, dtype=torch.long)

        # source_mask = torch.cat([f.source_mask for f in train_features], dim=0)       
        # target_ids = torch.cat([f.target_ids for f in train_features], dim=0)

        eval_data = TensorDataset(
            padded_input_ids, padded_attention_mask, padded_decoder_input_ids,padded_decoder_attention_mask
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
            decoder_input_ids = inputs['decoder_input_ids']
            
          
            with torch.no_grad():
                outputs = self.model(**inputs)
                lm_logits = F.linear(outputs[0], self.model.shared.weight, bias=self.model.final_logits_bias)
        
                loss_fct = nn.CrossEntropyLoss(reduction="sum", ignore_index=self.model.config.pad_token_id)
                loss = loss_fct(lm_logits.view(-1, self.model.config.vocab_size),
                              decoder_input_ids.view(-1))
                
            

            if self.args.n_gpu > 1:
                loss = loss.mean()
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps
            eval_loss += loss      
            nb_eval_steps += 1

        
        eval_loss = eval_loss / nb_eval_steps
        results["eval_loss"] = eval_loss

        output_eval_file = os.path.join(self.args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))
       
        result , preds = self.predict(eval_data, output_dir='eval/')
        results.update(result)# correct_num, correct_ratio
        result = self.compute_metrics(eval_data["target_text"].tolist(), preds, **kwargs)
        results.update(result)# metrics
        
        return results
            

    def predict(self, pred_data, output_dir=None, suffix=None, verbose=True, silent=False):
        """
        Performs predictions on a list of text.
        Args:
            pred_data: Pandas DataFrame containing the 2 columns - `input_text`, `target_text`.
                        - `input_text`: The input text sequence.
                        - `target_text`: The target text sequence.            
            output_dir: The directory where predictition results files will be saved. If not given, self.args.output_dir will be used.
            suffix: The supplementary suffix of prediction results name.
        Returns:
            preds: A python list of the generated sequences.
        """  # noqa: ignore flake8"

        to_predict = pred_data["input_text"].tolist()
        target_predict = pred_data["target_text"].tolist()

        assert len(to_predict)==len(target_predict)

        
        if not output_dir:
            output_dir = self.args.output_dir

        

        os.makedirs(output_dir, exist_ok=True)


        all_outputs = []
        # Batching
        for batch in tqdm(
            [to_predict[i : i + self.args.eval_batch_size] for i in range(0, len(to_predict), self.args.eval_batch_size)],
            desc='Predicting', 
            disable=self.args.silent, 
            mininterval=0,):
            
            input_ids = self.encoder_tokenizer.batch_encode_plus(
                batch,
                max_length=self.args.max_seq_length,
                padding=True,
                return_tensors="pt",
                truncation=True,
            )["input_ids"]
            input_ids = input_ids.to(self.device)

            outputs = self.model.generate(
                input_ids=input_ids,
                num_beams=self.args.num_beams,
                max_length=self.args.max_length,
                length_penalty=self.args.length_penalty,
                early_stopping=self.args.early_stopping,
                repetition_penalty=self.args.repetition_penalty,
                do_sample=self.args.do_sample,
                top_k=self.args.top_k,
                top_p=self.args.top_p,
                num_return_sequences=self.args.num_return_sequences,
            )
            
            all_outputs.extend(outputs.cpu().numpy())

        if self.args.use_multiprocessed_decoding:
            self.model.to("cpu")
            with Pool(self.args.process_count) as p:
                outputs = list(
                    tqdm(
                        p.imap(self._decode, all_outputs, chunksize=self.args.multiprocessing_chunksize),
                        total=len(all_outputs),
                        desc="Decoding outputs",
                        disable=self.args.silent,
                    )
                )
            self._move_model_to_device()
        else:
            outputs = [
                self.tokenizer.decode(output_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for output_id in all_outputs
            ]

        output_predication_file = os.path.join(output_dir, "predictions_{}.txt".format(suffix))
        correct_num = 0

        with open(output_predication_file, "w", encoding="utf8", errors="ignore") as writer:
            writer.write("to_predict\n\toutput\n\ttarget\n")
            for i in range(len(outputs)):
                outputs[i] = outputs[i].strip()
                writer.write(to_predict[i]+"\t"+outputs[i]+"\n\t"+target_predict[i])
                if outputs[i].strip().lower() in target_predict[i].strip().lower().split('\t'):
                        print(outputs[i].strip().lower()+'\n'+str(target_predict[i].strip().lower().split('\t')))
                        correct_num += 1

        correct_ratio = correct_num/float(len(outputs))
        print("correct number: {}, correct ratio: {}".format(correct_num, correct_ratio))

        if self.args.num_return_sequences > 1:
            outputs = [
                outputs[i : i + self.args.num_return_sequences]
                for i in range(0, len(outputs), self.args.num_return_sequences)
            ]
        else:
            outputs = outputs

        return {
            result:{
                correct_num:correct_num,
                correct_ratio: correct_ratio, 
            },          
            outputs:outputs
        }


    def _decode(self, output_id):
        return self.tokenizer.decode(output_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    def compute_metrics(self, labels, preds, **kwargs):
        """
        Computes the evaluation metrics for the model predictions.

        Args:
            labels: List of target sequences
            preds: List of model generated outputs
            **kwargs: Custom metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use).
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions. Both inputs
                        will be lists of strings. Note that this will slow down evaluation significantly as the predicted sequences need to be generated.

        Returns:
            result: Dictionary containing evaluation results.
        """  # noqa: ignore flake8"
        # assert len(labels) == len(preds)

        results = {}

        for metric, func in kwargs.items():
            results[metric] = func(labels, preds)

        return results    
   