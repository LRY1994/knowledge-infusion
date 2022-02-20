import datetime
import sys
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, TensorDataset, Dataset
from tqdm import tqdm
import pandas as pd
from torch import Tensor, nn
from .abstract_processor import convert_examples_to_features
from .bert_evaluator import BertEvaluator
from transformers.modeling_bart import shift_tokens_right
from transformers import T5ForConditionalGeneration, BartForConditionalGeneration
from transformers.modeling_bart import shift_tokens_right
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
   
        inputs = {
            'input_ids':batch[0].to(device),#torch.Size([4, 512])
            'attention_mask':batch[1].to(device),#torch.Size([4, 512])
            'encoder_outputs':None,
            'decoder_input_ids':batch[2].to(device),#torch.Size([4, 512])
            'decoder_attention_mask':batch[3].to(device),#torch.Size([4, 512])
            'decoder_cached_states':None,
            'use_cache':False            
        }
        return inputs
 
    def train_epoch(self, train_dataloader,epoch_number):
        self.tr_loss = 0
        train_losses = []

        batch_iterator = tqdm(
            train_dataloader,
            desc=f"Running Epoch {epoch_number} of {self.args.epochs} ",
            mininterval=0,
        )
       
        
        for step, batch in enumerate(train_dataloader):
            self.model.train()
           
            inputs = self._get_inputs_dict(batch)

            decoder_input_ids = inputs['decoder_input_ids']
            _decoder_input_ids = shift_tokens_right(decoder_input_ids, self.model.config.pad_token_id)
          

            
            print('bbbbbbbbbbbbbbbbbbbbbbbbbbbbbb')
            print(inputs['input_ids'].shape)#torch.Size([4, 32])
            print(inputs['attention_mask'].shape)#torch.Size([4, 32])
            print(_decoder_input_ids.shape)#torch.Size([4,36])
            print(inputs['decoder_attention_mask'].shape)#torch.Size([4,36])

            print(self.model)
            outputs = self.model.model(
                input_ids=inputs['input_ids'],#torch.Size([4, 32])
                attention_mask=inputs['attention_mask'],#torch.Size([4, 32])
                encoder_outputs=None,
                decoder_input_ids=_decoder_input_ids,#torch.Size([4,36])
                decoder_attention_mask=inputs['decoder_attention_mask'],#torch.Size([4,36])
                decoder_cached_states=None,
                use_cache=False 
            )
          
            print('cccccccccccccccccccccccccccccc')
            print(outputs[0].shape)#torch.Size([4, 36, 50265])  应该是torch.Size([4, 36, 1024])
            # print(self.model.shared.weight.shape)#torch.Size([50265, 1024])
            # print(self.final_logits_bias.shape)#torch.Size([1, 50265])

            lm_logits = F.linear(outputs[0], self.model.model.shared.weight, bias=self.model.final_logits_bias)
       
            loss_fct = nn.CrossEntropyLoss(reduction="sum", ignore_index=self.model.config.pad_token_id)
            loss = loss_fct(lm_logits.view(-1, self.model.config.vocab_size),
                              decoder_input_ids.view(-1))
            
            
            # # model outputs are always tuple in pytorch-transformers (see doc)
            # loss = outputs[0]
            # print(type(loss))
            # # with open('ouput.txt','w') as f:
            # #     f.write(str(outputs))
            # # print('loss.item():',loss)
            # current_loss = loss.item()
            if self.args.n_gpu > 1:
                loss = loss.mean()
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps
            
            batch_iterator.set_description(
                f"Epochs {epoch_number}/{self.args.epochs}. Running Loss: {loss:9.4f}"
            )


            
            train_losses.append(loss.detach().cpu())
            loss.backward()

            # self.tr_loss += loss.item()
            self.nb_tr_steps += 1
            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.iterations += 1

        print(train_losses)

    def train(self):
        train_features = convert_examples_to_features(
            self.train_examples, self.args.max_seq_length, self.tokenizer
        )

          
        print("Number of examples: ", len(self.train_examples))
        print("Batch size:", self.args.batch_size)
        print("Num of steps:", self.num_train_optimization_steps)
        
        # print(train_features['decoder_input_ids'].shape)

        padded_input_ids = torch.LongTensor(train_features['input_ids'])
        padded_attention_mask = torch.LongTensor(train_features['attention_mask'])
        padded_decoder_input_ids= torch.LongTensor(train_features['decoder_input_ids'])
        padded_decoder_attention_mask= torch.LongTensor(train_features['decoder_attention_mask'])

        

        train_data = TensorDataset(
            padded_input_ids, 
            padded_attention_mask, 
            padded_decoder_input_ids,
            padded_decoder_attention_mask
        )

        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(
            train_data, 
            sampler=train_sampler, 
            batch_size=self.args.batch_size
        )

      
      
        for epoch in tqdm(range(self.args.epochs), file=sys.stdout, desc="Epoch"):
            self.train_epoch(train_dataloader,epoch)

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
    
