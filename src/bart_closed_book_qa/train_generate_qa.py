import logging

import argparse
import os

import sklearn

from simpletransformers.seq2seq import Seq2SeqModel
from simpletransformers.t5 import T5Model

from data_reader.data_reader import read_data_source_target
import os
import random
import shutil
import time
from argparse import ArgumentParser
from datetime import datetime
from os import listdir
from statistics import mean, stdev

import numpy as np
import torch

from transformers import (
    AdamW,
    AdapterConfig,
    AdapterFusionConfig,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
)
from transformers import get_linear_schedule_with_warmup as WarmupLinearSchedule

import wandb
from fusionutils.bert_evaluator import BertEvaluator
from fusionutils.bert_trainer import BertTrainer
from fusionutils.bioasq_processor import BioAsqProcessor
from fusionutils.common_utils import print_args_as_table

wandb.init(project="WebQuestions")


def get_args():
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)
    parser = ArgumentParser(
        description="Evaluate model on WebQuestions dataset "
    )
    parser.add_argument(
        "--train_mode",
        default="fusion",
        type=str,
        required=True,
        help="three modes: fusion, adapter, base",
    )
    parser.add_argument("--model", default=None, type=str, required=True)
    parser.add_argument("--base_model", default=None, type=str, required=True)
    parser.add_argument("--tokenizer", default=None, type=str, required=False)
    parser.add_argument("--cuda", action="store_true", help="to use gpu")
    parser.add_argument("--amp", action="store_true", help="use auto mixed precision")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--repeat_runs", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--pretrain_epoch", type=int, default=50)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="t=1: softmax fusion, 0<t<1: gumbel softmax fusion, t<0: MOE",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=1,
        help="training examples ratio to be kept.",
    )
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.00001)
    parser.add_argument("--groups", type=str, default=None, help="groups to be chosen")

    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--train_file", default="train.tsv")
    parser.add_argument("--dev_file", default="dev.tsv")
    parser.add_argument("--test_file", default="test.tsv")

    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length.",
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Proportion of training to perform linear learning rate warmup for",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass",
    )
    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the source and target files for the task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type, choose from [seq2seq, T5]",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )


    # Other parameters
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the valid set.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run prediction on the test set.")
    parser.add_argument("--predict_on_valid", action="store_true", help="Whether to run prediction on the valid set. If yes, it will only run predication on valid set.")
    parser.add_argument("--init_model_weights", action="store_true", help="Whether to initialize the model weights")
    parser.add_argument("--overwrite_output_dir", action="store_true", help="Whether to overwrite on the existing output dir")
    parser.add_argument("--use_multiprocessed_decoding", action="store_true",
                        help="Whether to use multiprocess when decoding")
    parser.add_argument("--save_model_every_epoch", action="store_true",
                        help="Whether to save model every epoch during training")
    parser.add_argument("--predict_during_training", action="store_true",
                        help="Whether to predict after each checkpoint-saving during training")
    parser.add_argument("--evaluate_during_training", action="store_true",
                        help="Whether to evaluate after each checkpoint-saving during training")
    parser.add_argument(
        "--output_dir",
        default='output_dir/', type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--save_step",
        default=0, type=int,
        help="Save checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--train_batch_size",
        default=16, type=int,
        help="Size of each train batch",
    )
    parser.add_argument(
        "--eval_batch_size",
        default=16, type=int,
        help="Size of each eval/predict batch",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        default=1, type=int,
        help="gradient accumulation steps",
    )
    parser.add_argument(
        "--learning_rate",
        default=4e-5, type=float,
        help="learning rate",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=100, type=int,
        help="Number of train epochs",
    )
    parser.add_argument(
        "--max_seq_length",
        default=None, type=int,
        help="Max input seq length",
    )
    parser.add_argument(
        "--max_length",
        default=None, type=int,
        help="Max output seq length",
    )
    parser.add_argument(
        "--prediction_dir",
        default=None, type=str,
        help="The output directory where the predictions results will be written.",
    )
    parser.add_argument(
        "--prediction_suffix",
        default=None, type=str,
        help=" The supplementary suffix of prediction results name.",
    )
    parser.add_argument(
        "--mask_ratio",
        default=0.0, type=float,
        help="the proportion of masked words in the source",
    )
    parser.add_argument(
        "--mask_length",
        default="span-poisson", type=str,
        choices=['subword', 'word', 'span-poisson'],
        help="when masking words, the length of mask segments",
    )
    parser.add_argument(
        '--replace_length', default=-1, type=int,
        help='when masking N tokens, replace with 0, 1, or N tokens (use -1 for N)'
    )
    parser.add_argument(
        '--poisson_lambda',
        default=3.0, type=float,
        help='randomly shuffle sentences for this proportion of inputs'
    )
    parser.add_argument(
        '--dataloader_num_workers', default=0, type=int,
        help='the number of cpus used in collecting data in dataloader, '
             'note that if it is large than cpu number, the program may be stuck'
    )
    parser.add_argument(
        '--evaluation_metric', default='qa', type=str,
        help='if pretrain passages, use \'passage\', else use \'qa\''
    )
    args = parser.parse_args()
    return args


def evaluate_split(model, processor, tokenizer, args, split="dev"):
    evaluator = BertEvaluator(model, processor, tokenizer, args, split, True)
    result = evaluator.get_scores(silent=True)
    split_result = {}
    for k, v in result[0].items():
        split_result[f"{split}_{k}"] = v
    return split_result, result[1]


def get_tf_flag(args):
    from_tf = False
    if (
        (
            ("BioRedditBERT" in args.model)
            or ("BioBERT" in args.model)
            or ("SapBERT" in args.model)
        )
        and "step_" not in args.model
        and "epoch_" not in args.model
    ):
        from_tf = True

    if ("SapBERT" in args.model) and ("original" in args.model):
        from_tf = False
    return from_tf


def search_adapters(args):
    """[Search the model_path, take all the sub directions as adapter_names]

    Args:
        args (ArgumentParser)

    Returns:
        [dict]: {model_path:[adapter_names]}
    """
    adapter_paths_dic = {}
    if "," in args.model:
        for model in args.model.split(","):  # need to fusion from two or more models
            model_path = args.model_dir + model
            adapter_paths = [f for f in listdir(model_path)]
            print(f"Found {len(adapter_paths)} adapter paths")
            adapter_paths = check_adapter_names(model_path, adapter_paths)
            adapter_paths_dic[model_path] = adapter_paths
    else:
        model_path = args.model_dir + args.model
        adapter_paths = [f for f in listdir(model_path)]
        print(f"Found {len(adapter_paths)} adapter paths")
        adapter_paths = check_adapter_names(model_path, adapter_paths)
        adapter_paths_dic[model_path] = adapter_paths
    return adapter_paths_dic


def check_adapter_names(model_path, adapter_names):
    """[Check if the adapter path contrains the adapter model]

    Args:
        model_path ([type]): [description]
        adapter_names ([type]): [description]

    Raises:
        ValueError: [description]
    """
    checked_adapter_names = []
    print(f"Checking adapter namer:{model_path}:{len(adapter_names)}")
    for adapter_name in adapter_names:  # group_0_epoch_1
        adapter_model_path = os.path.join(model_path, adapter_name)
        if f"epoch_{args.pretrain_epoch}" not in adapter_name:
            # check pretrain_epoch
            continue
        if args.groups and int(adapter_name.split("_")[1]) not in set(args.groups):
            # check selected groups
            continue
        adapter_model_path = os.path.join(adapter_model_path, "pytorch_adapter.bin")
        assert os.path.exists(
            adapter_model_path
        ), f"{adapter_model_path} adapter not found."

        checked_adapter_names.append(adapter_name)
    print(f"Valid adapters ({len(checked_adapter_names)}):{checked_adapter_names}")
    return checked_adapter_names


def prepare_opt_sch(model, args):
    """Prepare optimizer and scheduler.

    Args:
        model ([type]): [description]
        args ([type]): [description]

    Returns:
        [type]: [description]
    """
    train_examples = processor.get_train_examples(args.data_dir, args.train_file)
    num_train_optimization_steps = (
        int(len(train_examples) / args.batch_size / args.gradient_accumulation_steps)
        * args.epochs
    )
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.lr,
        weight_decay=0.01,
        correct_bias=False,
    )
    scheduler = WarmupLinearSchedule(
        optimizer,
        num_training_steps=num_train_optimization_steps,
        num_warmup_steps=args.warmup_proportion * num_train_optimization_steps,
    )
    return optimizer, scheduler


def load_fusion_adapter_model(args,model_args):
    """Load fusion adapter model.

    Args:
        args ([type]): [description]

    Returns:
        [type]: [description]
    """
    adapter_names_dict = search_adapters(args)
    fusion_adapter_rename = []
    for model_path, adapter_names in adapter_names_dict.items():
        for adapter_name in adapter_names:
            adapter_dir = os.path.join(model_path, adapter_name)
            new_adapter_name = model_path[-14:][:-8] + "_" + adapter_name
            base_model = get_base_model(args,model_args)
            base_model.load_adapter(adapter_dir, load_as=new_adapter_name)
            print(f"Load adapter:{new_adapter_name}")
            fusion_adapter_rename.append(new_adapter_name)
    fusion_config = AdapterFusionConfig.load("dynamic", temperature=args.temperature)
    base_model.add_fusion(fusion_adapter_rename, fusion_config)
    base_model.set_active_adapters(fusion_adapter_rename)
    # config = AutoConfig.from_pretrained(
    #     os.path.join(adapter_dir, "adapter_config.json")
    # )
    # base_model.train_fusion([adapter_names])
    # return config, base_model
    return  base_model


def get_base_model(args,model_args):
    if args.model_type == 'seq2seq':
        base_model = Seq2SeqModel(
            encoder_decoder_type="bart",
            encoder_decoder_name=args.model_name_or_path,
            args=model_args,
        )
    elif args.model_type == 't5':
        base_model = T5Model(
            model_name=args.model_name_or_path,
            args=model_args,
        )
    else:
        raise ValueError(
            "The {} model is not supported now".format(args.model_type)
        )
    return base_model

def main():

    args = get_args()
    print(args)
    device = torch.device(
        "cuda" if (torch.cuda.is_available() and args.cuda) else "cpu"
    )
    n_gpu = torch.cuda.device_count()
    model_str = args.model
    if "/" in model_str:
        model_str = model_str.split("/")[1]
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Device:", str(device).upper())
    print("Number of GPUs:", n_gpu)
    print("AMP:", args.amp)
    train_acc_list = []
    dev_acc_list = []
    test_acc_list = []
    seed_list = []
    args.batch_size = args.batch_size // args.gradient_accumulation_steps
    args.device = device
    args.n_gpu = n_gpu  
    args.best_model_dir = f"./temp/model_{timestamp_str}/"
    # Record config on wandb
    wandb.config.update(args)
    print_args_as_table(args)


    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # if args.do_train == True:
    #     train_df = read_data_source_target(args.data_dir + "train.source", args.data_dir + "train.target")
    # else:
    #     train_df = None

    # if args.do_eval == True or args.evaluate_during_training == True:
    #     eval_df = read_data_source_target(args.data_dir + "valid.source", args.data_dir + "valid.target")
    # else:
    #     eval_df = None

    # if args.do_predict == True or args.predict_during_training == True:
    #     if args.predict_on_valid == True:
    #         test_df = read_data_source_target(args.data_dir + "valid.source", args.data_dir + "valid.target")
    #     else:
    #         test_df = read_data_source_target(args.data_dir + "test.source", args.data_dir + "test.target")
    # else:
    #     test_df = None

    model_args = {
        "reprocess_input_data": True,
        "overwrite_output_dir": args.overwrite_output_dir,
        "init_model_weights": args.init_model_weights,
        "max_seq_length": args.max_seq_length,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "save_eval_checkpoints": False,
        "save_model_every_epoch": args.save_model_every_epoch,
        "save_steps": args.save_step,
        "evaluate_during_training": args.evaluate_during_training,
        "evaluate_generated_text": True,
        "evaluate_during_training_verbose": True,
        "predict_during_training": args.predict_during_training,
        "use_multiprocessing": False,
        "output_dir": args.output_dir,
        "max_length": args.max_length,
        "manual_seed": 4,
        "mask_ratio": args.mask_ratio,
        "mask_length": args.mask_length,
        "replace_length": args.replace_length,
        "poisson_lambda": args.poisson_lambda,
        "fp16":False,
        "truncation":True,
        "dataloader_num_workers":args.dataloader_num_workers,
        "use_multiprocessed_decoding":args.use_multiprocessed_decoding,
        "evaluation_metric": args.evaluation_metric,
        "predict_on_valid": args.predict_on_valid
    }

    if args.tokenizer is None:
        args.tokenizer = args.base_model
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    processor = BioAsqProcessor(args)
    for i in range(args.repeat_runs):
        print(f"Start the {i}th training.")
        # Set random seed for reproducibility
        seed = int(time.time())
        print(f"Generate random seed {seed}.")
        seed_list.append(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        args.best_model_dir = f"./best_model/model_{seed}/"
        os.makedirs(args.best_model_dir, exist_ok=True)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(seed)
        if args.train_mode == "fusion":
            # args.base_model will be a folder of pre-trained models over partitions
            model = load_fusion_adapter_model(args,model_args)       
        elif args.train_mode == "base":
            model = get_base_model(args,model_args)
        
        model.to(device)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)
        optimizer, scheduler = prepare_opt_sch(model, args)

        print("Training Model")
        trainer = BertTrainer(model, optimizer, processor, scheduler, tokenizer, args)
        trainer.train()
        print("Evaluating Model")
        model = torch.load(args.best_model_dir + "model.bin")
        train_result = evaluate_split(model, processor, tokenizer, args, split="train")
        train_result[0]["run_num"] = i
        wandb.log(train_result[0])  # Record Dev Result
        train_acc_list.append(train_result[0]["train_accuracy"])
        dev_result = evaluate_split(model, processor, tokenizer, args, split="dev")
        dev_result[0]["run_num"] = i
        wandb.log(dev_result[0])  # Record Dev Result
        dev_acc_list.append(dev_result[0]["dev_accuracy"])
        test_result = evaluate_split(model, processor, tokenizer, args, split="test")
        test_result[0]["run_num"] = i
        wandb.log(test_result[0])  # Record Testing Result
        test_acc_list.append(test_result[0]["test_accuracy"])
        if (
            test_result[0]["test_accuracy"] < 0.86
        ):  # keep the models with excellent performance
            shutil.rmtree(args.best_model_dir)
        else:
            print(f"Saving model to {args.best_model_dir}.")
            print(f"test_accuracy of {test_result[0]['test_accuracy']}.")

        # # Train the model
        # if args.do_train == True:
        #     print("Training Model")
        #     model.train_model(train_data=train_df, eval_data=eval_df, test_data=test_df, output_dir=args.output_dir)

        # # Evaluate the model
        # if args.do_eval == True:
        #     print("Evaluating Model")
        #     results = model.eval_model(eval_data=eval_df)
        #     print(results)

        # # Use the model for prediction
        # if args.do_predict == True:
        #     print("Predicting Model")
        #     results = model.predict(pred_data=test_df, output_dir=args.prediction_dir, suffix=args.prediction_suffix)
        #     print(results)

    result = {}
    result["seed_list"] = seed_list
    result["train_acc_mean"] = mean(train_acc_list)  # average of the ten runs
    result["train_acc_std"] = stdev(train_acc_list)  # average of the ten runs
    result["dev_acc_mean"] = mean(dev_acc_list)  # average of the ten runs
    result["dev_acc_std"] = stdev(dev_acc_list)  # average of the ten runs
    result["test_acc_mean"] = mean(test_acc_list)  # average of the ten runs
    result["test_acc_std"] = stdev(test_acc_list)  # average of the ten runs
    wandb.config.update(result)

if __name__ == '__main__':
    main()