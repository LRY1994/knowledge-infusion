MODEL="roberta-base"
TOKENIZER="roberta-base"
INPUT_DIR="/home/simon/wikidata5m"
OUTPUT_DIR="checkpoints"
DATASET_NAME="wikidata5m"
ADAPTER_NAMES="entity_predict"
PARTITION=10
TRIPLE_PER_RELATION=5000

python run_pretrain.py \
--model $MODEL \
--tokenizer $TOKENIZER \
--input_dir $INPUT_DIR \
--output_dir $OUTPUT_DIR \
--n_partition $PARTITION \
--use_adapter \
--non_sequential \
--adapter_names  $ADAPTER_NAMES\
--amp \
--cuda \
--num_workers 32 \
--max_seq_length 64 \
--batch_size 64  \
--lr 1e-04 \
--epochs 2 \
--save_step 2000 \
--triple_per_relation $TRIPLE_PER_RELATION


### QA-tuning
Example training command for QA-tuning
```bash
MODEL_PATH=/* YOUR_LM-TUNED_MODEL_PATH */
DATA_PATH=/* YOUR_DATA_PATH */
OUTPUT_DIR=/* YOUR_OUTPUT_DIRECTORY */
MAX_SEQ_LENGTH=64
MAX_LENGTH=64
TRAIN_BATCH_SIZE=8

CUDA_VISIBLE_DEVICES=1 python train_generate_qa.py \
--model_type seq2seq \
--data_dir ${DATASET} \
--model_name_or_path $MODEL_PATH \
--output_dir $OUTPUT_DIR \
--do_train \
--max_seq_length $MAX_SEQ_LENGTH \
--max_length $MAX_LENGTH \
--train_batch_size $TRAIN_BATCH_SIZE \
--save_step 500 \
--num_train_epochs 30 \
--dataloader_num_workers 0 \
--overwrite_output_dir \
--predict_on_valid \
--gradient_accumulation_steps 4 \
--evaluation_metric qa
```

Example testing command for QA-tuning
```bash
MODEL_PATH=/* YOUR_LM-TUNED_MODEL_PATH */
DATA_PATH=/* YOUR_DATA_PATH */
OUTPUT_DIR=/* YOUR_OUTPUT_DIRECTORY */
PREDICTION_DIR=/* YOUR_PREDICTION_DIRECTORY */
MAX_SEQ_LENGTH=64
MAX_LENGTH=64
TRAIN_BATCH_SIZE=8

CUDA_VISIBLE_DEVICES=1 python train_generate_qa.py \
--model_type seq2seq \
--data_dir ${DATASET} \
--model_name_or_path $MODEL_PATH \
--prediction_dir $PREDICTION_DIR \
--max_seq_length $MAX_SEQ_LENGTH \
--max_length $MAX_LENGTH \
--train_batch_size $TRAIN_BATCH_SIZE \
--save_step 500 \
--num_train_epochs 30 \
--dataloader_num_workers 0 \
--overwrite_output_dir \
--do_predict \
--gradient_accumulation_steps 4 \
--evaluation_metric qa
```