MODEL_PATH='../../train_adpter/relation_prompt/checkpoints/'
DATA_PATH="../../datasets/WebQuestion/" 
OUTPUT_DIR="./output/WebQuestion/" 
MAX_SEQ_LENGTH=64
MAX_LENGTH=64
TRAIN_BATCH_SIZE=8
BASE_MODEL="facebook/bart-base"
MODEL="WebQuestion"
T=1
LR=1e-5
TRAIN_MODE="fusion"

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
--evaluation_metric qa \
--train_mode $TRAIN_MODE \
--model_dir $MODEL_DIR \
--data_dir $DATA_DIR  \
--base_model $BASE_MODEL \
--tokenizer $BASE_MODEL  \
--model $MODEL  \
--max_seq_length 512   \
--batch_size 8 \
--lr $LR   \
--pretrain_epoch 0 \
--epochs 25 \
--temperature $T \
--cuda