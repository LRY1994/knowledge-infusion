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