DATASET="WebQuestion"
# TriviaQA
# NaturalQuestions
# SQuAD2
MODEL_DIR="src/train_adpter/relation_prompt/checkpoints/"
DATA_DIR="src/datasets/WebQuestion/splitted/"
BASE_MODEL="facebook/bart-base"
MODEL="WebQuestion"
T=1
LR=1e-5
TRAIN_MODE="base"
OUTPUT_DIR="output"
python src/evaluation/eval_webquestion.py \
--train_mode $TRAIN_MODE \
--model_dir $MODEL_DIR \
--data_dir $DATA_DIR  \
--base_model $BASE_MODEL \
--tokenizer $BASE_MODEL  \
--model $MODEL  \
--max_seq_length 50  \
--batch_size 4 \
--eval_batch_size 4 \
--lr $LR   \
--pretrain_epoch 0 \
--epochs 2 \
--repeat_runs 2 \
--temperature $T \
--output_dir  $OUTPUT_DIR  \
--cuda \
