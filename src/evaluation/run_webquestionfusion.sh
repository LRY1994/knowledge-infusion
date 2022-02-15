DATASET="WebQuestion"
# TriviaQA
# NaturalQuestions
# SQuAD2

MODEL_DIR="../train_adpter/relation_prompt/checkpoints/"
DATA_DIR="../datasets/WebQuestion/splitted/" 
BASE_MODEL="facebook/bart-large"
MODEL="WebQuestion"
T=1
LR=1e-5
# TRAIN_MODE="fusion"
TRAIN_MODE="base"

    
python eval_webquestion.py \
--train_mode $TRAIN_MODE \
--model_dir $MODEL_DIR \
--data_dir $DATA_DIR  \
--base_model $BASE_MODEL \
--tokenizer $BASE_MODEL  \
--model $MODEL  \
--max_seq_length 512   \
--batch_size 4 \
--lr $LR   \
--pretrain_epoch 0 \
--epochs 25 \
--temperature $T \
--cuda

