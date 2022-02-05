DATASET="WebQuestions"
# TriviaQA
# NaturalQuestions
# SQuAD2

MODEL_DIR="../train_adpter/relation_prompt/checkpoints/"
DATA_DIR="../data/WebQuestions/" 
BASE_MODEL="facebook/bart-base"
MODEL="WebQuestions"
T=1
LR=1e-5
TRAIN_MODE="fusion"
    
python eval.py \
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

