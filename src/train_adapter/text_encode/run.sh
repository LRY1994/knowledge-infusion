#Convert the original Wikidata5M files into the numerical format used in pre-training:
# text="/home/simon/wikidata5m/wikidata5m_text.txt"
# train="/home/simon/wikidata5m/wikidata5m_transductive_train.txt"
# valid="/home/simon/wikidata5m/wikidata5m_transductive_valid.txt "
# converted_text="./data/Qdesc.txt "
# converted_train="./data/train.txt"
# converted_valid="./data/valid.txt"


# python convert.py \
# --text $text \
# --train $train \
# --valid $valid\
# --converted_text $converted_text \
# --converted_train $converted_train \
# --converted_valid $converted_valid


#Encode the entity descriptions with the GPT-2 BPE:
# mkdir -p gpt2_bpe
# wget -O gpt2_bpe/encoder.json https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json
# wget -O gpt2_bpe/vocab.bpe https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe
python  /home/simon/KEPLER/examples/roberta/multiprocessing_bpe_encoder.py \
		--encoder-json gpt2_bpe/encoder.json \
		--vocab-bpe gpt2_bpe/vocab.bpe \
		--inputs ./data/Qdesc.txt \
		--outputs ./data/Qdesc.bpe \
		--keep-empty \
		--workers 20