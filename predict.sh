dataset=msr
model_name=sgb-a
max_word_len=3
model_dir=model/${dataset}/${model_name}-${max_word_len}/


CUDA_VISIBLE_DEVICES="2" python predict.py \
--test_set ./icwb2-data/testing/${dataset}_test.utf8 \
--model_dir ${model_dir} \
--checkpoint checkpoint_last.pt \
--output_file  ${model_dir}/predict.txt \
--batch_size 64 \
--replace_special_symbols \
--postprocess_punct

perl ./icwb2-data/scripts/score ./icwb2-data/gold/${dataset}_training_words.utf8 \
    ./icwb2-data/gold/${dataset}_test_gold.utf8 ${model_dir}/predict.txt 