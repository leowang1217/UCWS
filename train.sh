dataset=msr
model_name=sgb-c
max_word_len=4

if [ ! -d model  ];then
  mkdir model
fi
if [ ! -d model/${dataset}  ];then
  mkdir model/${dataset}
fi

model_dir=model/${dataset}/${model_name}-${max_word_len}/

if [ ! -d ${model_dir}  ];then
  mkdir ${model_dir}
fi

CUDA_VISIBLE_DEVICES="2" python train.py \
--train_set ./icwb2-data/training/${dataset}_training.utf8 ./icwb2-data/testing/${dataset}_test.utf8 \
--model_dir ${model_dir} \
--batch_size 64 \
--model ${model_name} \
--max_word_len ${max_word_len} \
--n_epoch 5 \
--replace_special_symbols