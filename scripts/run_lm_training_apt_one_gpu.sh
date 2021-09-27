set -e -x

start_checkpoint='roberta-base'
train_data_file="datasets/mlm/test.txt"
save_model_dir="saved_model/"

model_conf_path=${start_checkpoint}
model_vocab_path=${start_checkpoint}

if [ ! -d ${save_model_dir} ]
then
	mkdir -p ${save_model_dir} 
else
	echo "${save_model_dir} existed"
fi

epoch=100
seq_length=128
batch_size=64
grad=64
learning_rate="4e-4"
logging_steps=500

python model/mlm/run_language_modeling.py \
	--seed 42 \
	--line_by_line \
	--mlm \
	--mlm_probability 0.15 \
        --train_data_file ${train_data_file} \
	--output_dir ${save_model_dir} \
   	--config_name ${model_conf_path}  \
    	--tokenizer_name ${model_vocab_path} \
    	--do_train \
	--per_device_train_batch_size ${batch_size} \
	--num_train_epochs ${epoch} \
	--overwrite_output_dir \
	--learning_rate ${learning_rate} \
	--weight_decay 0.01 \
	--adam_epsilon  1e-6 \
	--logging_steps ${logging_steps} \
	--block_size ${seq_length} \
	--gradient_accumulation_steps ${grad} \
	--overwrite_cache \
	--model_name_or_path ${start_checkpoint}
	

