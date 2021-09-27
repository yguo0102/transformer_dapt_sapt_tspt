set -e 

model_name='roberta-base'
output_path=output_roberta_cls_metric

train_file='train.csv'
dev_file='dev.csv'
test_file='test.csv'

batch_size=32
epoch=10
save_steps=128

data_name="sample"
data_path="datasets/classification/sample"
metric='f1_micro'

for learning_rate in 3e-5 2e-5;
do
	for i in 42 62 82;
	do
		surfix2="${data_name}_${learning_rate}_${i}"
		output_dir="${output_path}/model_${surfix2}"

		#echo $output_dir
		if [ ! -d ${output_dir} ]
		then
			mkdir -p ${output_dir}
			echo "create folder ${output_dir}, running ${surfix2}..."
			date
		
			python model/classification/run_classification.py \
				--seed ${i} \
				--model_name_or_path ${model_name} \
	   			--config_name ${model_name}  \
	    			--tokenizer_name ${model_name} \
				--task_name social_media \
				--data_dir  ${data_path} \
				--train_file  ${train_file} \
				--dev_file  ${dev_file} \
				--test_file  ${test_file} \
				--metric ${metric} \
				--output_dir ${output_dir} \
				--per_device_train_batch_size ${batch_size} \
				--per_device_eval_batch_size ${batch_size} \
				--num_train_epochs ${epoch} \
				--overwrite_cache \
				--overwrite_output_dir \
				--save_steps ${save_steps} \
				--logging_steps 1 \
				 --evaluate_during_training \
				--learning_rate ${learning_rate} \
				--do_train --do_eval --do_predict
			echo "finish ${surfix2}"
			date
		fi
	done
done
