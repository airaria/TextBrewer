#set hyperparameters
OUTPUT_ROOT_DIR=/path/to/output_root_dir
DATA_ROOT_DIR=/path/to/data_root_dir


accu=1
ep=10
lr=2
temperature=8
batch_size=32
length=128
torch_seed=9580

taskname='mnli'
NAME=${taskname}_t${temperature}_MTbaseST4tiny_lr${lr}e${ep}_bs${batch_size}
DATA_DIR=${DATA_ROOT_DIR}/MNLI
OUTPUT_DIR=${OUTPUT_ROOT_DIR}/${NAME}

mkdir -p $OUTPUT_DIR
model_config_json_file=DistillMultiBertToTiny.json
cp jsons/${model_config_json_file} ${OUTPUT_DIR}/${model_config_json_file}.run


python -u main.multiteacher.py \
    --data_dir  $DATA_DIR \
    --do_train \
    --do_eval \
    --do_predict \
    --max_seq_length ${length} \
    --train_batch_size ${batch_size} \
    --random_seed $torch_seed \
    --num_train_epochs ${ep} \
    --learning_rate ${lr}e-5 \
    --ckpt_frequency 1 \
    --output_dir $OUTPUT_DIR \
    --gradient_accumulation_steps ${accu} \
    --temperature ${temperature} \
    --task_name ${taskname} \
    --model_config_json ${OUTPUT_DIR}/${model_config_json_file}.run \
    --fp16
