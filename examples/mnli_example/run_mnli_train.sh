#set hyperparameters
BERT_DIR=/path/to/bert-base-cased
OUTPUT_ROOT_DIR=/path/to/output_root_dir
DATA_ROOT_DIR=/path/to/data_root_dir


STUDENT_CONF_DIR=../student_config/bert_base_cased_config

accu=1
ep=3
lr=2
batch_size=24
length=128
sopt1=30 # The final learning rate is 1/sopt1 of the initial learning rate
torch_seed=9580

taskname='mnli'
NAME=${taskname}_base_lr${lr}e${ep}_bs${batch_size}_teacher
DATA_DIR=${DATA_ROOT_DIR}/MNLI
OUTPUT_DIR=${OUTPUT_ROOT_DIR}/${NAME}



mkdir -p $OUTPUT_DIR


python -u main.trainer.py \
    --vocab_file $BERT_DIR/vocab.txt \
    --data_dir  $DATA_DIR \
    --bert_config_file_T none \
    --bert_config_file_S $BERT_DIR/bert_config.json \
    --init_checkpoint_S  $BERT_DIR/pytorch_model.bin \
    --do_train \
    --do_eval \
    --do_predict \
    --max_seq_length ${length} \
    --train_batch_size ${batch_size} \
    --random_seed $torch_seed \
    --num_train_epochs ${ep} \
    --learning_rate ${lr}e-5 \
    --ckpt_frequency 1 \
    --schedule slanted_triangular \
    --s_opt1 ${sopt1} \
    --output_dir $OUTPUT_DIR \
    --gradient_accumulation_steps ${accu} \
    --task_name ${taskname} \
    --output_att_sum false  \
    --output_encoded_layers false \
    --output_attention_layers false
