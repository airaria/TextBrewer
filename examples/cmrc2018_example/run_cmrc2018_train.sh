#set hyperparameters
BERT_DIR=/path/to/roberta-wwm-base
OUTPUT_ROOT_DIR=/path/to/output_root_dir
DATA_ROOT_DIR=/path/to/data_root_dir

STUDENT_CONF_DIR=../student_config/roberta_wwm_config
cmrc_train_file=$DATA_ROOT_DIR/cmrc2018/squad-style-data/cmrc2018_train.json
cmrc_dev_file=$DATA_ROOT_DIR/cmrc2018/squad-style-data/cmrc2018_dev.json

accu=1
ep=2
lr=3
batch_size=24
length=512
torch_seed=9580

NAME=cmrc2018_base_lr${lr}e${ep}_teacher
OUTPUT_DIR=${OUTPUT_ROOT_DIR}/${NAME}



mkdir -p $OUTPUT_DIR

python -u main.trainer.py \
    --vocab_file $BERT_DIR/vocab.txt \
    --do_lower_case \
    --bert_config_file_T none \
    --bert_config_file_S $STUDENT_CONF_DIR/bert_config.json \
    --init_checkpoint_S $BERT_DIR/pytorch_model.bin \
    --do_train \
    --do_eval \
    --do_predict \
    --doc_stride 320 \
    --max_seq_length ${length} \
    --train_batch_size ${batch_size} \
    --random_seed $torch_seed \
    --train_file $cmrc_train_file \
    --predict_file $cmrc_dev_file \
    --num_train_epochs ${ep} \
    --learning_rate ${lr}e-5 \
    --ckpt_frequency 1 \
    --schedule slanted_triangular \
    --s_opt1 30 \
    --output_dir $OUTPUT_DIR \
    --gradient_accumulation_steps ${accu} \
    --output_encoded_layers false \
    --output_attention_layers false
