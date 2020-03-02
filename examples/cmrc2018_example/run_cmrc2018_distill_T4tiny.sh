#set hyperparameters
BERT_DIR=/path/to/roberta-wwm-base
OUTPUT_ROOT_DIR=/path/to/output_root_dir
DATA_ROOT_DIR=/path/to/data_root_dir
trained_teacher_model=/path/to/trained_teacher_model_file

STUDENT_CONF_DIR=../student_config/roberta_wwm_config
cmrc_train_file=$DATA_ROOT_DIR/cmrc2018/squad-style-data/cmrc2018_train.json
cmrc_dev_file=$DATA_ROOT_DIR/cmrc2018/squad-style-data/cmrc2018_dev.json
DA_file=$DATA_ROOT_DIR/drcd/DRCD_training.json # used for data augmentation

accu=1
ep=50
lr=10
temperature=8
batch_size=24
length=512
sopt1=1 # The final learning rate is 1/sopt1 of the initial learning rate; 30 is used in most cases
torch_seed=9580

NAME=cmrc2018_t${temperature}_TbaseST4tiny_AllSmmdH1_lr${lr}e${ep}_opt${sopt1}
OUTPUT_DIR=${OUTPUT_ROOT_DIR}/${NAME}



mkdir -p $OUTPUT_DIR


python -u main.distill.py \
    --vocab_file $BERT_DIR/vocab.txt \
    --do_lower_case \
    --bert_config_file_T $BERT_DIR/bert_config.json \
    --bert_config_file_S $STUDENT_CONF_DIR/bert_config_L4t.json \
    --tuned_checkpoint_T $trained_teacher_model \
    --load_model_type none \
    --do_train \
    --do_eval \
    --do_predict \
    --doc_stride 128 \
    --max_seq_length ${length} \
    --train_batch_size ${batch_size} \
    --random_seed $torch_seed \
    --train_file $cmrc_train_file \
    --fake_file_1 $DA_file \
    --predict_file $cmrc_dev_file \
    --num_train_epochs ${ep} \
    --learning_rate ${lr}e-5 \
    --ckpt_frequency 1 \
    --schedule slanted_triangular \
    --s_opt1 ${sopt1} \
    --output_dir $OUTPUT_DIR \
    --gradient_accumulation_steps ${accu} \
    --temperature ${temperature} \
    --output_att_score true \
    --output_att_sum false  \
    --output_encoded_layers true \
    --output_attention_layers true \
    --matches L4t_hidden_mse \
              L4_hidden_smmd \
    --tag RB \
