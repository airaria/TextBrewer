ELECTRA_DIR_BASE=/path/to/Chinese-electra-base
OUTPUT_DIR=/path/to/output_dir
DATA_DIR=/path/to/MSRA_NER_data

mkdir -p $OUTPUT_DIR

ngpu=2
lr=10
batch_size=12
length=160
ep=2
lr=30

python -m torch.distributed.launch --nproc_per_node=${ngpu} main.train.dist.py \
    --vocab_file $ELECTRA_DIR_BASE/vocab.txt \
    --do_lower_case \
    --bert_config_file_T none \
    --bert_config_file_S $ELECTRA_DIR_BASE/config.json \
    --init_checkpoint_S $ELECTRA_DIR_BASE/pytorch_model.bin \
    --do_train \
    --do_eval \
    --do_predict \
    --max_seq_length ${length} \
    --train_batch_size ${batch_size} \
    --random_seed 1337 \
    --train_file $DATA_DIR/msra_train_bio.txt \
    --predict_file $DATA_DIR/msra_test_bio.txt \
    --num_train_epochs ${ep} \
    --learning_rate ${lr}e-5 \
    --ckpt_frequency 2 \
    --official_schedule linear \
    --output_dir $OUTPUT_DIR \
    --gradient_accumulation_steps 1 \
    --output_encoded_layers true \
    --output_attention_layers false \
    --lr_decay 0.8
    #--fp16
