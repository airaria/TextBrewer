export OUTPUT_DIR=output-model-T3
export BATCH_SIZE=32
export NUM_EPOCHS=2
export SAVE_STEPS=750
export SEED=42
export MAX_LENGTH=128
export BERT_MODEL_TEACHER=path/to/teacher/dir
python run_ner_distill.py \
--data_dir data \
--model_type bert \
--model_name_or_path $BERT_MODEL_TEACHER \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_gpu_train_batch_size $BATCH_SIZE \
--num_hidden_layers 3 \
--save_steps $SAVE_STEPS \
--learning_rate 1e-4 \
--warmup_steps 0.1 \
--seed $SEED \
--do_distill \
--do_train \
--do_eval \
--do_predict
