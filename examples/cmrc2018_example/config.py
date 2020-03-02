import argparse

args = None

def parse(opt=None):
    parser = argparse.ArgumentParser()

    ## Required parameters

    parser.add_argument("--vocab_file", default=None, type=str, required=True,
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--train_file", default=None, type=str, help="SQuAD json for training. E.g., train-v2.0.json")
    parser.add_argument("--predict_file", default=None, type=str,
                        help="SQuAD json for predictions. E.g., dev-v2.0.json or test-v2.0.json")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Whether to lower case the input text. Should be True for uncased "
                             "models and False for cased models.")
    parser.add_argument("--max_seq_length", default=416, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--do_train", default=False, action='store_true', help="Whether to run training.")
    parser.add_argument("--do_predict", default=False, action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--predict_batch_size", default=8, type=int, help="Total batch size for predictions.")
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% "
                             "of training.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                             "output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--verbose_logging", default=False, action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precisoin instead of 32-bit")

    parser.add_argument('--random_seed',type=int,default=10236797)
    parser.add_argument('--fake_file_1',type=str,default=None)
    parser.add_argument('--fake_file_2',type=str,default=None)
    parser.add_argument('--load_model_type',type=str,default='bert',choices=['bert','all','none'])
    parser.add_argument('--weight_decay_rate',type=float,default=0.01)
    parser.add_argument('--do_eval',action='store_true')
    parser.add_argument('--PRINT_EVERY',type=int,default=200)
    parser.add_argument('--weight',type=float,default=1.0)
    parser.add_argument('--ckpt_frequency',type=int,default=2)

    parser.add_argument('--tuned_checkpoint_T',type=str,default=None)
    parser.add_argument('--tuned_checkpoint_S',type=str,default=None)
    parser.add_argument("--init_checkpoint_S", default=None, type=str)
    parser.add_argument("--bert_config_file_T", default=None, type=str, required=True)
    parser.add_argument("--bert_config_file_S", default=None, type=str, required=True)
    parser.add_argument("--temperature", default=1, type=float, required=False)
    parser.add_argument("--teacher_cached",action='store_true')

    parser.add_argument('--s_opt1',type=float,default=1.0, help="release_start / step1 / ratio")
    parser.add_argument('--s_opt2',type=float,default=0.0, help="release_level / step2")
    parser.add_argument('--s_opt3',type=float,default=1.0, help="not used / decay rate")
    parser.add_argument('--schedule',type=str,default='warmup_linear_release')
    parser.add_argument('--null_score_diff_threshold',type=float,default=99.0)

    parser.add_argument('--tag',type=str,default='RB')
    parser.add_argument('--no_inputs_mask',action='store_true')
    parser.add_argument('--no_logits', action='store_true')
    parser.add_argument('--output_att_score',default='true',choices=['true','false'])
    parser.add_argument('--output_att_sum', default='false',choices=['true','false'])
    parser.add_argument('--output_encoded_layers'  ,default='true',choices=['true','false'])
    parser.add_argument('--output_attention_layers',default='true',choices=['true','false'])
    parser.add_argument('--matches',nargs='*',type=str)
    global args
    if opt is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(opt)


if __name__ == '__main__':
    print (args)
    parse(['--SAVE_DIR','test'])
    print(args)
