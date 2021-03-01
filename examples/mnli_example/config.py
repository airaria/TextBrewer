import argparse
from utils_glue import processors
args = None

def parse(opt=None):
    parser = argparse.ArgumentParser()

    ## Required parameters

    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--data_dir", default=None, type=str)
    parser.add_argument("--max_seq_length", default=128, type=int)
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
    parser.add_argument('--weight_decay_rate',type=float,default=0.01)
    parser.add_argument('--do_eval',action='store_true')
    parser.add_argument('--PRINT_EVERY',type=int,default=200)
    parser.add_argument('--ckpt_frequency',type=int,default=2)

    parser.add_argument("--temperature", default=1, type=float)

    parser.add_argument("--teacher_cached",action='store_true')
    parser.add_argument('--task_name',type=str,choices=list(processors.keys()))
    parser.add_argument('--aux_task_name',type=str,choices=list(processors.keys()),default=None)
    parser.add_argument('--aux_data_dir', type=str)

    parser.add_argument('--matches',nargs='*',type=str)
    parser.add_argument('--model_config_json',type=str)
    parser.add_argument('--do_test',action='store_true')


    global args
    if opt is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(opt)

if __name__ == '__main__':
    print (args)
    parse(['--SAVE_DIR','test'])
    print(args)
