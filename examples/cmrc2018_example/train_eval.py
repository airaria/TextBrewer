import torch
from collections import OrderedDict
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import os, logging
from tqdm import tqdm, trange
from processing import RawResult, write_predictions_google
from cmrc2018_evaluate import evaluate
import json
import numpy as np

logger = logging.getLogger("Train_eval")
logger.setLevel(logging.INFO)
handler_stream = logging.StreamHandler()
handler_stream.setLevel(logging.INFO)
formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -  %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
handler_stream.setFormatter(formatter)
logger.addHandler(handler_stream)


def predict(model, eval_examples, eval_features, step, args):
    device = args.device
    logger.info("Predicting...")
    logger.info("***** Running predictions *****")
    logger.info("  Num orig examples = %d", len(eval_examples))
    logger.info("  Num split examples = %d", len(eval_features))
    logger.info("  Batch size = %d", args.predict_batch_size)

    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_doc_mask = torch.tensor([f.doc_mask for f in eval_features], dtype=torch.float)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_doc_mask, all_segment_ids, all_example_index)
    if args.local_rank == -1:
        eval_sampler = SequentialSampler(eval_data)
    else:
        eval_sampler = DistributedSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.predict_batch_size)

    model.eval()
    all_results = []
    logger.info("Start evaluating")

    if os.path.exists('all_results.tmp') and not args.do_train:
        pass  # all_results = pickle.load(open('all_results.tmp', 'rb'))
    else:
        for input_ids, input_mask, doc_mask, segment_ids, example_indices \
                in tqdm(eval_dataloader, desc="Evaluating", disable=None):
            if len(all_results) % 1000 == 0:
                logger.info("Processing example: %d" % (len(all_results)))
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            doc_mask = doc_mask.to(device)
            segment_ids = segment_ids.to(device)
            with torch.no_grad():
                batch_start_logits, batch_end_logits = model(input_ids, segment_ids, input_mask,
                                                                               doc_mask)
            for i, example_index in enumerate(example_indices):
                start_logits = batch_start_logits[i].detach().cpu().tolist()
                end_logits = batch_end_logits[i].detach().cpu().tolist()
                cls_logits = 0 # Not used batch_cls_logits[i].detach().cpu().tolist()
                eval_feature = eval_features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
                all_results.append(RawResult(unique_id=unique_id,
                                             start_logits=start_logits,
                                             end_logits=end_logits,
                                             cls_logits=cls_logits))
        if not args.do_train:
            pass
            # try:
            #    pickle.dump(all_results, open('all_results.tmp', 'wb'))
            # except:
            #    print("can't save all_results.tmp")

    logger.info("Write predictions...")
    output_prediction_file = os.path.join(args.output_dir, "predictions_%d.json" % step)

    all_predictions, scores_diff_json = \
        write_predictions_google(eval_examples, eval_features, all_results,
                                 args.n_best_size, args.max_answer_length,
                                 args.do_lower_case, output_prediction_file,
                                 output_nbest_file=None, output_null_log_odds_file=None)
    model.train()
    if args.do_eval:
        eval_data = json.load(open(args.predict_file, 'r', encoding='utf-8'))
        F1, EM, TOTAL, SKIP = evaluate(eval_data, all_predictions)  # ,scores_diff_json, na_prob_thresh=0)
        AVG = (EM+F1)*0.5
        output_result = OrderedDict()
        output_result['AVERAGE'] = '%.3f' % AVG
        output_result['F1'] = '%.3f' % F1
        output_result['EM'] = '%.3f' % EM
        output_result['TOTAL'] = TOTAL
        output_result['SKIP'] = SKIP
        logger.info("***** Eval results {} *****".format(step))
        logger.info(json.dumps(output_result)+'\n')

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "a") as writer:
            writer.write(f"Step: {step} {json.dumps(output_result)}\n")

    #output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_%d.json" % step)
    #output_null_odds_file = os.path.join(args.output_dir, "null_odds_%d.json" % (step))

    # torch.save(state_dict, os.path.join(args.output_dir,"EM{:.4f}_F{:.4f}_gs{}.pkl".format(em,f1,global_step)))
    # print ("saving at finish")
    # coreModel = model.module if 'DataParallel' in model.__class__.__name__ else model
    # torch.save(coreModel.state_dict(),os.path.join(args.output_dir,"%d.pkl" % (global_step)))
    # predict(global_step)
