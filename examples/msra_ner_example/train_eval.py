import torch
from collections import OrderedDict
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import os, logging
from tqdm import tqdm, trange
import json
import numpy as np
from seqeval.metrics import accuracy_score, precision_score, f1_score, classification_report
from utils_ner import id2label_dict

logger = logging.getLogger("Train_eval")

def ensemble(models, eval_examples, eval_dataset, step, args):
    device = args.device
    logger.info("Predicting...")
    logger.info("***** Running predictions *****")
    logger.info("  Num orig examples = %d", len(eval_examples))
    logger.info("  Num split examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.predict_batch_size)

    if args.local_rank == -1:
        eval_sampler = SequentialSampler(eval_dataset)
    else:
        eval_sampler = DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.predict_batch_size)

    for model in models:
        model.eval()
    all_labels = []
    all_predictions = []
    logger.info("Start evaluating")

    for input_ids, input_mask, labels in tqdm(eval_dataloader, desc="Evaluating", disable=None):
        if len(all_predictions) % 1000 == 0:
            logger.info("Processing example: %d" % (len(all_predictions)))
        input_ids = input_ids.to(device)
        lengths = input_mask.sum(dim=-1) # batch_size
        input_mask = input_mask.to(device)

        with torch.no_grad():
            logits_list = [model(input_ids, input_mask)[0] for model in models]
            logits = sum(logits_list)/len(logits_list)
            predictions = logits.argmax(dim=-1) #batch_size * length
            for i in range(len(labels)):
                length = lengths[i]
                eval_label = [id2label_dict[k] for k in labels[i][1:length-1].tolist()]
                eval_prediction = [id2label_dict[k] for k in predictions[i].cpu()[1:length-1].tolist()]
                assert len(eval_label) == len(eval_prediction)

                all_labels.append(eval_label)
                all_predictions.append(eval_prediction)

    for model in models:
        model.train()
    #eval
    f1 = f1_score(all_labels, all_predictions) * 100
    precision = precision_score(all_labels, all_predictions) * 100
    accuracy  = accuracy_score(all_labels, all_predictions) * 100
    report = classification_report(all_labels, all_predictions)

    logger.info("Eval results:")
    logger.info(f"\nF1 : {f1:.3f}\nP  : {precision:.3f}\nAcc: {accuracy:.3f}")
    logger.info(f"\n{report}")

    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    with open(output_eval_file, "a") as writer:
        writer.write(f"Step: {step}\nF1: {f1:.3f}\nP: {precision:.3f}\nAcc: {accuracy:.3f}")

    logger.info("Write predictions...")
    output_prediction_file = os.path.join(args.output_dir, "predictions_%d.json" % step)
    write_predictions(eval_examples, all_labels, all_predictions, output_prediction_file)

def ddp_predict(model, eval_examples, eval_dataset, step, args):
    if args.local_rank == -1 or torch.distributed.get_rank() == 0:
        logger.info(f"Do predict in local rank : {args.local_rank}")
        predict(model, eval_examples, eval_dataset, step, args)
        if args.local_rank != -1: # DDP is enabled
            torch.distributed.barrier()
    else:
        torch.distributed.barrier()

def predict(model, eval_examples, eval_dataset, step, args):
    device = args.device
    logger.info("Predicting...")
    logger.info("***** Running predictions *****")
    logger.info("  Num orig examples = %d", len(eval_examples))
    logger.info("  Num split examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.predict_batch_size)

    eval_sampler = SequentialSampler(eval_dataset) # eval in single process mode
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.predict_batch_size)

    model.eval()
    all_labels = []
    all_predictions = []
    logger.info("Start evaluating")

    for input_ids, input_mask, labels in tqdm(eval_dataloader, desc="Evaluating", disable=None):
        if len(all_predictions) % 1000 == 0:
            logger.info("Processing example: %d" % (len(all_predictions)))
        input_ids = input_ids.to(device)
        lengths = input_mask.sum(dim=-1) # batch_size
        input_mask = input_mask.to(device)

        with torch.no_grad():
            logits, _ = model(input_ids, input_mask)
            predictions = logits.argmax(dim=-1) #batch_size * length
            for i in range(len(labels)):
                length = lengths[i]
                eval_label = [id2label_dict[k] for k in labels[i][1:length-1].tolist()]
                eval_prediction = [id2label_dict[k] for k in predictions[i].cpu()[1:length-1].tolist()]
                assert len(eval_label) == len(eval_prediction)

                all_labels.append(eval_label)
                all_predictions.append(eval_prediction)

    model.train()
    #eval
    f1 = f1_score(all_labels, all_predictions) * 100
    precision = precision_score(all_labels, all_predictions) * 100
    accuracy  = accuracy_score(all_labels, all_predictions) * 100
    report = classification_report(all_labels, all_predictions)

    logger.info("Eval results:")
    logger.info(f"\nF1 : {f1:.3f}\nP  : {precision:.3f}\nAcc: {accuracy:.3f}")
    logger.info(f"\n{report}")

    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    with open(output_eval_file, "a") as writer:
        writer.write(f"Step: {step}\nF1: {f1:.3f}\nP: {precision:.3f}\nAcc: {accuracy:.3f}")

    logger.info("Write predictions...")
    output_prediction_file = os.path.join(args.output_dir, "predictions_%d.json" % step)
    write_predictions(eval_examples, all_labels, all_predictions, output_prediction_file)


def write_predictions(examples, labels, predictions,filename):
    assert len(examples)==len(predictions), (len(examples),len(predictions))
    with open(filename,'w') as f:
        for i in range(len(examples)):
            example = examples[i]
            prediction = predictions[i]
            label = labels[i]
            for token,gt,pred in zip(example.tokens,label,prediction):
                f.write(f"{token}\t{gt}\t{pred}\n")
            f.write("\n")
