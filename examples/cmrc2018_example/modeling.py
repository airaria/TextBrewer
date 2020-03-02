import logging
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
import torch.nn.functional as F
import config as Conf
from pytorch_pretrained_bert.my_modeling import BertModel, BertLayerNorm


def initializer_builder(std):
    _std = std
    def init_bert_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=_std)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    return init_bert_weights


class BertForQA(nn.Module):
    def __init__(self, config):
        super(BertForQA, self).__init__()
        self.bert = BertModel(config, output_score=True, output_sum=1)
        self.qa_outputs  = nn.Linear(config.hidden_size, 2)
        self.cls_outputs = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        initializer = initializer_builder(config.initializer_range)
        self.apply(initializer)
        self.num_heads = config.num_attention_heads

    def forward(self, input_ids, token_type_ids, attention_mask, doc_mask,
                start_positions=None, end_positions=None, start_logits_T=None, end_logits_T=None,
                attention_probs_sum_layer=None, attention_probs_sum_T=None):
        if attention_probs_sum_layer is not None:
            sequence_output, pooled_output, all_attention_probs_sum = self.bert(input_ids, token_type_ids, attention_mask,
                                                                                output_all_encoded_layers=False, output_attention_layer=[attention_probs_sum_layer])
            attention_probs_sum = all_attention_probs_sum[attention_probs_sum_layer]
        else:
            sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,  output_all_encoded_layers=False)
        output_for_cls = self.dropout(pooled_output)

        span_logits = self.qa_outputs(sequence_output)
        cls_logits  = self.cls_outputs(output_for_cls).squeeze(-1)  # output size: batch_size

        doc_mask[:,0] = 0
        span_logits = span_logits + (1.0 - doc_mask.unsqueeze(-1)) * -10000.0
        start_logits, end_logits = span_logits.split(1, dim=-1)  # use_cls
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if start_logits_T is not None or start_positions is not None:
            total_loss = 0
            att_loss = None
            if start_logits_T is not None:
                temp=Conf.args.temperature
                temp_2 = temp*temp
                start_logits_T /= temp
                end_logits_T /= temp
                start_logits /= temp
                end_logits /= temp
                start_prob_T = F.softmax(start_logits_T,dim=-1)
                end_prob_T = F.softmax(end_logits_T,dim=-1)
                ce_loss = -(start_prob_T * F.log_softmax(start_logits, dim=-1) + end_prob_T * F.log_softmax(end_logits, dim=-1)).sum(dim=-1)
                ce_loss = ce_loss.mean() #* temp_2
                total_loss += ce_loss
                if attention_probs_sum_T is not None:
                    attention_probs_sum_T = (attention_probs_sum_T / self.num_heads)
                    attention_probs_sum   = (attention_probs_sum   / self.num_heads)
                    attention_probs_sum_T = F.softmax(attention_probs_sum_T, dim=-1)
                    att_loss = -((attention_probs_sum_T * F.log_softmax(attention_probs_sum, dim=-1)).sum(dim=-1) * attention_mask.to(attention_probs_sum)).sum()/attention_mask.sum() * Conf.args.att_loss_weight
                    #att_loss = F.mse_loss(attention_probs_sum, attention_probs_sum_T) * Conf.args.att_loss_weight
                    total_loss += att_loss
                #mle_loss = (F.mse_loss(start_logits,start_logits_T) + F.mse_loss(end_logits,end_logits_T))/2
                #total_loss += mle_loss
            if start_positions is not None:
                # If we are on multi-GPU, split add a dimension - if not this is a no-op
                if len(start_positions.size()) > 1:
                    start_positions = start_positions.squeeze(-1)
                    end_positions = end_positions.squeeze(-1)
                # sometimes the start/end positions are outside our model inputs, we ignore these terms
                ignored_index = start_logits.size(1)
                start_positions.clamp_(0, ignored_index)
                end_positions.clamp_(0, ignored_index)
                is_noans = (start_positions == 0).float()
                loss_fct_span  = CrossEntropyLoss(ignore_index=ignored_index,reduction='none')
                #loss_fct_noans = BCEWithLogitsLoss()
                start_loss = (loss_fct_span(start_logits, start_positions)*(1-is_noans)).mean()
                end_loss   = (loss_fct_span(end_logits,   end_positions  )*(1-is_noans)).mean()
                #cls_loss   = loss_fct_noans(cls_logits, is_noans)
                total_loss += (start_loss + end_loss)/2 #+ cls_loss
            return total_loss, att_loss
        else:
            if attention_probs_sum_layer is not None:
                return start_logits, end_logits, cls_logits, attention_probs_sum
            else:
                return start_logits, end_logits, cls_logits


class BertForQASimple(nn.Module):
    def __init__(self, config,args):
        super(BertForQASimple, self).__init__()
        self.output_encoded_layers   = (args.output_encoded_layers=='true')
        self.output_attention_layers = (args.output_attention_layers=='true')
        self.bert = BertModel(config, output_score=(args.output_att_score=='true'), output_sum=(args.output_att_sum=='true'))
        self.qa_outputs  = nn.Linear(config.hidden_size, 2)
        self.cls_outputs = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        initializer = initializer_builder(config.initializer_range)
        self.apply(initializer)

    def forward(self, input_ids, token_type_ids, attention_mask, doc_mask,
                start_positions=None, end_positions=None):
        sequence_output, pooled_output, attention_output = self.bert(input_ids, token_type_ids, attention_mask,
                                                                    output_all_encoded_layers=(self.output_encoded_layers),
                                                                    output_all_attention_layers=(self.output_attention_layers))
        #output_for_cls = self.dropout(pooled_output)
        if self.output_encoded_layers is True:
            span_logits = self.qa_outputs(sequence_output[-1])
        else:
            span_logits = self.qa_outputs(sequence_output)
        #cls_logits  = self.cls_outputs(output_for_cls).squeeze(-1)  # output size: batch_size

        doc_mask[:,0] = 0
        span_logits = span_logits + (1.0 - doc_mask.unsqueeze(-1)) * -10000.0
        start_logits, end_logits = span_logits.split(1, dim=-1)  # use_cls
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if start_positions is not None:
            total_loss = 0
            # If we are on multi-GPU, split add a dimension - if not this is a no-op
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)
            is_noans = (start_positions == 0).float()
            loss_fct_span  = CrossEntropyLoss(ignore_index=ignored_index,reduction='none')
            #loss_fct_noans = BCEWithLogitsLoss()
            start_loss = (loss_fct_span(start_logits, start_positions)*(1-is_noans)).mean()
            end_loss   = (loss_fct_span(end_logits,   end_positions  )*(1-is_noans)).mean()
            #cls_loss   = loss_fct_noans(cls_logits, is_noans)
            total_loss += (start_loss + end_loss)/2 #+ cls_loss
            return start_logits, end_logits, sequence_output, attention_output, total_loss
        else:
            return start_logits, end_logits

#def BertForQASimpleAdaptor(batch, model_outputs):
#    return {'logits':      (model_outputs[0],model_outputs[1]),
#            'hidden':      model_outputs[2],
#            'attention':   model_outputs[3],
#            'inputs_mask': batch[2]
#           }

def BertForQASimpleAdaptor(batch, model_outputs, no_mask=False, no_logits=False):
    dict_obj = {'hidden':      model_outputs[2], 'attention':   model_outputs[3]}
    if no_mask is False:
        dict_obj['inputs_mask'] = batch[2]
    if no_logits is False:
        dict_obj['logits'] = (model_outputs[0],model_outputs[1])
    return dict_obj

def BertForQASimpleAdaptorNoMask(batch, model_outputs):
    return {'logits':      (model_outputs[0],model_outputs[1]),
            'hidden':      model_outputs[2],
            'attention':   model_outputs[3]}

def BertForQASimpleAdaptorTraining(batch, model_outputs):
    return {'losses':(model_outputs[4],)}
