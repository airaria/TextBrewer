import tokenization
from scipy.misc import logsumexp
from tqdm import tqdm
import json
import logging
import collections
import six
import numpy as np
import math
import config
#from tokenization import spacy_parser
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler_stream = logging.StreamHandler()
handler_stream.setLevel(logging.INFO)
formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -  %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
handler_stream.setFormatter(formatter)
logger.addHandler(handler_stream)

class SquadExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (tokenization.printable_text(self.qas_id))
        s += ", question_text: %s" % (
            tokenization.printable_text(self.question_text))
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.start_position:
            s += ", end_position: %d" % (self.end_position)
        if self.is_impossible:
            s += ", is_impossible: %s" % (self.is_impossible)
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 doc_mask,input_span_mask,
                 segment_ids,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.doc_mask = doc_mask
        self.input_span_mask = input_span_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

def _is_chinese_char(cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
        return True
    return False

def customize_tokenizer(text, do_lower_case=True):
    temp_x = ""
    text = tokenization.convert_to_unicode(text)
    for c in text:
        if _is_chinese_char(ord(c)) or tokenization._is_punctuation(c) or tokenization._is_whitespace(c) or tokenization._is_control(c):
            temp_x += " " + c + " "
        else:
            temp_x += c
    if do_lower_case:
        temp_x = temp_x.lower()
    return temp_x.split()

class ChineseFullTokenizer(object):
    """Runs end-to-end tokenziation."""

    def __init__(self, vocab_file, do_lower_case=True):
        self.vocab = tokenization.load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.wordpiece_tokenizer = tokenization.WordpieceTokenizer(vocab=self.vocab)
        self.do_lower_case = do_lower_case
    def tokenize(self, text):
        split_tokens = []
        for token in customize_tokenizer(text, do_lower_case=self.do_lower_case):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)

        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        return tokenization.convert_tokens_to_ids(self.vocab, tokens)


def read_squad_examples(input_file, is_training,do_lower_case):
    """Read a SQuAD json file into a list of SquadExample."""
    with open(input_file, "r",encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]

    examples = []
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]
            raw_doc_tokens = customize_tokenizer(paragraph_text, do_lower_case=do_lower_case)
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True

            k = 0
            temp_word = ""
            for c in paragraph_text:
                if tokenization._is_whitespace(c):
                    char_to_word_offset.append(k-1)
                    continue
                else:
                    temp_word += c
                    char_to_word_offset.append(k)
                if do_lower_case is True:
                    temp_word = temp_word.lower()
                if temp_word == raw_doc_tokens[k]:
                    doc_tokens.append(temp_word)
                    temp_word = ""
                    k += 1

            try:
                assert k==len(raw_doc_tokens)
            except AssertionError:
                print (len(raw_doc_tokens),len(doc_tokens))
                for i in range(min(len(doc_tokens),len(raw_doc_tokens))):
                    if raw_doc_tokens[i]!=doc_tokens[i]:
                        print (raw_doc_tokens[i-3:i+3],doc_tokens[i-3:i+3])
                        break
                print (''.join(doc_tokens[500:]))
                print ("----")
                print (''.join(raw_doc_tokens[500:]))
                raise AssertionError

            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                is_impossible = False
                start_position = None
                end_position = None
                orig_answer_text = None
                if is_training:
                    is_impossible = len(qa['answers']) == 0
                    if len(qa["answers"]) > 1:
                        pass
                        #raise ValueError(
                        #    "For training, each question should have less than 1 answer.")
                    if len(qa['answers']) == 0:
                        orig_answer_text = ""
                        start_position = end_position = 0 # use_cls
                    else:
                        answer = qa["answers"][0]
                        orig_answer_text = answer["text"]
                        if orig_answer_text not in paragraph_text:
                            logger.warning("Could not find answer")
                            continue
                        answer_offset = paragraph_text.index(orig_answer_text)
                        #answer_offset = answer["answer_start"]
                        answer_length = len(orig_answer_text)
                        start_position = char_to_word_offset[answer_offset]
                        end_position = char_to_word_offset[answer_offset + answer_length - 1]
                        # Only add answers where the text can be exactly recovered from the
                        # document. If this CAN'T happen it's likely due to weird Unicode
                        # stuff so we will just skip the example.
                        #
                        # Note that this means for training mode, every example is NOT
                        # guaranteed to be preserved.
                        actual_text = "".join(doc_tokens[start_position:(end_position + 1)])
                        cleaned_answer_text = "".join(tokenization.whitespace_tokenize(orig_answer_text))
                        if do_lower_case:
                            cleaned_answer_text = cleaned_answer_text.lower()
                        if actual_text.find(cleaned_answer_text) == -1:
                            logger.warning("Could not find answer: '%s' vs. '%s'",
                                               actual_text, cleaned_answer_text)
                            continue

                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=is_impossible)
                examples.append(example)
    return examples

def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000

    num_na = 0
    num_all = 0

    features = []
    for (example_index, example) in enumerate(tqdm(examples,disable=None)):
        query_tokens = tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []

        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)


        tok_start_position = None
        tok_end_position = None
        if is_training:
            if example.is_impossible:
                tok_start_position = -1
                tok_end_position = -1
            else:
                tok_start_position = orig_to_tok_index[example.start_position]
                if example.end_position < len(example.doc_tokens) - 1:
                    tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
                else:
                    tok_end_position = len(all_doc_tokens) - 1
                (tok_start_position, tok_end_position) = _improve_answer_span(
                    all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                    example.orig_answer_text)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)
            #if not is_training:
            #    break  # only want 1 span when prediction

        #assert len(doc_spans) == 1 or is_training
        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            doc_mask = []
            input_span_mask = []

            tokens.append("[CLS]")
            segment_ids.append(0)
            doc_mask.append(0)
            input_span_mask.append(1)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
                doc_mask.append(0)
                input_span_mask.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)
            doc_mask.append(0)
            input_span_mask.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context

                doc_token = all_doc_tokens[split_token_index]
                tokens.append(doc_token)
                segment_ids.append(0) # Zero for RoBERTa
                doc_mask.append(1)
                input_span_mask.append(1)
            tokens.append("[SEP]")
            segment_ids.append(0) # Zero for RoBERTa
            doc_mask.append(0)
            input_span_mask.append(0)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                #ent.append(0)
                #pos.append(0)
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                doc_mask.append(0)
                input_span_mask.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(doc_mask) == max_seq_length
            assert len(input_span_mask) == max_seq_length

            start_position = None
            end_position = None
            if is_training:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                if example.is_impossible:
                    start_position = 0
                    end_position = 0
                    #continue #TODO if doc_span_index>0: continue
                else:
                    if tok_start_position < doc_start or tok_end_position > doc_end:
                        start_position = 0
                        end_position = 0
                        continue #alternative TODO
                    else:
                        doc_offset = len(query_tokens) + 2
                        start_position = tok_start_position - doc_start + doc_offset
                        end_position = tok_end_position - doc_start + doc_offset
                        # in case of exception
                        if start_position >= max_seq_length-1:
                            start_position = end_position = 0
                            logger.warning("exception")
                        elif end_position >= max_seq_length-1:
                            start_position = end_position = 0 #max_seq_length - 2
                            logger.warning("exception")

            if example_index == 300:
                logger.info("*** Example ***")
                logger.info("unique_id: %s" % (unique_id))
                logger.info("example_index: %s" % (example_index))
                logger.info("doc_span_index: %s" % (doc_span_index))
                logger.info("tokens: %s" % " ".join([tokenization.printable_text(x) for x in tokens]))
                logger.info("token_to_orig_map: %s" % " ".join(["%d:%d" % (x, y) for (x, y) in six.iteritems(token_to_orig_map)]))
                logger.info("token_is_max_context: %s" % " ".join(["%d:%s" % (x, y) for (x, y) in six.iteritems(token_is_max_context)]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                if is_training:
                    answer_text = " ".join(tokens[start_position:(end_position + 1)]) if not example.is_impossible else ""
                    logger.info("start_position: %d" % (start_position))
                    logger.info("is_impossible: %s" % (example.is_impossible))
                    logger.info("end_position: %d" % (end_position))
                    logger.info(
                        "answer: %s" % (tokenization.printable_text(answer_text)))

            features.append(
                InputFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context, #ent = ent, pos = pos,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    doc_mask = doc_mask,
                    input_span_mask = input_span_mask,
                    segment_ids=segment_ids,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible= start_position==0))  # use_cls
            unique_id += 1

            num_all += 1
            num_na += int(start_position==0)
    logger.info(f"Num all: {num_all}")
    logger.info(f"Num na : {num_na}")
    return features


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index

def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                            orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size, offset=0):
    """Get the n-best logits from a list."""
    sorted_indices = np.argsort(logits)[::-1] + offset
    return list(sorted_indices[:n_best_size])


def softmax2d(scores):
    if not scores:
      return []
    z = np.array(scores)
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]  # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]  # dito
    return e_x / div

def softmax1d(scores):
  if not scores:
    return []
  z = np.array(scores)
  e_x = np.exp(z - np.max(z))
  div = np.sum(e_x)
  return e_x / div

def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs

def log_softmax1d(scores):
    if not scores:
        return []
    x = np.array(scores)
    z = logsumexp(x)
    return x-z

def log_sigmoid(score):
    return math.log(1/(1+math.exp(-score)))

RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits", "cls_logits"])


def write_predictions_google(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file):
    """Write final predictions to the json file."""
    logger.info("Writing predictions to: %s" % (output_prediction_file))
    #logger.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0
        null_ls = 0
        #null_end_logit = 0
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            result_start_ls = log_softmax1d(result.start_logits)
            result_end_ls   = log_softmax1d(result.end_logits)
            start_indexes = _get_best_indexes(result_start_ls, n_best_size)
            end_indexes   = _get_best_indexes(result_end_ls, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant

            #feature_null_score =   log_sigmoid(result.cls_logits) #result.start_logits[0] + result.end_logits[0]
            #feature_HasAns_score = log_sigmoid(-result.cls_logits)
            span_start_ls = result_start_ls #+ feature_HasAns_score
            span_end_ls   = result_end_ls #+ feature_HasAns_score

            #if feature_null_score < score_null:
            #    score_null = feature_null_score
            #    min_null_feature_index = feature_index
            #    null_ls = feature_null_score #result.start_logits[0]
            #    #null_end_logit = result.end_logits[0]


            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=span_start_ls[start_index],
                            end_logit=span_end_ls[end_index]))


        #if FLAGS.version_2_with_negative:
        #prelim_predictions.append(
        #    _PrelimPrediction(
        #        feature_index=min_null_feature_index,
        #        start_index=0,
        #        end_index=0,
        #        start_logit=null_ls/2,
        #        end_logit=null_ls/2))


        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case)
                final_text = final_text.replace(' ','')
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))

        # if we didn't inlude the empty option in the n-best, inlcude it

        #if FLAGS.version_2_with_negative:
        #if "" not in seen_predictions:
        #    nbest.append(
        #        _NbestPrediction(
        #            text="",
        #            start_logit=null_ls/2,
        #            end_logit=null_ls/2))


        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        #index_best_non_null_entry = None
        for (i,entry) in enumerate(nbest):
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    #index_best_non_null_entry = i
                    best_non_null_entry = entry

        probs = np.exp(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1


        #if not FLAGS.version_2_with_negative:
        #    all_predictions[example.qas_id] = nbest_json[0]["text"]
        #else:
            # predict "" iff the null score - the score of best non-null > threshold

        #if best_non_null_entry is None:
        #    score_diff = 999999
        #else:
        #    score_diff = np.exp(score_null) - np.exp((best_non_null_entry.start_logit + best_non_null_entry.end_logit))
        #scores_diff_json[example.qas_id] = float(score_diff)
        ##scores_diff_json[example.qas_id] = float(np.exp(score_null))
        #if score_diff > config.args.null_score_diff_threshold:
        #    all_predictions[example.qas_id] = ""
        #else:
        #    all_predictions[example.qas_id] = best_non_null_entry.text

        all_predictions[example.qas_id] = nbest_json[0]["text"]
        all_nbest_json[example.qas_id] = nbest_json

    with open(output_prediction_file, "w",encoding='utf-8') as writer:
        writer.write(json.dumps(all_predictions, indent=4,ensure_ascii=False) + "\n")

    #with open(output_nbest_file, "w") as writer:
    #    writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    #with open(output_null_log_odds_file,"w") as writer:
    #    writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    return all_predictions, scores_diff_json
