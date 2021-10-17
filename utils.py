# -*- coding: utf-8 -*-
import collections
import math
from datasets import load_metric
import numpy as np
import logging
from tqdm import tqdm, trange
import torch
from transformers import AutoTokenizer, AutoConfig, BertConfig, BertTokenizer, BasicTokenizer
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor, squad_convert_examples_to_features, SquadExample
from transformers import squad_convert_examples_to_features


def train_enc(qas_ids, questions, titles, contexts, answers, is_impossibles, s_poses, tokenizer, args):
    datas = []
    for i in range(len(qas_ids)):
        data = SquadExample(qas_id=qas_ids[i], question_text=questions[i], context_text=contexts[i], title=titles[i], answer_text=answers[i], answers=[{'answer_start':s_poses[i],'text':answers[i]}], is_impossible=is_impossibles[i], start_position_character=s_poses[i])
        datas.append(data)
    
    features = squad_convert_examples_to_features(
        examples=datas,
        tokenizer=tokenizer,
        max_seq_length= args.max_seq_length,
        doc_stride= args.doc_stride,
        max_query_length= args.max_query_length,
        is_training=True,
    )
    
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
#     all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    all_start_pos =  torch.tensor([f.start_position for f in features], dtype=torch.long)
    all_end_pos = torch.tensor([f.end_position for f in features], dtype=torch.long)
    
    train_enc = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_start_pos, all_end_pos)
    return train_enc

def valid_enc(qas_ids, questions, titles, contexts, answers, is_impossibles, s_poses, tokenizer, args):
    datas = []
    for i in range(len(qas_ids)):
        data = SquadExample(qas_id=qas_ids[i], question_text=questions[i], context_text=contexts[i], title=titles[i], answer_text=answers[i], answers=[{'answer_start':s_poses[i],'text':answers[i]}], is_impossible=is_impossibles[i], start_position_character=s_poses[i])
        datas.append(data)
    
    features = squad_convert_examples_to_features(
        examples=datas,
        tokenizer=tokenizer,
        max_seq_length= args.max_seq_length,
        doc_stride= args.doc_stride,
        max_query_length= args.max_query_length,
        is_training=True,
    )
    
    return features, datas

def test_enc(qas_ids, questions, titles, contexts, tokenizer, args):
    datas = []
    for i in range(len(qas_ids)):
        data = SquadExample(qas_id=qas_ids[i], question_text=questions[i], context_text=contexts[i], title=titles[i], answer_text="", answers=[{'answer_start':-1,'text':""}], is_impossible=False, start_position_character=-1)
        datas.append(data)
    
    features = squad_convert_examples_to_features(
        examples=datas,
        tokenizer=tokenizer,
        max_seq_length= args.max_seq_length,
        doc_stride= args.doc_stride,
        max_query_length= args.max_query_length,
        is_training=False,
    )
    
    return features, datas

RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])


def get_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, verbose_logging,
                      version_2_with_negative, null_score_diff_threshold):
    """Write final predictions to the json file and log-odds of null if needed."""

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = []
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min mull score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            if version_2_with_negative:
                feature_null_score = result.start_logits[0] + result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
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
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))
        if version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit))
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit", "start_index", "end_index", "qas_id"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                orig_id = feature.qas_id
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

                final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True
                orig_doc_start = -1
                orig_doc_end = -1
                orig_id = "None"

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit,
                    start_index = orig_doc_start,
                    end_index = orig_doc_end,
                    qas_id = orig_id))
        # if we didn't include the empty option in the n-best, include it
        if version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(
                    _NbestPrediction(
                        text="",
                        start_logit=null_start_logit,
                        end_logit=null_end_logit,
                        start_index = -1,
                        end_index = -1,
                        qas_id="None"))
                
            # In very rare edge cases we could only have single null prediction.
            # So we just create a nonce prediction in this case to avoid failure.
            if len(nbest)==1:
                nbest.insert(0,
                    _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0, start_index = -1, end_index = -1, qas_id="None"))
                
        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0, start_index = -1, end_index = -1, qas_id="None"))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            output["start_index"] = entry.start_index
            output["end_index"] = entry.end_index
            output["qas_id"] = entry.qas_id
            nbest_json.append(output)

        assert len(nbest_json) >= 1
        
        all_predictions.append(nbest_json[0])
    return all_predictions

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
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
#         if verbose_logging:
#             logger.info(
#                 "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
#         if verbose_logging:
#             logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
#                         orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
#         if verbose_logging:
#             logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
#         if verbose_logging:
#             logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


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

def compute_metrics(predict, text_label):
  metric = load_metric("squad")
  formatted_predictions = [{"id": item['qas_id'], "prediction_text": item['text']} for item in predict]
  # references = [{"id": ex["id"], "answers": ex["answers"]} for ex in datasets["validation"]]
  metric.compute(predictions=formatted_predictions, references=text_label)
  return metric.compute(predictions=formatted_predictions, references=text_label)

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return "%s (remain %s)" % (asMinutes(s), asMinutes(rs))

def init_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)