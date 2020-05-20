'''
This file was based on QuASE code run_classifier.py
'''


# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""BERT fine-tuning runner."""

import csv
import os
import logging
import argparse
import random
from tqdm import tqdm, trange
from sklearn import metrics
import process_data

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
#from modeling import BertForSequenceClassification
from apply_models import BertForPredicateClassification
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, predicate_vector=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
            predicate: (Optional) string. The binary vector indicating the location
            of the predicate word in a sentence.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.predicate_vector = predicate_vector

    def __str__(self):
        return 'guid: %s\ttext_a: %s\ttext_b: %s\tlabel: %s\tpredicate: %s' % (self.guid, self.text_a, self.text_b,
                                                                               self.label, self.predicate_vector)

    def get_debug_string(self):
        return 'guid: %s\ttext_a: %s\ttext_b: %s\tlabel: %s' % (self.guid, self.text_a, self.text_b,
                                                                               self.label)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, predicate_vector):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.predicate_vector = predicate_vector

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    # print(outputs)
    return np.sum(outputs == labels)

def get_labels_from_file(path):
    if path:
        with open(path, "r") as f:
            labels = f.read().splitlines()
        return labels
    else:
        return []


def read_examples_from_file(data_dir, mode=None, has_label=True):

    file_path = os.path.join(data_dir, "{}.txt".format(mode))

    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f:
        words = []
        # labels = []
        label = None
        predicate = None
        predicate_vector = []
        num_sentences = 0
        for line in f:
            line = line.strip()
            if line == "":
                if words:
                    text_a = " ".join(words)
                    examples.append(InputExample(guid="{}-{}".format(mode, guid_index), text_a=text_a, text_b=predicate,
                                                 label=label, predicate_vector=predicate_vector))
                    guid_index += 1
                    words = []
                    num_sentences += 1
                    predicate_vector = []
            else:
                splits = line.split(" ")
                words.append(splits[0])
                if splits[1] != 'O':
                    if has_label:
                        label = splits[1]
                    predicate = splits[0]
                    predicate_vector.append(1)
                else:
                    predicate_vector.append(0)


    print('Total number of examples loaded: ', num_sentences)
    return examples


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    # @classmethod
    # def _read_txt(cls, input_file, quotechar=None):
    #     """Reads a tab separated value file."""
    #     with open(input_file, "r", encoding='utf-8') as f:
    #         reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
    #         lines = []
    #         for line in reader:
    #             lines.append(line)
    #         return lines


class PcProcessor(DataProcessor):
    """Processor for the predicate classification data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.txt")))
        return read_examples_from_file(data_dir, 'train')

    def get_dev_examples(self, data_dir):
        """See base class."""
        return read_examples_from_file(data_dir, 'dev')

    def get_test_examples_without_label(self, data_dir):
        return read_examples_from_file(data_dir, 'test_predictions', has_label=False)

    def get_test_examples(self, data_dir):
        return read_examples_from_file(data_dir, 'test')

    def get_arab_test_examples(self, data_dir):
        return read_examples_from_file(data_dir, 'test_arab')

    def get_examples_custom(self, data_dir, example_file, has_label):
        return read_examples_from_file(data_dir, example_file, has_label)

    def get_labels(self):
        """See base class."""
        return get_labels_from_file('./data/predicate-classification/labels.txt')
        # return get_labels_from_file('./data/second_quad/labels.txt')
        # return get_labels_from_file('./data/first_quad/labels.txt')


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}
    print(label_map)

    features = []
    for (ex_index, example) in enumerate(examples):
        predicate_vector = example.predicate_vector
        text_a_ls = example.text_a.split(' ')
        assert len(predicate_vector) == len(text_a_ls), 'length of predicate vector is not the same as length of text_a'

        predicate_vector_tokenized = []
        tokens_temp = []
        for i, word in enumerate(text_a_ls):
            word_token = tokenizer.tokenize(word)
            if predicate_vector[i] == 1:
                predicate_vector_tokenized.extend([1] * len(word_token))
            else:
                predicate_vector_tokenized.extend([0] * len(word_token))
            tokens_temp.extend(word_token)

        # tokenize the whole sentence
        tokens_a = tokenizer.tokenize(example.text_a)

        assert tokens_temp == tokens_a, 'Tokenization is incorrectly done.'

        tokens_b = None
        if example.text_b:
            # tokenize the whole sentence
            tokens_b = tokenizer.tokenize(example.text_b)


            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        predicate_vector_tokenized = [0] + predicate_vector_tokenized + [0]

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)
            predicate_vector_tokenized.extend([0] * (len(tokens_b) + 1))

        # print('tokens_a: ', tokens_a)
        # print('tokens_b: ', tokens_b)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # pad the predicate vector to match the length of input_ids
        predicate_vector_tokenized += [0] * (len(input_ids) - len(predicate_vector_tokenized))

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        predicate_vector_tokenized += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(predicate_vector_tokenized) == max_seq_length, "left: %s\tright: %s" % (len(predicate_vector_tokenized), max_seq_length)
        # if predicate_vector_tokenized.count(1) > 1:
        #     print('tokens:', tokens)
        #     print('predicate vector: ', predicate_vector_tokenized)

        if example.label is not None:
            label_id = label_map[example.label]
        else:
            label_id = None
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            if example.label is not None:
                logger.info("label: %s (id = %d)" % (example.label, label_id))
            logger.info("predicate vector: %s" % " ".join([str(x) for x in predicate_vector_tokenized]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          predicate_vector=predicate_vector_tokenized))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default='saved_models',
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--eval_while_training",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict",
                        action='store_true',
                        help="Whether output prediction file for the test set.")
    parser.add_argument("--load_pretrained",
                        action='store_true',
                        help="Whether to load a pretrained model or start with a new one.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--pretrain',
                        action='store_true',
                        help="Whether to load a pre-trained model for continuing training")
    parser.add_argument('--pretrained_model_file',
                        type=str,
                        help="The path of the pretrained_model_file")
    parser.add_argument("--percent",
                        default=100,
                        type=int,
                        help="The percentage of examples used in the training data.\n")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--use_predicate_indicator',
                        action='store_true',
                        help="Whether to use predicate position indicator as input.")
    parser.add_argument('--detector_prediction_file',
                        type=str,
                        help="The path of the detector_prediction_file")


    args = parser.parse_args()

    processors = {
        "pc": PcProcessor
    }

    num_labels_task = {
        "pc": 12
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval and not args.do_predict:
        raise ValueError("At least one of `do_train`, `do_eval` or `do_predict` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    num_labels = num_labels_task[task_name]
    label_list = processor.get_labels()

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    if args.do_train:

        train_examples = None
        num_train_steps = None
        if args.do_train:
            train_examples = processor.get_train_examples(args.data_dir)
            print('number of examples: ', len(train_examples))
            train_examples = train_examples[:int(len(train_examples) * args.percent / 100)]
            num_train_steps = int(
                len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

        # Prepare model
        model = BertForPredicateClassification.from_pretrained(args.bert_model,
                                                              cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(
                                                                  args.local_rank),
                                                              num_labels=num_labels)

        if args.pretrain:
            # Load a pre-trained model
            print('load a pre-trained model from ' + args.pretrained_model_file)
            model_state_dict = torch.load(args.pretrained_model_file)
            model = BertForPredicateClassification.from_pretrained(args.bert_model, state_dict=model_state_dict,
                                                                  num_labels=num_labels)
            model.to(device)

        if args.fp16:
            model.half()
        model.to(device)
        if args.local_rank != -1:
            try:
                from apex.parallel import DistributedDataParallel as DDP
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            model = DDP(model)
        elif n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Prepare optimizer
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        t_total = num_train_steps
        if args.local_rank != -1:
            t_total = t_total // torch.distributed.get_world_size()
        if args.fp16:
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=args.learning_rate,
                                  bias_correction=False,
                                  max_grad_norm=1.0)
            if args.loss_scale == 0:
                optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

        else:
            optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=args.learning_rate,
                                 warmup=args.warmup_proportion,
                                 t_total=t_total)

        global_step = 0
        nb_tr_steps = 0
        tr_loss = 0

        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_predicate_vectors = torch.tensor([f.predicate_vector for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_predicate_vectors)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        if args.do_train:
            train_features = convert_examples_to_features(
                train_examples, label_list, args.max_seq_length, tokenizer)

            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", len(train_examples))
            logger.info("  Batch size = %d", args.train_batch_size)
            logger.info("  Num steps = %d", num_train_steps)
            all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
            all_predicate_vectors = torch.tensor([f.predicate_vector for f in train_features], dtype=torch.long)
            train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_predicate_vectors)
            if args.local_rank == -1:
                train_sampler = RandomSampler(train_data)
            else:
                train_sampler = DistributedSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

            for epc in trange(int(args.num_train_epochs), desc="Epoch"):
                model.train()
                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0

                y_true = []
                y_pred = []

                for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, input_mask, segment_ids, label_ids, predicate_vectors = batch
                    # print(input_ids, input_mask, segment_ids, label_ids)
                    if args.use_predicate_indicator:
                        loss, logits = model(input_ids, segment_ids, input_mask, label_ids, predicate_vectors)
                    else:
                        loss, logits = model(input_ids, segment_ids, input_mask, label_ids)
                    if n_gpu > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu.
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    if args.fp16:
                        optimizer.backward(loss)
                    else:
                        loss.backward()

                    logits = logits.detach().cpu().numpy()
                    label_ids = label_ids.to('cpu').numpy()

                    y_pred.extend(np.argmax(logits, axis=1).tolist())
                    y_true.extend(label_ids.tolist())

                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        # modify learning rate with special warm up BERT uses
                        lr_this_step = args.learning_rate * warmup_linear(global_step / t_total, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                        optimizer.step()
                        optimizer.zero_grad()
                        global_step += 1

                another_acc = metrics.accuracy_score(y_true, y_pred)
                f1_score = metrics.f1_score(y_true, y_pred, average='macro')

                result = {'another_acc:': another_acc,
                          'f1 score:': f1_score}

                logger.info("***** Train results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))

                if args.eval_while_training and (args.local_rank == -1 or torch.distributed.get_rank() == 0):

                    model.eval()
                    eval_loss, eval_accuracy = 0, 0
                    nb_eval_steps, nb_eval_examples = 0, 0

                    y_true = []
                    y_pred = []

                    for input_ids, input_mask, segment_ids, label_ids, predicate_vectors in tqdm(eval_dataloader, desc="Evaluating"):
                        input_ids = input_ids.to(device)
                        input_mask = input_mask.to(device)
                        segment_ids = segment_ids.to(device)
                        label_ids = label_ids.to(device)
                        predicate_vectors = predicate_vectors.to(device)

                        with torch.no_grad():

                            if args.use_predicate_indicator:
                                tmp_eval_loss, _ = model(input_ids, segment_ids, input_mask, label_ids,
                                                         predicate_vectors)
                                logits = model(input_ids, segment_ids, input_mask, predicate_vector=predicate_vectors)
                            else:
                                tmp_eval_loss, _ = model(input_ids, segment_ids, input_mask, label_ids)
                                logits = model(input_ids, segment_ids, input_mask)

                        logits = logits.detach().cpu().numpy()
                        label_ids = label_ids.to('cpu').numpy()

                        y_pred.extend(np.argmax(logits, axis=1).tolist())
                        y_true.extend(label_ids.tolist())

                        tmp_eval_accuracy = accuracy(logits, label_ids)
                        eval_loss += tmp_eval_loss.mean().item()
                        eval_accuracy += tmp_eval_accuracy

                        nb_eval_examples += input_ids.size(0)
                        nb_eval_steps += 1

                    another_acc = metrics.accuracy_score(y_true, y_pred)
                    f1_score = metrics.f1_score(y_true, y_pred, average='macro')

                    eval_loss = eval_loss / nb_eval_steps
                    eval_accuracy = eval_accuracy / nb_eval_examples
                    loss = tr_loss / nb_tr_steps if args.do_train else None
                    result = {'eval_loss': eval_loss,
                              'eval_accuracy': eval_accuracy,
                              'global_step': global_step,
                              'loss': loss,
                              'another_acc:': another_acc,
                              'f1 score:': f1_score}

                    logger.info("***** Eval results *****")
                    for key in sorted(result.keys()):
                        logger.info("  %s = %s", key, str(result[key]))

                # save model periodically
                # if (epc+1) % 10 == 0:
                #     model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                #     output_model_file = os.path.join(args.output_dir, "pytorch_model_" + str(epc+1) + ".bin")
                #     if args.do_train:
                #         torch.save(model_to_save.state_dict(), output_model_file)

        # Save a trained model
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
        if args.do_train:
            torch.save(model_to_save.state_dict(), output_model_file)

        # Until here

    # Load a trained model that you have fine-tuned
    output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
    model_state_dict = torch.load(output_model_file)
    model = BertForPredicateClassification.from_pretrained(args.bert_model, state_dict=model_state_dict,
                                                          num_labels=num_labels)
    model.to(device)

    if args.do_predict and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Preprocess the detector output
        detector_prediction_file = args.detector_prediction_file
        #detector_prediction_file = 'detector_prediction/detector_predictions.txt'
        adjusted_detector_prediction_file = os.path.join(args.output_dir, 'adjusted_detector_predictions.txt')
        process_data.write_pc_data(mode='else', detector_out=detector_prediction_file,
                                   adjusted_detector_out=adjusted_detector_prediction_file)

        test_examples = processor.get_examples_custom(args.output_dir,
                                                      example_file='adjusted_detector_predictions',
                                                      has_label=False)
        test_features = convert_examples_to_features(
            test_examples, label_list, args.max_seq_length, tokenizer
        )

        logger.info("***** Running Prediction *****")
        logger.info("  Num examples = %d", len(test_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
        all_predicate_vectors = torch.tensor([f.predicate_vector for f in test_features], dtype=torch.long)
        test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_predicate_vectors)
        # Run prediction for full data
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)

        model.eval()

        y_pred = []

        for input_ids, input_mask, segment_ids, predicate_vectors in tqdm(test_dataloader, desc="Predicting"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            predicate_vectors = predicate_vectors.to(device)

            with torch.no_grad():
                if args.use_predicate_indicator:
                    logits = model(input_ids, segment_ids, input_mask, predicate_vector=predicate_vectors)
                else:
                    logits = model(input_ids, segment_ids, input_mask)

            logits = logits.detach().cpu().numpy()

            y_pred.extend(np.argmax(logits, axis=1).tolist())

        label_map = {i: label for i, label in enumerate(label_list)}
        y_pred_word = [label_map[lab] for lab in y_pred]
        classification_output = os.path.join(args.output_dir, 'classification_predictions.txt')
        with open(classification_output, 'w') as out_file:
            for w in y_pred_word:
                out_file.write(w+'\n')

        # classification_prediction_file = open(classification_output, 'r')
        #
        # classification_preds = []
        # for line in classification_prediction_file:
        #     line = line.strip()
        #     if line:
        #         classification_preds.append(line)
        #
        # classification_prediction_file.close()
        classification_preds = y_pred_word
        i = 0
        final_prediction_file = os.path.join(args.output_dir, 'final_predictions.txt')
        out_file = open(final_prediction_file, 'w')
        detection_prediction_file = open(detector_prediction_file, 'r')

        for line in detection_prediction_file:
            line = line.strip()
            if line:
                word, tag = line.split()
                if tag != 'O':
                    out_file.write('%s %s\n' % (word, classification_preds[i]))
                    i += 1
                else:
                    out_file.write('%s %s\n' % (word, tag))
            else:
                out_file.write('\n')

        detection_prediction_file.close()
        out_file.close()


    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = processor.get_arab_test_examples(args.data_dir)
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_predicate_vectors = torch.tensor([f.predicate_vector for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_predicate_vectors)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        y_true = []
        y_pred = []

        for input_ids, input_mask, segment_ids, label_ids, predicate_vectors in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)
            predicate_vectors = predicate_vectors.to(device)

            with torch.no_grad():
                if args.use_predicate_indicator:
                    tmp_eval_loss, _ = model(input_ids, segment_ids, input_mask, label_ids, predicate_vectors)
                    logits = model(input_ids, segment_ids, input_mask, predicate_vector=predicate_vectors)
                else:
                    tmp_eval_loss, _ = model(input_ids, segment_ids, input_mask, label_ids)
                    logits = model(input_ids, segment_ids, input_mask)

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()

            y_pred.extend(np.argmax(logits, axis=1).tolist())
            y_true.extend(label_ids.tolist())

            tmp_eval_accuracy = accuracy(logits, label_ids)
            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        another_acc = metrics.accuracy_score(y_true, y_pred)
        f1_score = metrics.f1_score(y_true, y_pred, average='macro')

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples
        # loss = tr_loss / nb_tr_steps if args.do_train else None
        result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy,
                  # 'global_step': global_step,
                  # 'loss': loss,
                  'another_acc:': another_acc,
                  'f1 score:': f1_score}

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        label_map = {i: label for i, label in enumerate(label_list)}
        y_true_word = [label_map[lab] for lab in y_true]
        y_pred_word = [label_map[lab] for lab in y_pred]

        wrong_pred_idx = np.array(y_true) != np.array(y_pred)
        rs = []
        for i, wrong in enumerate(wrong_pred_idx):
            if wrong:
                rs.append(eval_examples[i])
        print('wrong predictions total number: ', len(rs))
        wrong_file = open(os.path.join(args.output_dir, 'wrong_predictions.txt'), 'w')

        for i, entry in enumerate(rs):
            assert y_true_word[int(entry.guid.split('-')[-1])-1] == entry.label, \
                '#%s# len(%d) is not equal to #%s# len(%d)' % (y_true_word[int(entry.guid.split('-')[-1])],
                                                               len(y_true_word[int(entry.guid.split('-')[-1])]),
                                                               entry.label,
                                                               len(entry.label))
            wrong_file.write(entry.get_debug_string())
            wrong_file.write('\tpredicted: %s' % (y_pred_word[i]))
            wrong_file.write('\n')

        wrong_file.close()

        conf_mat = metrics.confusion_matrix(y_true_word, y_pred_word, labels=label_list)
        # print_dict(label_map)
        print_cm(conf_mat, label_list)

def print_dict(label_dict):
    total = sum(label_dict.keys())

    for label, count in sorted(label_dict.items(), key=lambda x:x[0]):
        print('%s : %d, percentage: %f' % (label, count, count / float(total)))


def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth

    # Begin CHANGES
    fst_empty_cell = (columnwidth - 3) // 2 * " " + "t/p" + (columnwidth - 3) // 2 * " "

    if len(fst_empty_cell) < len(empty_cell):
        fst_empty_cell = " " * (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell
    # Print header
    print("    " + fst_empty_cell, end=" ")
    # End CHANGES

    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")

    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()

if __name__ == "__main__":
    main()