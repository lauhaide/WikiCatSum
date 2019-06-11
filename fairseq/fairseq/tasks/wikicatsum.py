# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import os
import numpy as np

from fairseq import options
from fairseq.data import (
    data_utils, DictionaryWCS, WikiCatSumPairDataset, IndexedInMemoryDatasetStruct, DictionaryKW,
    IndexedRawTextDataset, WikiCatSumPairDatasetTopics, IndexedInMemoryDatasetStructKeys, RawTextDataset,
    IndexedInMemoryDataset, WikiCatSumPairDatasetTopicsFlatdec, Dictionary, WikiCatSumPairDatasetTopicsFlatEnc,
)

from . import FairseqTask, register_task


@register_task('wikicatsum')
class WikiCatSum(FairseqTask):

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser.
        """
        parser.add_argument('data', metavar='DIR', help='path to data directory')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language')
        parser.add_argument('--raw-text', action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left (default: True)')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left (default: False)')
        parser.add_argument('--max-source-positions', default=25, type=int, metavar='N', #80 #nb of paragraphs
                            help='max number of sentences/paragraphs in the source sequence')
        parser.add_argument('--max-target-positions', default=15, type=int, metavar='N',
                            help='max number of sentences in the target sequence')
        parser.add_argument('--max-src-sentence-length', default=202, type=int, metavar='N', #paragraph length
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-tgt-sentence-length', default=42, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--annotations', action='store_true',
                            help='load annotations associated to source/target files if they exist.')
        parser.add_argument('--num-topics', type=int, metavar='N',
                            help='number of topics (keys) for the dataset', required=False)
        parser.add_argument('--keywords-embed-path', type=str,
                            help='path to embeddings of topic words')
        parser.add_argument('--prefix', type=int, default=0,
                            help='decoder with prefix feeding', required=False)
        parser.add_argument('--target-raw-text', action='store_true',
                            help='load raw text dataset for targets only')
        parser.add_argument('--flatdec', action='store_true',
                            help='use a flat decoder module')
        parser.add_argument('--flatenc', action='store_true',
                            help='use a flat encoder module')
        parser.add_argument('--dechatt', action='store_true',
                            help='in hierarchical input says whether decoder uses hierarchical attention or not (used in test)')
        parser.add_argument('--flatten-source', action='store_true',
                            help='use a flat encoder module') #deprecated
        parser.add_argument('--coverage', action='store_true', help='use coverage on doc-level atte (see et al.)')
        parser.add_argument('--flatdata', metavar='DIR', help='path to data directory for if using --flatdec or --flatenc options')



    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.flatdec = args.flatdec
        self.flatenc = args.flatenc
        
        self.keywords_dict = None

    @classmethod
    def setup_task(cls, args, **kwargs):
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)

        #could remove the following .....
        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(args.data)
        if args.source_lang is None or args.target_lang is None:
            raise Exception('Could not infer language pair, please provide it explicitly')

        # load dictionaries
        if args.flatenc or args.flatdec:
            flatData = args.flatdata
        if args.flatenc:
            flatFile = os.path.join(flatData, 'dict.{}.txt'.format(args.source_lang))
            print("For flat encoder load dictionary: ",flatFile)
            src_dict = Dictionary.load(flatFile)
        else:
            src_dict = DictionaryWCS.load(os.path.join(args.data, 'dict.{}.txt'.format(args.source_lang)))
        if args.flatdec:
            flatFile = os.path.join(flatData, 'dict.{}.txt'.format(args.target_lang))
            print("For flat decoder load dictionary: ",flatFile)
            tgt_dict = Dictionary.load(flatFile)
        else:
            tgt_dict = DictionaryWCS.load(os.path.join(args.data, 'dict.{}.txt'.format(args.target_lang)))
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        print('| [{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
        print('| [{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))

        return cls(args, src_dict, tgt_dict)

    def load_dataset(self, split):
        """Load a dataset split."""

        def split_exists(src, tgt, lang):
            filename = os.path.join(self.args.data, '{}.{}-{}.{}'.format(split, src, tgt, lang))
            if self.args.raw_text and IndexedRawTextDataset.exists(filename):
                return True
            elif not self.args.raw_text and IndexedInMemoryDatasetStruct.exists(filename):
                return True
            return False

        def keysExists(filename, info):
            keysFile = filename.replace(split, split+info)
            if IndexedInMemoryDatasetStruct.exists(keysFile):
                return True
            return False

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang
        if split_exists(src, tgt, src):
            prefix = os.path.join(self.args.data, '{}.{}-{}.'.format(split, src, tgt))
        elif split_exists(tgt, src, src):
            prefix = os.path.join(self.args.data, '{}.{}-{}.'.format(split, tgt, src))
        else:
            raise FileNotFoundError('Dataset not found: {} ({})'.format(split, self.args.data))

        def indexed_dataset(path, dictionary):
            if self.args.raw_text:
                return IndexedRawTextDataset(path, dictionary)
            elif IndexedInMemoryDatasetStruct.exists(path):
                return IndexedInMemoryDatasetStruct(path, fix_lua_indexing=True)
            return None
        def indexed_dataset_flat(path, dictionary):
            if self.args.raw_text:
                return IndexedRawTextDataset(path, dictionary)
            elif IndexedInMemoryDataset.exists(path):
                return IndexedInMemoryDataset(path, fix_lua_indexing=True)
            return None
        def indexed_keys_dataset(path):
            keyspath = path.replace(split, split + "_keys")
            if IndexedInMemoryDatasetStructKeys.exists(keyspath):
                return IndexedInMemoryDatasetStructKeys(keyspath, fix_lua_indexing=True)
            print("should returned")
            return None
        def indexed_keywords_dataset(path):
            keyspath = path.replace(split, split + "_keywords")
            if IndexedInMemoryDatasetStruct.exists(keyspath):
                return IndexedInMemoryDatasetStruct(keyspath, fix_lua_indexing=True)
            print("should returned")
            return None
        import re
        if self.flatenc or self.flatdec:
            flatPrefix = prefix
            flatPrefix = re.sub("HSA([0-9])*", "SS", flatPrefix)
            print("Flat Format Prefix Dataset: "+flatPrefix)
        if self.flatenc:
            flatFile = flatPrefix + src
            print("For flat encoder load dataset: ",flatFile)
            src_dataset = indexed_dataset_flat(flatFile, self.src_dict)
        else:
            src_dataset = indexed_dataset(prefix + src, self.src_dict)
        if self.flatdec:
            flatFile = flatPrefix + tgt
            print("For flat decoder load dataset: ", flatFile)
            tgt_dataset = indexed_dataset_flat(flatFile, self.tgt_dict)
        else:
            tgt_dataset = indexed_dataset(prefix + tgt, self.tgt_dict)
        ori_tgt = None
        ori_src = None
        if self.args.target_raw_text:  ##change this argument to give directory of the file, not need to have original texts in compiled directories
            file = os.path.join(self.args.data , "{}.{}".format(split, tgt))
            print(file)
            if os.path.isfile(file):
                ori_tgt = RawTextDataset(file)

        # Read keys if indicated and they exist
        src_dataset_keys = None
        tgt_dataset_keys = None
        if self.args.annotations and keysExists(prefix + src, "_keys"):
            src_dataset_keys = indexed_keys_dataset(prefix + src)
        if self.args.annotations and keysExists(prefix + tgt, "_keys"):
            tgt_dataset_keys = indexed_keys_dataset(prefix + tgt)

        # Read keywords if indicated and they exist
        src_dataset_keywords = None
        tgt_dataset_keywords = None
        if self.args.annotations and keysExists(prefix + src, "_keywords"):
            src_dataset_keywords = indexed_keywords_dataset(prefix + src)
        if self.args.annotations and keysExists(prefix + tgt, "_keywords"):
            tgt_dataset_keywords = indexed_keywords_dataset(prefix + tgt)

        # load keywords vocabulary SPLIT_keyVocab.txt
        kvoc = os.path.join(self.args.data, split + "_keyVocab.txt")
        if self.args.annotations and os.path.exists(kvoc):
            self.keywords_dict = DictionaryKW.load(kvoc)


        #up to here each target and source item is not padded in any of the 2D dimensions!!!
        #recover 2D size of each element in the dataset
        if not self.flatenc:
            src_dataset_sizes = np.reshape(src_dataset.sizes,(-1,2))
        if not self.flatdec:
            tgt_dataset_sizes = np.reshape(tgt_dataset.sizes,(-1,2))

        if self.flatdec:
            self.datasets[split] = WikiCatSumPairDatasetTopicsFlatdec(
                None, None,
                src_dataset, src_dataset_sizes, self.src_dict,
                src_txt=ori_src, tgt_txt=ori_tgt,
                tgt=tgt_dataset, tgt_sizes=tgt_dataset.sizes, tgt_dict=self.tgt_dict,
                keywords_dict=self.keywords_dict,
                src_dataset_keywords=None,
                tgt_dataset_keywords=None,
                left_pad_source=self.args.left_pad_source,
                left_pad_target=self.args.left_pad_target,
                max_source_positions=self.args.max_source_positions,
                max_target_positions=self.args.max_target_positions,
                max_src_sentence_length=self.args.max_src_sentence_length,
                max_tgt_sentence_length=self.args.max_tgt_sentence_length, #shouldn be None ????
                kdim=self.args.num_topics,
                use_prefix=self.args.prefix,
            )
        elif self.flatenc:
            self.datasets[split] = WikiCatSumPairDatasetTopicsFlatEnc(
                None, tgt_dataset_keys,
                src_dataset, src_dataset.sizes, self.src_dict,
                src_txt=ori_src, tgt_txt=ori_tgt,
                tgt=tgt_dataset, tgt_sizes=tgt_dataset_sizes, tgt_dict=self.tgt_dict,
                keywords_dict=self.keywords_dict,
                src_dataset_keywords=src_dataset_keywords,
                tgt_dataset_keywords=None,
                left_pad_source=self.args.left_pad_source,
                left_pad_target=self.args.left_pad_target,
                max_source_positions=self.args.max_source_positions,
                max_target_positions=self.args.max_target_positions,
                max_tgt_sentence_length=self.args.max_tgt_sentence_length,
                kdim=self.args.num_topics,
                use_prefix=self.args.prefix,
            )
        elif src_dataset_keys is None:
            self.datasets[split] = WikiCatSumPairDataset(
                src_dataset, src_dataset_sizes, self.src_dict,
                src_txt=ori_src, tgt_txt=ori_tgt,
                tgt=tgt_dataset, tgt_sizes=tgt_dataset_sizes, tgt_dict=self.tgt_dict,
                left_pad_source=self.args.left_pad_source,
                left_pad_target=self.args.left_pad_target,
                max_source_positions=self.args.max_source_positions,
                max_target_positions=self.args.max_target_positions,
                max_src_sentence_length=self.args.max_src_sentence_length,
                max_tgt_sentence_length=self.args.max_tgt_sentence_length,
                use_prefix=self.args.prefix,
                flattenSource=self.args.flatten_source,
            )
        else:
            print("*********************** "+split)
            self.datasets[split] = WikiCatSumPairDatasetTopics(
                src_dataset_keys, tgt_dataset_keys,
                src_dataset, src_dataset_sizes, self.src_dict,
                src_txt=ori_src, tgt_txt=ori_tgt,
                tgt=tgt_dataset, tgt_sizes=tgt_dataset_sizes, tgt_dict=self.tgt_dict,
                keywords_dict=self.keywords_dict,
                src_dataset_keywords=src_dataset_keywords,
                tgt_dataset_keywords=tgt_dataset_keywords,
                left_pad_source=self.args.left_pad_source,
                left_pad_target=self.args.left_pad_target,
                max_source_positions=self.args.max_source_positions,
                max_target_positions=self.args.max_target_positions,
                max_src_sentence_length=self.args.max_src_sentence_length,
                max_tgt_sentence_length=self.args.max_tgt_sentence_length,
                kdim=self.args.num_topics,
                use_prefix=self.args.prefix,
            )

    @property
    def source_dictionary(self):
        return self.src_dict

    @property
    def target_dictionary(self):
        return self.tgt_dict

    @property
    def keywords_dictionary(self):
        return self.keywords_dict
