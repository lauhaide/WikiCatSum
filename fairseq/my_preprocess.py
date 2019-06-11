#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

import argparse
from itertools import zip_longest
import os
import shutil
import numpy as np

from fairseq.data import indexed_dataset, dictionaryWCS, dictionary
from fairseq.tokenizerWCSSentence import TokenizerWCSSentence
from fairseq.tokenizerWCSParagraph import TokenizerWCSParagraph
from fairseq.tokenizerWCSSingleSequence import TokenizerWCSSingleSequence
from fairseq.tokenizer import Tokenizer, tokenize_line
import fairseq.tokenizer as tokenizer
from fairseq.preprocess.GensimLDATopicDistAnnotator import GensimLDATopicDistAnnotator
import fairseq.preprocess.Constants

def get_parser():
    parser = argparse.ArgumentParser(
        description='Data pre-processing: Create dictionary and store data in binary format')
    parser.add_argument('-s', '--source-lang', default=None, metavar='SRC', help='source language')
    parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET', help='target language')
    parser.add_argument('--trainpref', metavar='FP', default=None, help='target language')
    parser.add_argument('--validpref', metavar='FP', default=None, help='comma separated, valid language prefixes')
    parser.add_argument('--testpref', metavar='FP', default=None, help='comma separated, test language prefixes')
    parser.add_argument('--destdir', metavar='DIR', default='data-bin', help='destination dir')
    parser.add_argument('--thresholdtgt', metavar='N', default=0, type=int,
                        help='map words appearing less than threshold times to unknown')
    parser.add_argument('--thresholdsrc', metavar='N', default=0, type=int,
                        help='map words appearing less than threshold times to unknown')
    parser.add_argument('--tgtdict', metavar='FP', help='reuse given target dictionary')
    parser.add_argument('--srcdict', metavar='FP', help='reuse given source dictionary')
    parser.add_argument('--nwordstgt', metavar='N', default=-1, type=int, help='number of target words to retain')
    parser.add_argument('--nwordssrc', metavar='N', default=-1, type=int, help='number of source words to retain')
    parser.add_argument('--alignfile', metavar='ALIGN', default=None, help='an alignment file (optional)')
    parser.add_argument('--output-format', metavar='FORMAT', default='binary', choices=['binary', 'raw'],
                        help='output format (optional)')
    parser.add_argument('--joined-dictionary', action='store_true', help='Generate joined dictionary')
    parser.add_argument('--only-source', action='store_true', help='Only process the source language')
    parser.add_argument('--padding-factor', metavar='N', default=8, type=int,
                        help='Pad dictionary size to be multiple of N')

    parser.add_argument('--singleSeq', action='store_true', help='Source and target are a single long sequence.')
    parser.add_argument('--src-chunk-length', metavar='N', default=200, type=int,
                        help='Maximum length of source text chunks when structured encoding (e.g. by paragraphs).')
    parser.add_argument('--tgt-chunk-length', metavar='N', default=40, type=int,
                        help='Maximum length of target text chunks (e.g. sentences).')
    parser.add_argument('--L', metavar='N', default=800, type=int,
                        help='Nb of tokens to take from the input')
    parser.add_argument('--addAnnotations', help='Path to LDA model.')
    parser.add_argument('--numTopics', type=int, help='Num of topics on LDA model.')
    parser.add_argument('--tgt-annotation', default='td', help='either topic distributions or 0/1 topic presence [td/mc]')

    return parser


def main(args):
    print(args)
    os.makedirs(args.destdir, exist_ok=True)
    target = not args.only_source

    def build_dictionary(filenames):
        if args.singleSeq:
            d = dictionary.Dictionary()
        else:
            d = dictionaryWCS.DictionaryWCS()
        for filename in filenames:
            tokenizer.Tokenizer.add_file_to_dictionary(filename, d, tokenize_line, args.L)

        return d

    def load_dictionary(filename):
        if args.singleSeq:
            d= dictionary.Dictionary.load(filename)
        else:
            d = dictionaryWCS.DictionaryWCS.load(filename)

        return d

    def train_path(lang):
        return '{}{}'.format(args.trainpref, ('.' + lang) if lang else '')

    def file_name(prefix, lang):
        fname = prefix
        if lang is not None:
            fname += f'.{lang}'
        return fname

    def dest_path(prefix, lang):
        return os.path.join(args.destdir, file_name(prefix, lang))

    def dict_path(lang):
        return dest_path('dict', lang) + '.txt'

    def dataset_dest_path(output_prefix, lang, extension):
        base = f'{args.destdir}/{output_prefix}'
        lang_part = f'.{args.source_lang}-{args.target_lang}.{lang}' if lang is not None else ''
        return f'{base}{lang_part}.{extension}'

    if args.joined_dictionary:
        assert not args.srcdict, 'cannot combine --srcdict and --joined-dictionary'
        assert not args.tgtdict, 'cannot combine --tgtdict and --joined-dictionary'
        src_dict = build_dictionary(set([
            train_path(lang)
            for lang in [args.source_lang, args.target_lang]
        ]))
        tgt_dict = src_dict
    else:
        if args.srcdict:
            src_dict = load_dictionary(args.srcdict)
        else:
            assert args.trainpref, "--trainpref must be set if --srcdict is not specified"
            src_dict = build_dictionary([train_path(args.source_lang)])
        if target:
            if args.tgtdict:
                tgt_dict = load_dictionary(args.tgtdict)
            else:
                assert args.trainpref, "--trainpref must be set if --tgtdict is not specified"
                tgt_dict = build_dictionary([train_path(args.target_lang)])

    src_dict.finalize(
        threshold=args.thresholdsrc,
        nwords=args.nwordssrc,
        padding_factor=args.padding_factor,
    )
    src_dict.save(dict_path(args.source_lang))
    if target:
        if not args.joined_dictionary:
            tgt_dict.finalize(
                threshold=args.thresholdtgt,
                nwords=args.nwordstgt,
                padding_factor=args.padding_factor,
            )
        tgt_dict.save(dict_path(args.target_lang))

    def make_binary_dataset(input_prefix, output_prefix, lang):
        #debugging, do only targets
        #if 'src' in lang:
        #if 'tgt' in lang:
        #    print("skip src files...")
        #    #print("skip tgt files...")
        #    return

        dict = load_dictionary(dict_path(lang))
        print('| [{}] Dictionary: {} types'.format(lang, len(dict) - 1))

        ds = indexed_dataset.IndexedDatasetBuilder(dataset_dest_path(output_prefix, lang, 'bin'))

        annotator = None
        aconsumer = None
        if args.addAnnotations:
            brigramFile = None
            if os.path.isfile(args.addAnnotations  + "_bigrams"):
                brigramFile = args.addAnnotations  + "_bigrams"
            if args.addAnnotations.endswith(".mallet"):
                annotator = MalletLDATopicDistAnnotator(args.addAnnotations, args.numTopics, brigramFile)
            else:
                annotator = GensimLDATopicDistAnnotator(args.addAnnotations, args.numTopics, brigramFile)
            #generate embeddings for topic keywords
            annotator.generateKeywordEmbeddings(dataset_dest_path(output_prefix+"_keyEmbeddings", None, 'txt'))
            annotator.generateKeywordDict(dataset_dest_path(output_prefix+"_keyVocab", None, 'txt'))
            ads = indexed_dataset.IndexedDatasetBuilder(
                dataset_dest_path(output_prefix+"_keys", lang, 'bin'),
                dtype=np.double)
            def aconsumer(tensor):
                ads.add_item(tensor)
            awds = indexed_dataset.IndexedDatasetBuilder(
                dataset_dest_path(output_prefix+"_keywords", lang, 'bin'))
            def awconsumer(tensor):
                awds.add_item(tensor)


        def consumer(tensor):
            ds.add_item(tensor)

        input_file = '{}{}'.format(input_prefix, ('.' + lang) if lang is not None else '')

        if args.singleSeq:
            if '.src' in input_file:
                res = TokenizerWCSSingleSequence.binarize(input_file, dict, consumer, L=args.L,
                                                          aconsumer=aconsumer, annotator=annotator)
                TokenizerWCSSingleSequence.printStats(res, lang, input_file, dict)
            else:
                res = TokenizerWCSSingleSequence.binarize(input_file, dict, consumer,
                                                          aconsumer=aconsumer, annotator=annotator)
                TokenizerWCSSingleSequence.printStats(res, lang, input_file, dict)
        else:
            if '.src' in input_file:
                res = TokenizerWCSParagraph.binarize(input_file, dict, consumer,
                                                     max_chunk_length= args.src_chunk_length, L=args.L,
                                                     aconsumer=aconsumer, awconsumer=awconsumer, annotator=annotator)
                TokenizerWCSParagraph.printStats(res, lang, input_file, dict)
            else:
                res = TokenizerWCSSentence.binarize(input_file, dict, consumer, max_chunk_length=args.tgt_chunk_length,
                                                    aconsumer=aconsumer, annotator=annotator)
                TokenizerWCSSentence.printStats(res, lang, input_file, dict)

        ds.finalize(dataset_dest_path(output_prefix, lang, 'idx'))
        if args.addAnnotations:
            ads.finalize(dataset_dest_path(output_prefix+"_keys", lang, 'idx'))
            awds.finalize(dataset_dest_path(output_prefix+"_keywords", lang, 'idx'))


    def make_dataset(input_prefix, output_prefix, lang):
        if args.output_format == 'binary':
            make_binary_dataset(input_prefix, output_prefix, lang)
        elif args.output_format == 'raw':
            # Copy original text file to destination folder
            output_text_file = dest_path(
                output_prefix + '.{}-{}'.format(args.source_lang, args.target_lang),
                lang,
            )
            shutil.copyfile(file_name(input_prefix, lang), output_text_file)

    def make_all(lang):
        if args.trainpref:
            make_dataset(args.trainpref, 'train', lang)
        if args.validpref:
            for k, validpref in enumerate(args.validpref.split(',')):
                outprefix = 'valid{}'.format(k) if k > 0 else 'valid'
                make_dataset(validpref, outprefix, lang)
        if args.testpref:
            for k, testpref in enumerate(args.testpref.split(',')):
                outprefix = 'test{}'.format(k) if k > 0 else 'test'
                make_dataset(testpref, outprefix, lang)

    make_all(args.source_lang)
    if target:
        make_all(args.target_lang)

    print('| Wrote preprocessed data to {}'.format(args.destdir))

    if args.alignfile:
        assert args.trainpref, "--trainpref must be set if --alignfile is specified"
        src_file_name = train_path(args.source_lang)
        tgt_file_name = train_path(args.target_lang)
        src_dict = load_dictionary(dict_path(args.source_lang))
        tgt_dict = load_dictionary(dict_path(args.target_lang))
        freq_map = {}
        with open(args.alignfile, 'r') as align_file:
            with open(src_file_name, 'r') as src_file:
                with open(tgt_file_name, 'r') as tgt_file:
                    for a, s, t in zip_longest(align_file, src_file, tgt_file):
                        si = Tokenizer.tokenize(s, src_dict, add_if_not_exist=False)
                        ti = Tokenizer.tokenize(t, tgt_dict, add_if_not_exist=False)
                        ai = list(map(lambda x: tuple(x.split('-')), a.split()))
                        for sai, tai in ai:
                            srcidx = si[int(sai)]
                            tgtidx = ti[int(tai)]
                            if srcidx != src_dict.unk() and tgtidx != tgt_dict.unk():
                                assert srcidx != src_dict.pad()
                                assert srcidx != src_dict.eos()
                                assert tgtidx != tgt_dict.pad()
                                assert tgtidx != tgt_dict.eos()

                                if srcidx not in freq_map:
                                    freq_map[srcidx] = {}
                                if tgtidx not in freq_map[srcidx]:
                                    freq_map[srcidx][tgtidx] = 1
                                else:
                                    freq_map[srcidx][tgtidx] += 1

        align_dict = {}
        for srcidx in freq_map.keys():
            align_dict[srcidx] = max(freq_map[srcidx], key=freq_map[srcidx].get)

        with open(os.path.join(args.destdir, 'alignment.{}-{}.txt'.format(
                args.source_lang, args.target_lang)), 'w') as f:
            for k, v in align_dict.items():
                print('{} {}'.format(src_dict[k], tgt_dict[v]), file=f)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
