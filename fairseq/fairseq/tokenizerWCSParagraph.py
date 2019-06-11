# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from collections import Counter
import numpy as np
import torch

import fairseq.tokenizer as tokenizer
import fairseq.data.data_utils as du

EOT = " <EOT> "  # end-of-title string
EOP = " <EOP> "  # end-of-paragraph string

class TokenizerWCSParagraph(tokenizer.Tokenizer):

    @staticmethod
    def getPaddingFor(filename):
        """
        gggrrrrrrr
        This is awful, but need to binarise seqs of seqs in tensors, then need same lengths.
        These parameters are used when training/inference when preparing batches. But I'll
        need these here before :(
        """
        #Parameters in Task class are:
        #parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
        #                    help='pad the source on the left (default: True)')
        #parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
        #                    help='pad the target on the left (default: False)')
        #init this values as used for the model default (couldn find where they are defined)
        left_pad_source = True
        left_pad_target= False
        if "src" in filename:
            return left_pad_source
        elif "tgt" in filename:
            return left_pad_target

    @staticmethod
    def binarize(filename, dict, consumer, tokenize=tokenizer.tokenize_line,
                 append_eos=True, reverse_order=False, max_chunk_length=300, L=500,
                 aconsumer=None, awconsumer=None, annotator=None):

        def buildBOWAnnotationFileName(name):
            path = name[:name.rfind("/")]
            file = name[name.rfind("/"):]
            file = file.replace(".", ".bow.")
            return path + file

        ntok = 0
        replaced = Counter()

        def replaced_consumer(word, idx):
            if idx == dict.unk_index and word != dict.unk_word:
                replaced.update([word])

        def chunks(phrase, n):
            """Yield successive n-sized chunks from l."""
            l = phrase.split()
            return [" ".join(l[i:i + n]) for i in range(0, len(l), n)]

        leftPad = TokenizerWCSParagraph.getPaddingFor(filename)
        print("*    Left padding for file {} ({})".format(filename, leftPad))
        print("* max_chunk_length={}, L={}".format(max_chunk_length, L))
        with open(filename, 'r') as f:
          with open(buildBOWAnnotationFileName(filename)) as b:
            nbparagraphs = []
            nbxparagraphs = []
            lenXparagraphs = []
            lenparagraphs = []

            longerChunk = 0
            example = 0
            cnt=0
            for line, bowline in zip(f, b):

                toAnnotate = []
                # As used in WikiSum inputs: "Construct inputs from Wiki title and references"
                # title tokens go at the begining, just remove separation token
                # source lines might have "<EOT>" separating the title at the beginning
                # leave it as WikiSum does

                # take the fist L tokens from the source texts
                tokLine = line.split()
                if len(tokLine) > L:
                    tokLine = tokLine[:L]

                paragraph_split = line.split(EOP) ## no need to remove sentence markers " <SNT>", there are none
                bowparagraph_split = bowline.split(EOP)
                assert len(paragraph_split)==len(bowparagraph_split), \
                        "INCONSISTENT LENGTHS @{}\t{}\t{}\n".format(cnt, len(paragraph_split), len(bowparagraph_split))

                # separate the first L=800 in paragraphs
                paragraph_split = " ".join(tokLine).split(EOP)

                text_ids = []
                annotation = []
                nbparagraphs.append(len(paragraph_split)) #nb of paragraphs
                nbxpara = 0
                for pa, paragraph in enumerate(paragraph_split):
                        bowparagraph = bowparagraph_split[pa].strip()

                        plen = len(paragraph.strip().split()) #nb of tokens per pargargraph
                        lenparagraphs.append(plen)
                        if plen > max_chunk_length-1: #do not count EOS added by tokenize
                            longerChunk +=1
                            xphrases = chunks(paragraph, max_chunk_length - 1)
                        else:
                            xphrases = [paragraph]
                        nbxpara += len(xphrases)
                        for phrase in xphrases:
                            lenXparagraphs.append(len(phrase.split())) #new nb of tokens per pargargraph
                            #print(phrase)
                            ids = tokenizer.Tokenizer.tokenize(
                                line=phrase,
                                dict=dict,
                                tokenize=tokenize,
                                add_if_not_exist=False,
                                consumer=replaced_consumer,
                                append_eos=append_eos,
                                reverse_order=reverse_order,
                            )
                            text_ids.append(ids)
                            ntok += len(ids)
                            toAnnotate.append(bowparagraph.split())

                if toAnnotate and annotator:
                    annotation = annotator.topicDistrib(toAnnotate)
                    annotation_kw = annotator.keywords(toAnnotate)

                ids = du.collate_tokens(text_ids, dict.pad(), dict.eos(), left_pad=leftPad)

                consumer(ids)
                if annotation and annotator:
                    if aconsumer:
                        aconsumer(torch.stack(annotation))
                    if awconsumer:
                        awconsumer(torch.stack(annotation_kw))

                example +=1
                nbxparagraphs.append(nbxpara)  # new nb of paragraphs
                cnt+=1

        paras = np.asarray(nbparagraphs)
        xparas = np.asarray(nbxparagraphs)
        parasxlens = np.asarray(lenXparagraphs)
        paraslens = np.asarray(lenparagraphs)
        return {'ntok': ntok,  'nunk': sum(replaced.values()), 'replaced': len(replaced),
                'avg_nb_paras': paras.mean(), 'avg_nb_xparas': xparas.mean(), 'avg_paraXlens':parasxlens.mean(),'avg_paralens':paraslens.mean(),
                'std_nb_paras': paras.std(), 'std_nb_xparas': xparas.std(), 'std_paraXlens': parasxlens.std(), 'std_paralens': paraslens.std(),
                'max_nb_paras': paras.max(), 'max_nb_xparas': xparas.max(), 'max_paraXlens': parasxlens.max(),'max_paralens': paraslens.max(),
                'longerChunk': longerChunk, '#examples':example,
                }

    @staticmethod
    def printStats(res, lang, input_file, dict):

        print('| [{}] {}: {} tokens, {:.3}% replaced by {}'.format(
            lang, input_file, res['ntok'],
            100 * res['nunk'] / res['ntok'], dict.unk_word))
        print("* Nb. original paragraphs: {} (+- {} | {})".format(round(res['avg_nb_paras'],2), round(res['std_nb_paras'],2), res['max_nb_paras']))
        print("* Original Paragraphs lengths: {} (+- {} | {})".format(round(res['avg_paralens'], 2), round(res['std_paralens'], 2), res['max_paralens']))
        print("* Nb. segmented paragraphs: {} (+- {} | {})".format(round(res['avg_nb_xparas'],2), round(res['std_nb_xparas'],2), res['max_nb_xparas']))
        print("* Segmented Paragraphs lengths: {} (+- {} | {})".format(round(res['avg_paraXlens'],2), round(res['std_paraXlens'],2), res['max_paraXlens']))
        print("* Nb. Segmented paragraphs {}".format(res['longerChunk']))
        print("  #examples: {}".format(res['#examples']))