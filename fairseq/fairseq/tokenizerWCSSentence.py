# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from collections import Counter
import string
import numpy as np


import torch

import fairseq.tokenizer as tokenizer
import fairseq.data.data_utils as du
from fairseq.preprocess.Constants import EOP, SNT

class TokenizerWCSSentence(tokenizer.Tokenizer):

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
                 append_eos=True, reverse_order=False, max_chunk_length=40,
                 aconsumer=None, annotator=None):
        """

        :param filename:
        :param dict:
        :param consumer:
        :param tokenize:
        :param append_eos:
        :param reverse_order:
        :param max_chunk_length:
        :return:
        """
        def buildBOWAnnotationFileName(name):
            path = name[:name.rfind("/")]
            file = name[name.rfind("/"):]
            file = file.replace(".", ".bow.")
            return path + file


        nseq, ntok = 0, 0
        replaced = Counter()

        def replaced_consumer(word, idx):
            if idx == dict.unk_index and word != dict.unk_word:
                replaced.update([word])

        def chunks(phrase, n):
            """Yield successive n-sized chunks from l."""
            l = phrase.split()
            return [" ".join(l[i:i + n]) for i in range(0, len(l), n)]


        leftPad = TokenizerWCSSentence.getPaddingFor(filename)
        print("*    Left padding for file {} ({})".format(filename, leftPad))
        print("* max_chunk_length={}".format(max_chunk_length))

        line4 = 0
        line8 = 0
        with open(filename, 'r') as f:
          with open(buildBOWAnnotationFileName(filename)) as b:
            nbsentences = []
            nbsentences_paras = []
            realsentlens = []
            sentlens = []
            longerSent = 0
            examplesWithChunks = 0
            example = 0
            cnt=0
            ids_minlen2=0
            ids_minlen4 = 0
            toolong = 0
            toolong_sampleID = []
            for e, (line, bowline) in enumerate(zip(f, b)):
                toAnnotate = []
                text_ids = []
                annotation = []
                paragraph_split = line.split(EOP)
                bowparagraph_split = bowline.split(EOP)
                assert len(paragraph_split)==len(bowparagraph_split), \
                        "INCONSISTENT LENGTHS @{}\n\n{}\n{}".format(cnt, len(paragraph_split), len(bowparagraph_split))

                chunk = False
                nbsent = 0
                for pa, paragraph in enumerate(paragraph_split): #there is only 1, now lead section is split by sentences only #TODO
                    bowparagraph = bowparagraph_split[pa]

                    # now strip() here bow has empty BoWs for some sentences!
                    phrase_split = paragraph.split(SNT)
                    bowphrase_split = bowparagraph.split(SNT)

                    assert len(phrase_split)==len(bowphrase_split), \
                        "INCONSISTENT LENGTHS @{}\n\n{}\n{}".format(cnt, paragraph_split, bowparagraph_split)

                    nbsentences_paras.append(len(phrase_split))
                    nbsent += len(phrase_split)

                    for ph, xphrase in enumerate(phrase_split):
                        bowxphrase = bowphrase_split[ph]

                        fragment_to_discard = (len(xphrase.split()) > 200)
                        if len(xphrase.split()) > 200:
                            toolong_sampleID.append(e)
                            toolong+=1

                        # some xphrase come here with \n at the end, it seems not a problem for subsequent tokenization
                        if len(xphrase.split()) <=4:
                            print(xphrase)
                            line4+=1
                        if len(xphrase.split()) <=8:
                            print(xphrase)
                            assert len(xphrase.split()) == len(xphrase.strip().split())
                            line8+=1

                        realsentlens.append(len(xphrase.split()))
                        if len(xphrase.split()) > max_chunk_length-1: #do not count EOS added by tokenize
                            longerSent +=1
                            if fragment_to_discard:
                                # should use sample_ids; for the moment do this, not nice but this needs no extra code
                                # fragment into multiple pieces to make this example be discarded later on.
                                xphrases = chunks(xphrase, 10)
                            else:
                                xphrases = chunks(xphrase, max_chunk_length - 1)
                            chunk = True
                        else:
                            xphrases = [xphrase]
                        for phrase in xphrases:
                            sentlens.append(len(phrase.split()))
                            ids = tokenizer.Tokenizer.tokenize(
                                line=phrase,
                                dict=dict,
                                tokenize=tokenize,
                                add_if_not_exist=False,
                                consumer=replaced_consumer,
                                append_eos=append_eos,
                                reverse_order=reverse_order,
                            )

                            if len(ids) <=2: # count how many got chunked right before fullstop :(
                                ids_minlen2+=1
                            elif len(ids) <=4:
                                ids_minlen4+=1

                            text_ids.append(ids)
                            ntok += len(ids)
                            toAnnotate.append(bowxphrase.split()) #annotate all splits of a given phrase with same topics

                if toAnnotate and annotator:
                    annotation = annotator.topicDistrib(toAnnotate, 'sent')

                if chunk:
                    examplesWithChunks +=1
                nbsentences.append(nbsent)
                nseq += 1

                #append the end of document token
                text_ids.append(tokenizer.Tokenizer.tokenize(
                        line=dict.eod_word,
                        dict=dict,
                        tokenize=tokenize,
                        add_if_not_exist=False,
                        consumer=replaced_consumer,
                        append_eos=append_eos,
                        reverse_order=reverse_order,
                    ))
                ids = du.collate_tokens(text_ids, dict.pad(), dict.eos(), left_pad=leftPad)

                consumer(ids)
                example +=1
                if annotation and annotator:
                    aconsumer(torch.stack(annotation))


                cnt+=1

        print("\n>200: ",toolong)
        print("short ids: ", ids_minlen2, ids_minlen4)
        print("************* this shorts??")
        print("<4: ",line4)
        print("<8: ",line8)
        print("list of examples with sentence length >200: ")
        print(toolong_sampleID)

        avg_sents_paras = np.asarray(nbsentences_paras) # nb sentences per paragraph
        avg_sents = np.asarray(nbsentences) # nb sentences per lead section (now should be same)
        avg_sentlens = np.asarray(sentlens)
        avg_realsentlens = np.asarray(realsentlens)
        #return None
        return {'nseq': nseq, 'nunk': sum(replaced.values()),
                'ntok': ntok, 'replaced': len(replaced),
                'avg_sents_ppara': avg_sents_paras.mean(), 'avg_sentlens':avg_sentlens.mean(),
                'std_sents_ppara': avg_sents_paras.std(),  'std_sentlens': avg_sentlens.std(),
                'max_sents_ppara': avg_sents_paras.max(), 'max_sentlens': avg_sentlens.max(),
                'avg_realsentlens':avg_realsentlens.mean(), 'std_realsentlens': avg_realsentlens.std(),  'max_realsentlens': avg_realsentlens.max(),
                'avg_sents':avg_sents.mean(), 'std_sents': avg_sents.std(), 'max_sents': avg_sents.max(),
                'longerSent': longerSent,
                'examplesWithChunks':examplesWithChunks, 'totalSents':avg_sents.sum(),'#examples':example,
                }

    @staticmethod
    def printStats(res, lang, input_file, dict):

        print('| [{}] {}: {} sents, {} tokens, {:.3}% replaced by {}'.format(
            lang, input_file, res['nseq'], res['ntok'],
            100 * res['nunk'] / res['ntok'], dict.unk_word))
        print("* Nb. original sentences: {} (+- {} | {})".format(round(res['avg_sents_ppara'],2), round(res['std_sents_ppara'],2), res['max_sents_ppara']))
        print("* Original sentence lengths: {} (+- {} | {})".format(round(res['avg_realsentlens'], 2), round(res['std_realsentlens'], 2), res['max_realsentlens']))
        print("* Total segmented sentences {} / #examples with chunked sents.: {}".format(res['longerSent'], res['examplesWithChunks']))
        print("* Total nb sentences {}".format(res['totalSents']))
        print("* Segmented Sentence lengths: {} (+- {} | {})".format(round(res['avg_sentlens'], 2), round(res['std_sentlens'], 2),
                                                           res['max_sentlens']))
        print("  #examples: {}".format(res['#examples']))
