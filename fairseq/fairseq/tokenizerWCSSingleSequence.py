# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from collections import Counter
import numpy as np

import fairseq.tokenizer as tokenizer
import fairseq.data.data_utils as du
from fairseq.preprocess.Constants import EOP, SNT, BLANK

class TokenizerWCSSingleSequence(tokenizer.Tokenizer):

    @staticmethod
    def binarize(filename, dict, consumer, tokenize=tokenizer.tokenize_line,
                 append_eos=True, reverse_order=False, L=None,
                 aconsumer=None, annotator=None):
        """
        :param filename:
        :param dict:
        :param consumer:
        :param tokenize:
        :param append_eos:
        :param reverse_order:
        :param L: maximum length of the input in tokens
        :return:
        """


        ntok = 0
        replaced = Counter()

        def replaced_consumer(word, idx):
            if idx == dict.unk_index and word != dict.unk_word:
                replaced.update([word])

        with open(filename, 'r') as f:
            example = 0
            for line in f:

                # As used in WikiSum inputs: "Construct inputs from Wiki title and references"
                # title tokens go at the begining, just remove separation token
                # source lines might have "<EOT>" separating the title at the beginning
                # leave it as WikiSum does

                line = line.replace(EOP, BLANK) # paragraphs markers are removed in hierarchical input format, also in sigle seqs
                line = line.replace(SNT, BLANK) # sentence markers are neither in single seqs, this is for tgt

                # take the fist L tokens from the source texts
                if L:
                    tokLine = line.split()
                    if len(tokLine) > L:
                        tokLine = tokLine[:L]
                        line = " ".join(tokLine)

                ids = tokenizer.Tokenizer.tokenize(
                    line=line,
                    dict=dict,
                    tokenize=tokenize,
                    add_if_not_exist=False,
                    consumer=replaced_consumer,
                    append_eos=append_eos,
                    reverse_order=reverse_order,
                )

                consumer(ids)
                ntok += len(ids)
                example +=1

        return {'ntok': ntok,  'nunk': sum(replaced.values()), 'replaced': len(replaced),
                '#examples':example,
                }

    @staticmethod
    def printStats(res, lang, input_file, dict):

        print('| [{}] {}: {} tokens, {:.3}% replaced by {}'.format(
            lang, input_file, res['ntok'],
            100 * res['nunk'] / res['ntok'], dict.unk_word))
        print("  #examples: {}".format(res['#examples']))