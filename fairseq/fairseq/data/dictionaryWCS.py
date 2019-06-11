# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from collections import Counter
import os

import torch

from .dictionary import Dictionary

class DictionaryWCS(Dictionary):

    def __init__(self, pad='<pad>', eos='</s>', unk='<unk>', eod='</d>', sod='<d>'):
        self.unk_word, self.pad_word, self.eos_word = unk, pad, eos
        self.sod_word, self.eod_word = sod, eod
        self.symbols = []
        self.count = []
        self.indices = {}
        # dictionary indexing starts at 1 for consistency with Lua
        self.add_symbol('<Lua heritage>')
        self.pad_index = self.add_symbol(pad)
        self.eos_index = self.add_symbol(eos)
        self.unk_index = self.add_symbol(unk)
        #for document
        self.sod_index = self.add_symbol(sod)
        self.eod_index = self.add_symbol(eod)
        self.nspecial = len(self.symbols)

    def dummy_sentence(self, length):
        nbsent, sentlen = length
        t = torch.Tensor(nbsent, sentlen).uniform_(self.nspecial + 1, len(self)).long()
        t[:, -1] = self.eos()
        return t

    def sod(self):
        """Helper to get index of sod symbol"""
        return self.sod_index

    def eod(self):
        """Helper to get index of eod symbol"""
        return self.eod_index