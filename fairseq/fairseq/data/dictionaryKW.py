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

class DictionaryKW(object):

    def __init__(self, pad='<kpad>'):
        self.pad_word = pad
        self.symbols = []
        self.indices = {}
        # dictionary indexing starts at 1 for consistency with Lua
        self.add_symbol('<Lua heritage>')
        self.pad_index = self.add_symbol(pad)
        self.nspecial = len(self.symbols)

    def dummy_sentence(self, length):
        nbsent, sentlen = length
        t = torch.Tensor(nbsent, sentlen).uniform_(self.nspecial + 1, len(self)).long()
        return t

    @classmethod
    def load(cls, f, ignore_utf_errors=False):
        """Loads the dictionary from a text file with the format:

        ```
        <symbol0> <count0>
        <symbol1> <count1>
        ...
        ```
        """
        if isinstance(f, str):
            try:
                if not ignore_utf_errors:
                    with open(f, 'r', encoding='utf8') as fd:
                        return cls.load(fd)
                else:
                    with open(f, 'r', encoding='utf8', errors='ignore') as fd:
                        return cls.load(fd)
            except FileNotFoundError as fnfe:
                raise fnfe
            except Exception:
                raise Exception("Incorrect encoding detected in {}, please "
                                "rebuild the dataset".format(f))

        d = cls()
        for line in f.readlines():
            word = line.strip()
            if word == d.pad_word: #skip it is added in the __init__()
                continue
            d.indices[word] = len(d.symbols)
            d.symbols.append(word)
        return d

    def __len__(self):
        """Returns the number of symbols in the dictionary"""
        return len(self.symbols)

    def add_symbol(self, word, n=1):
        """Adds a word to the dictionary"""
        if word in self.indices:
            idx = self.indices[word]
            return idx
        else:
            idx = len(self.symbols)
            self.indices[word] = idx
            self.symbols.append(word)
            return idx

    def index(self, sym):
        """Returns the index of the specified symbol"""
        if sym in self.indices:
            return self.indices[sym]
        return self.unk_index

    def pad(self):
        """Helper to get index of pad symbol"""
        return self.pad_index