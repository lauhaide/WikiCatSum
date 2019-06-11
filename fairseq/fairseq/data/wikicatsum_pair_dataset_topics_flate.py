# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch
import numpy as np

from .wikicatsum_pair_dataset import WikiCatSumPairDataset, collate_tokens2D
from .wikicatsum_pair_dataset_topics import collate_vectors2D
from . import data_utils

def collate(samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False, dummy=False, use_prefix=False):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return collate_tokens2D(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    def merge_flat(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge_flat('source', left_pad=left_pad_source)
    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        # we create a shifted version of targets for feeding the
        # previous output token(s) into the next decoder step
        prev_output_tokens = merge(
            'target',
            left_pad=left_pad_target,
            move_eos_to_beginning=True,
        )
        prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
        target = target.index_select(0, sort_order)

        ntokens = sum(s['target'].ne(pad_idx).sum() for s in samples).item()
    else:
        ntokens = sum(s['source'].ne(pad_idx).sum() for s in samples).item()

    # prepare source topic keywords
    if samples[0].get('src_keywords', None) is not None:
        src_keywords = merge('src_keywords', left_pad=left_pad_source)
        src_keywords = src_keywords.index_select(0, sort_order)

    tgt_keys = collate_vectors2D([s['tgt_key'] for s in samples], dummy)
    tgt_keys = tgt_keys.index_select(0, sort_order.cuda())

    return {
        'id': id,
        'ntokens': ntokens,
        'net_input': {
            'tgt_key': tgt_keys,  # [b x s x kdim]
            'src_keywords' : src_keywords,
            'src_tokens': src_tokens, # [b x s x w]
            'src_lengths': src_lengths, # b x nb.tokens TODO: see if this is the info we need later on, *nb.sent* instead???
            'prev_output_tokens': prev_output_tokens, # b x sent x word
        },
        'target': target,
    }


class WikiCatSumPairDatasetTopicsFlatEnc(WikiCatSumPairDataset):
    """
    A pair of torch.utils.data.Datasets. Source and Target are
    structured/hierarchical representations (e.g. sentences, paragraphs).
    Textual elements are associated with content keys (eg. lda topic distribution or cluster keywords).
    """

    def __init__(
        self, src_keys, tgt_keys, src, src_sizes, src_dict, src_txt=None, tgt_txt=None,
        tgt=None, tgt_sizes=None, tgt_dict=None,
        keywords_dict=None, src_dataset_keywords=None, tgt_dataset_keywords=None,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=15,
        max_src_sentence_length=40, max_tgt_sentence_length=None, shuffle=True,
        kdim=None, use_prefix=None,
    ):
        super().__init__(src, src_sizes, src_dict, src_txt=src_txt, tgt_txt=tgt_txt,
        tgt=tgt, tgt_sizes=tgt_sizes, tgt_dict=tgt_dict,
        left_pad_source=left_pad_source, left_pad_target=left_pad_target,
        max_source_positions=max_source_positions, max_target_positions=max_target_positions,
        max_src_sentence_length=max_src_sentence_length, max_tgt_sentence_length=max_tgt_sentence_length, shuffle=shuffle)

        self.tgt_keys = tgt_keys
        self.kdim = kdim
        self.keywords_dict = keywords_dict
        self.src_keywords = src_dataset_keywords
        self.tgt_keywords = tgt_dataset_keywords
        self.use_prefix = use_prefix

    def __getitem__(self, index):
        return {
            'id': index,
            'tgt_key': self.tgt_keys[index],
            'src_keywords': self.src_keywords[index],
            'source': self.src[index],
            'target': self.tgt[index] if self.tgt is not None else None,
        }

    def collater(self, samples, dummy=False):
        """Merge a list of samples to form a mini-batch."""
        return collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target, dummy=dummy,
            use_prefix=self.use_prefix,
        )

    def get_dummy_batch(self, num_tokens, max_positions, src_len=128, tgt_len=128, sent_len=128):
        max_source_positions, \
        (max_target_positions, max_tsentence_length) = self._get_max_positions(max_positions)

        src_len, tgt_len = min(src_len, max_source_positions), min(tgt_len, max_target_positions)
        tsent_len = min(sent_len, max_tsentence_length)
        bsz = num_tokens // max(src_len , tgt_len * tsent_len)
        #bsz is the nb of sentences per batch, for us will be nb of docs
        src_tensor = self.src_dict.dummy_sentence(src_len)
        tgt_tensor = self.tgt_dict.dummy_sentence((tgt_len, tsent_len)) if self.tgt_dict is not None else None
        tgt_key_tensor = torch.Tensor(tgt_len, self.kdim).uniform_().type('torch.cuda.FloatTensor')
        src_keywords_tensor = self.keywords_dict.dummy_sentence((src_len, 20))
        print("* dynamic batch size {}".format(bsz))
        return self.collater([
            {
                'id': i,
                'tgt_key': tgt_key_tensor,
                'src_keywords': src_keywords_tensor,
                'source': src_tensor,
                'target': tgt_tensor,
            }
            for i in range(bsz)
        ], True)

    def num_tokens(self, index):
        """Return an example's length (number of tokens), used for batching."""
        return self.src_sizes[index] + (self.tgt_sizes[index][0] * self.tgt_sizes[index][1] if self.tgt_sizes is not None else 0)


    def ordered_indices(self):
        """Ordered indices for batching."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))

        # take the sentence_length dimension of the document (nb_sentences x sentence_length)
        if self.tgt_sizes is not None:
            indices = indices[np.argsort(self.tgt_sizes[indices][:,-1], kind='mergesort')]

        return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]


    def valid_size(self, index, max_positions):
        """Check if an example's size is valid according to max_positions."""
        max_source_positions, (max_target_positions, mtgtlen) = self._get_max_positions(max_positions)

        return (
            self.src_sizes[index] <= max_source_positions
            and (self.tgt_sizes is None or (
                self.tgt_sizes[index][0] <= max_target_positions)
                 and self.tgt_sizes[index][1] <= mtgtlen)
        )

    def _get_max_positions(self, max_positions):
        if max_positions is None:
            return self.max_source_positions,\
                   (self.max_target_positions, self.max_tgt_sentence_length)
        assert len(max_positions) == 2
        max_src_pos, (max_tgt_pos, max_tsent_len) = max_positions
        return min(self.max_source_positions, max_src_pos), \
               (min(self.max_target_positions, max_tgt_pos), \
               min(self.max_tgt_sentence_length, max_tsent_len))
