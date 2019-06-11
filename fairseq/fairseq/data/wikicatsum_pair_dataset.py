# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import numpy as np
import torch

from . import data_utils, FairseqDataset

def collate_tokens2D(values, pad_idx, eos_idx, left_pad, move_eos_to_beginning=False, use_prefix=None):
    """***Modified to (LP)*** Convert a list of 2d tensors into a padded 3d tensor."""

    size1 = max(v.size(0) for v in values)
    size2 = max(v.size(1) for v in values)
    l=[v.size(1) for v in values]
    v = values[0]
    if use_prefix : #and move_eos_to_beginning: #this can happen when processing target only
        res = values[0].new(len(values), size1, use_prefix + size2).fill_(pad_idx)
    else:
        res = values[0].new(len(values), size1, size2).fill_(pad_idx)

    def copy_tensor(src, dst):
        if use_prefix :
            assert dst.numel() == src.numel() + (src.size(0)*use_prefix), \
                "{} ... {} / {} / {} / {} / {}".format(dst.shape, src.shape, l, size2, v.shape, len(v))
        else:
            assert dst.numel() == src.numel(), \
                "{} ... {} / {} / {} / {} / {}".format(dst.shape, src.shape, l, size2, v.shape, len(v))

        ##See how to move 2D targets one position to the right, shift them for the input to decoder
        if move_eos_to_beginning:
            assert src.eq(eos_idx).sum() == len(src),  src
            if use_prefix:
                dst[:, use_prefix + 1:] = src[:,:-1]
                dst[dst.eq(eos_idx)] = pad_idx
                dst[:,use_prefix] = eos_idx
                # now fill prefixes!
                prex = dst.new(dst.size(0), use_prefix).fill_(pad_idx)
                for i in reversed(range(1, prex.size(0))): # copy prefix from previous sentence
                    prevDstWoEos =  dst[i-1, use_prefix + 1:] # skip prefix space and EOS positions
                    notpadded = prevDstWoEos.ne(pad_idx).nonzero().squeeze(1) # get non padded token positions
                    if notpadded.size(0) >= use_prefix:
                        notpadded = notpadded[-use_prefix:] # take the last <use_prefix> ones
                        prex[i, :] = prevDstWoEos[notpadded]  # take those from ith source that were not pads
                    else:
                        prex[i, -notpadded.size(0):] = prevDstWoEos[notpadded]
                prex[0,:] = pad_idx # make prefix of first sentence pad
                dst[:, :use_prefix] = prex
            else:
                dst[:,1:] = src[:,:-1]
                dst[dst.eq(eos_idx)] = pad_idx
                dst[:,0] = eos_idx
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        offset = 0 if use_prefix is None else use_prefix
        copy_tensor(v, res[i][:v.size(0) , size2 - v.size(1):] if left_pad else res[i][:v.size(0), :v.size(1) + offset ])

    return res


def collate_flatten_tokens(values, pad_idx, eos_idx, left_pad, move_eos_to_beginning=False):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.ne(pad_idx).sum()-v.eq(eos_idx).sum()+1 for v in values)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel(), "{} / {} / size={} / {}".format(dst.numel(), src.numel(), size, res.shape)
        if move_eos_to_beginning:
            assert src[-1] == eos_idx
            dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        vlen = v.ne(pad_idx).sum()-v.eq(eos_idx).sum()+1
        vflat = torch.masked_select(v, v.ne(pad_idx)) # remove all/intermediate pad symbols
        vflat = torch.masked_select(vflat, vflat.ne(eos_idx))# remove all/intermediate eos symbols

        copy_tensor(torch.cat((vflat,vflat.new(1).fill_(eos_idx)), 0), res[i][-vlen:] if left_pad else res[i][:vlen])

    return res


def collate(samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False, use_prefix=None, flatten_source=False):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False, use_prefix=None):
        return collate_tokens2D(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
            use_prefix,
        )
    def merge_flat(key, left_pad, move_eos_to_beginning=False, use_prefix=None):
        return collate_flatten_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad_source)
    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    if flatten_source:
        src_tokens_flat = merge_flat('source', left_pad=left_pad_source)
        src_tokens_flat = src_tokens_flat.index_select(0, sort_order)
    else:
        src_tokens_flat = None


    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        if use_prefix: #mask prefix in target
            prex_mask = target.new(target.size(0), target.size(1), use_prefix).fill_(pad_idx)
            target = torch.cat((prex_mask,target), dim=2)

        # we create a shifted version of targets for feeding the
        # previous output token(s) into the next decoder step
        prev_output_tokens = merge(
            'target',
            left_pad=left_pad_target,
            move_eos_to_beginning=True,
            use_prefix=use_prefix,
        )
        prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
        target = target.index_select(0, sort_order)
        ntokens = sum(s['target'].ne(pad_idx).sum() for s in samples).item()
    else:
        ntokens = sum(s['source'].ne(pad_idx).sum() for s in samples).item()

    if flatten_source:
        netin = {
                'src_tokens': src_tokens, # [b x s x w]
                'src_lengths': src_lengths, # b x nb.tokens TODO: see if this is the info we need later on, *nb.sent* instead???
                'src_tokens_flat': src_tokens_flat, # [b x s*w] without intermediate pad and eos symbols
                'prev_output_tokens': prev_output_tokens, # b x sent x word
            }
    else:
         netin = {
                'src_tokens': src_tokens, # [b x s x w]
                'src_lengths': src_lengths, # b x nb.tokens TODO: see if this is the info we need later on, *nb.sent* instead???
                'prev_output_tokens': prev_output_tokens, # b x sent x word
            }

    return {
        'id': id,
        'ntokens': ntokens,
        'net_input': netin,
        'target': target,
    }


class WikiCatSumPairDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets. Source and Target are
    structured/hierarchical representations (e.g. sentences, paragraphs)
    """

    def __init__(
        self, src, src_sizes, src_dict, src_txt=None, tgt_txt=None,
        tgt=None, tgt_sizes=None, tgt_dict=None,
        left_pad_source=True, left_pad_target=False,
        #max_source_positions=1024, max_target_positions=1024,
        max_source_positions=None, max_target_positions=None, # at this point this will be for nb. sentences
        max_src_sentence_length=None, max_tgt_sentence_length=None, shuffle=True,
        use_prefix=None, flattenSource=False
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        self.src = src
        self.tgt = tgt
        self.src_sizes = src_sizes
        self.tgt_sizes = tgt_sizes if tgt_sizes is not None else None
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.max_src_sentence_length = max_src_sentence_length
        self.max_tgt_sentence_length = max_tgt_sentence_length
        self.shuffle = shuffle
        self.use_prefix = use_prefix
        self.target_txt = tgt_txt
        self.source_txt = src_txt
        self.flattenSource = flattenSource

    def __getitem__(self, index):
        return {
            'id': index,
            'source': self.src[index],
            'target': self.tgt[index] if self.tgt is not None else None,
        }

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch."""
        return collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            use_prefix=self.use_prefix, flatten_source=self.flattenSource,
        )

    def get_dummy_batch(self, num_tokens, max_positions, src_len=128, tgt_len=128, sent_len=128):
        (max_source_positions, max_ssentence_length), \
        (max_target_positions, max_tsentence_length) = self._get_max_positions(max_positions)

        src_len, tgt_len = min(src_len, max_source_positions), min(tgt_len, max_target_positions)
        ssent_len =  min(sent_len, max_ssentence_length)
        tsent_len = min(sent_len, max_tsentence_length)
        bsz = num_tokens // max(src_len * ssent_len, tgt_len * tsent_len)
        #bsz is the nb of sentences per batch, for us will be nb of docs
        src_tensor = self.src_dict.dummy_sentence((src_len, ssent_len))
        tgt_tensor = self.tgt_dict.dummy_sentence((tgt_len, tsent_len - self.use_prefix)) if self.tgt_dict is not None else None
        print("* dynamic batch size {}".format(bsz))
        return self.collater([
            {
                'id': i,
                'source': src_tensor,
                'target': tgt_tensor,
            }
            for i in range(bsz)
        ])

    def num_tokens(self, index):
        """Return an example's length (number of tokens), used for batching."""
        return self.src_sizes[index][0]*self.src_sizes[index][1]+ (self.tgt_sizes[index][0]*self.tgt_sizes[index][1]
                        if self.tgt_sizes is not None else 0)

    def ordered_indices(self):
        """Ordered indices for batching."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))

        # take the sentence_length dimension of the document (nb_sentences x sentence_length)
        if self.tgt_sizes is not None:
            indices = indices[np.argsort(self.tgt_sizes[indices][:,-1], kind='mergesort')]

        return indices[np.argsort(self.src_sizes[indices][:,-1], kind='mergesort')]

    def valid_size(self, index, max_positions):
        """Check if an example's size is valid according to max_positions."""
        (max_source_positions, msrclen), (max_target_positions, mtgtlen) = self._get_max_positions(max_positions)

        return (
            (self.src_sizes[index][0] <= max_source_positions
             and self.src_sizes[index][1] <= msrclen)
            and (self.tgt_sizes is None or (
                self.tgt_sizes[index][0] <= max_target_positions)
                 and self.tgt_sizes[index][1] <= mtgtlen)
        )

    def _get_max_positions(self, max_positions):
        if max_positions is None:
            return (self.max_source_positions, self.max_src_sentence_length),\
                   (self.max_target_positions, self.max_tgt_sentence_length)
        assert len(max_positions) == 2
        (max_src_pos, max_ssent_len), (max_tgt_pos, max_tsent_len) = max_positions
        return (min(self.max_source_positions, max_src_pos), \
               min(self.max_src_sentence_length, max_ssent_len)), \
               (min(self.max_target_positions, max_tgt_pos), \
               min(self.max_tgt_sentence_length, max_tsent_len))

    def get_target_original_text(self, i):
        return self.target_txt.get_original_text(i)

    def get_source_original_text(self, i):
        return self.source_txt.get_original_text(i)