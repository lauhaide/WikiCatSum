# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from .dictionary import Dictionary
from .fairseq_dataset import FairseqDataset
from .indexed_dataset import IndexedInMemoryDataset, IndexedRawTextDataset, \
    IndexedInMemoryDatasetStruct, IndexedInMemoryDatasetStructKeys, RawTextDataset
from .language_pair_dataset import LanguagePairDataset
from .monolingual_dataset import MonolingualDataset
from .token_block_dataset import TokenBlockDataset

from .data_utils import EpochBatchIterator, collate_tokens

from .wikicatsum_pair_dataset import WikiCatSumPairDataset
from .dictionaryWCS import DictionaryWCS
from .dictionaryKW import DictionaryKW

from .wikicatsum_pair_dataset_topics import WikiCatSumPairDatasetTopics
from .wikicatsum_pair_dataset_topics_flatd import WikiCatSumPairDatasetTopicsFlatdec
from .wikicatsum_pair_dataset_topics_flate import WikiCatSumPairDatasetTopicsFlatEnc