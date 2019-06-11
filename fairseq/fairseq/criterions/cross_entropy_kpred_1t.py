# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch.nn.functional as F
import torch

from fairseq import utils

from . import FairseqCriterion, register_criterion


@register_criterion('cross_entropy_kpred_1t')
class CrossEntropyCombinedKPred1TCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)

        self.topic_padding_idx = 0

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)

        loss = F.nll_loss(lprobs, target, size_average=False, ignore_index=self.padding_idx,
                          reduce=reduce)

        keys_lprobs = self.getKeyProbs(net_output[3])
        keys_target = model.get_target_keys(sample, net_output)
        key_tokens = keys_target.ne(self.topic_padding_idx).sum().item()

        loss_keys = F.nll_loss(keys_lprobs, keys_target, size_average=False,
                               ignore_index=self.topic_padding_idx,
                               reduce=reduce)

        lambdaWeight = model.getLambda()
        total_loss = loss + (1 if lambdaWeight is None else lambdaWeight) * loss_keys

        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(total_loss.data) if reduce else total_loss.data,
            'ntokens': sample['ntokens'],
            'sample_size': sample_size,
        }
        return total_loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        agg_output = {
            'loss': loss_sum / sample_size / math.log(2),
            'sample_size': sample_size,
        }
        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)
        return agg_output

    @staticmethod
    def getKeyProbs(sentenceOutVectors):
        """Returns the paired consecutive sentence vectors"""

        logits = sentenceOutVectors.float() #[b*s, d]
        keysProbs = F.log_softmax(logits, dim=-1)

        return keysProbs

