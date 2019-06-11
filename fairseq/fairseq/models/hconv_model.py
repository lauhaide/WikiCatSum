# Laura Perez

import torch.nn.functional as F

from . import ( FairseqEncoder, BaseFairseqModel, FairseqDecoder, )


class HFairseqModel(BaseFairseqModel):
    """Class for encoder-decoder hierarchical models."""

    def __init__(self, docEncoder, docDecoder):
        super().__init__()

        self.docEncoder = docEncoder
        self.docDecoder = docDecoder
        assert isinstance(self.docEncoder.encoder, FairseqEncoder)
        assert isinstance(self.docDecoder.decoder, FairseqDecoder)

    def forward(self, src_tokens, src_lengths, prev_output_tokens):

        encoder_out = self.docEncoder(src_tokens, src_lengths)
        dec_input = {}
        dec_input['prev_output_tokens'] = prev_output_tokens

        decoder_out = self.docDecoder(dec_input, encoder_out)

        return decoder_out

    def run_encoder_forward(self, input, srclen, ssentlen, beam_size):
        """Only do a forward encoder pass, used at test time (decoding will be incremental)"""
        src_tokens = input['src_tokens']
        src_lengths = input['src_lengths']

        return self.docEncoder(
                src_tokens.repeat(1, beam_size, 1).view(-1, srclen, ssentlen),
                src_lengths.expand(beam_size, src_lengths.numel()).t().contiguous().view(-1),
            )

    def max_positions(self):
        """Maximum length supported by the model."""
        return (self.encoder.max_positions(), self.decoder.max_positions())

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.docDecoder.decoder.get_normalized_probs(net_output, log_probs, sample)

    def get_targets(self, sample, net_output):
        """Get targets from either the sample or the net's output."""
        return sample['target']

    def max_decoder_positions(self):
        """Maximum length supported by the decoder."""
        return self.docDecoder.decoder.max_positions()



class HKVPredFairseqModel(HFairseqModel):
    """Class for encoder-decoder hierarchical models with topic label (key) prediction; hierarchical source and source topic labels."""

    def checkOOTKeyword(self, src_keywords):
        if self.docEncoder.getMaskOOT():
            OOTidx = self.docEncoder.kw_dictionary.index("OOT")
            ootmask = src_keywords.eq(OOTidx)
            if ootmask.any():
                src_keywords.masked_fill_(ootmask, self.docEncoder.kw_dictionary.pad())

    def forward(self, src_key, tgt_key, src_keywords, src_tokens, src_lengths, prev_output_tokens):
        """parameters here are according to 'net_input'"""

        self.checkOOTKeyword(src_keywords)

        encoder_out = self.docEncoder(src_tokens, src_lengths, src_key, src_keywords)

        dec_input = {}
        dec_input['prev_output_tokens'] = prev_output_tokens
        dec_input['tgt_key'] = self.get_target_keys__(tgt_key, prev_output_tokens)

        decoder_out = self.docDecoder(dec_input, encoder_out)

        return decoder_out

    def run_encoder_forward(self, input, srclen, ssentlen, beam_size):
        """Only do a forward encoder pass, used at test time (decoding will be incremental)."""

        src_tokens = input['src_tokens']
        src_lengths = input['src_lengths']
        src_keys = input['src_key']
        src_keywords = input['src_keywords']
        b, s, d = src_keys.size()

        self.checkOOTKeyword(src_keywords)

        return self.docEncoder(
                src_tokens.repeat(1, beam_size, 1).view(-1, srclen, ssentlen),
                src_lengths.expand(beam_size, src_lengths.numel()).t().contiguous().view(-1),
                src_keys.repeat(1, beam_size, 1).view(-1, s, d),
                src_keywords.repeat(1, beam_size, 1).view(-1, src_keywords.size(1), src_keywords.size(2)),
            )

    def get_target_keys(self, sample, net_output):
        """Get targets from either the sample or the net's output."""

        #sample['net_input']['tgt_key'] # [b x s x t] t is nb of topics

        return self.get_target_keys__(sample['net_input']['tgt_key'], sample['target'])

    def get_target_keys__(self, tgt_key, prev_tgt_tokens):
        """
        :param tgt_key: topic distribution vectors
        :param prev_tgt_tokens: previous target tokens
        :return: generated sequences of topics will be  [t1 t2 ... tn <eot>] padded wih with 0s.
        Fully padded sentences with generate  ti=0 .
        """

        # generate topic labels from topic distribution
        b, s, t = tgt_key.size()
        self.proto_tgt_key = tgt_key.new(b, s, t+1) # add one topic dim more for padding and zeros topic distinction
        self.proto_tgt_key.fill_(0.0)
        self.proto_tgt_key[:, :, 1:].copy_(tgt_key)

        values, indices = self.proto_tgt_key.max(2)
        # take threshold for topic score, if less that it, make the topic of the sentence OtherTopic
        topicScoreThreshold = self.docDecoder.getTargetTopicThreshold()
        indices[values.lt(topicScoreThreshold)-values.eq(0)] = t

        # add end of topic sequence token
        self.eod = self.docDecoder.decoder.dictionary.eod()  # get token used for end-of-document (i.e. end of sequence of sentences)
        eod_sentences = prev_tgt_tokens.eq(self.eod).sum(2)

        if eod_sentences.sum() > 0:
            eots = indices.new(b).fill_(t+1)
            indices = indices.masked_scatter_(eod_sentences.eq(1), eots)

        return indices.view(-1)  # [b*s]

    def get_target_keys_padidx__(self):
        return 0

    def getNormalizedProbs_Keys(self, keyout):
        keylogits = keyout.float()
        return F.softmax(keylogits, dim=-1)