# Laura Perez

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from fairseq import options, utils
from fairseq.modules import ( AdaptiveSoftmax, )

from . import ( FairseqEncoder, register_model, register_model_architecture, )

from .hconv_model import HKVPredFairseqModel

from .fconv import FConvEncoder, FConvDecoder

from .lstm import LSTMCell, Linear, AttentionLayer

class HKVPredFairseqModel2(HKVPredFairseqModel):
    """overrides forward and encoder forward calls"""

    def forward(self, tgt_key, src_keywords, src_tokens, src_lengths, prev_output_tokens):
        """parameters here are according to 'net_input'"""

        encoder_out = self.docEncoder(src_tokens, src_lengths, src_keywords)

        dec_input = {}
        dec_input['prev_output_tokens'] = prev_output_tokens
        dec_input['tgt_key'] = self.get_target_keys__(tgt_key, prev_output_tokens)

        decoder_out = self.docDecoder(dec_input, encoder_out)

        return decoder_out

    def run_encoder_forward(self, input, srclen, beam_size):
        """Only do a forward encoder pass, used at test time (decoding will be incremental)"""

        src_tokens = input['src_tokens']
        src_lengths = input['src_lengths']
        src_keywords = input['src_keywords']
        bsz = src_tokens.size(0)
        b, s, k = src_keywords.size()
        assert bsz==b

        return self.docEncoder(
                src_tokens.view(bsz, 1, srclen).expand(bsz, beam_size, srclen).contiguous().view(bsz*beam_size, srclen),
                src_lengths.expand(beam_size, src_lengths.numel()).t().contiguous().view(-1),
                src_keywords.view(b, 1, s, k).expand(b, beam_size, s, k).contiguous().view(b*beam_size, s, k),
            )

@register_model('fconv_fatte_nokey')
class FEncFairseqModel(HKVPredFairseqModel2):
    def __init__(self, docEncoder, docDecoder):
        super().__init__(docEncoder, docDecoder)
        self.docEncoder.encoder.num_attention_layers = sum(layer is not None for layer in docDecoder.decoder.attention)


    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--dropout', default=0.1, type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-embed-path', default=None, type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-layers', type=str, metavar='EXPR',
                            help='encoder layers [(dim, kernel_size), ...]')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-embed-path', default=None, type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-layers', type=str, metavar='EXPR',
                            help='decoder layers [(dim, kernel_size), ...]')
        parser.add_argument('--decoder-out-embed-dim', type=int, metavar='N',
                            help='decoder output embedding dimension')
        parser.add_argument('--decoder-attention', type=str, metavar='EXPR',
                            help='decoder attention [True, ...]')
        parser.add_argument('--normalization-constant', type=float, default=0.5, metavar='D',
                            help='multiplies the result of the residual block by sqrt(value)')
        parser.add_argument('--share-input-output-embed', action='store_true',
                            help='share input and output embeddings (requires'
                                 ' --decoder-out-embed-dim and --decoder-embed-dim'
                                 ' to be equal)')
        parser.add_argument('--hidemb', action='store_true',
                            help='document attn uses hidden or hidden+embeddings')
        parser.add_argument('--normpos', action='store_true',
                            help='normalise addition of sentence and word positional embeddings')
        parser.add_argument('--threshold-tgt-topicscore', type=float, default=0.1, metavar='D',
                        help='minimum score a topic should have to be considered as target topic label')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure that all args are properly defaulted (in case there are any new ones)
        base_architecture(args)


        encoder_embed_dict = None
        if args.encoder_embed_path:
            encoder_embed_dict = utils.parse_embedding(args.encoder_embed_path)
            utils.print_embed_overlap(encoder_embed_dict, task.source_dictionary)

        decoder_embed_dict = None
        if args.decoder_embed_path:
            decoder_embed_dict = utils.parse_embedding(args.decoder_embed_path)
            utils.print_embed_overlap(decoder_embed_dict, task.target_dictionary)

        kw_embed_dict = None
        if args.keywords_embed_path:
            kw_embed_dict = utils.parse_embedding(args.keywords_embed_path)
            print(">> Keywords embeddings loaded!")

        docEncoder = ExtendedEncoder(args,
          FConvEncoder(
            dictionary=task.source_dictionary,
            embed_dim=args.encoder_embed_dim,
            embed_dict=encoder_embed_dict,
            convolutions=eval(args.encoder_layers),
            dropout=args.dropout,
            max_positions=args.max_source_positions,
            normalization_constant=args.normalization_constant,
        ))
        decoder = CondFConvDecoder(
            dictionary=task.target_dictionary,
            embed_dim=args.decoder_embed_dim,
            embed_dict=decoder_embed_dict,
            convolutions=eval(args.decoder_layers),
            out_embed_dim=args.decoder_out_embed_dim,
            attention=eval(args.decoder_attention),
            dropout=args.dropout,
            max_positions=args.max_tgt_sentence_length,
            share_embed=args.share_input_output_embed,
            normalization_constant=args.normalization_constant,
        )

        docDecoder = LDocumentDecoder(args, decoder,
                                      embed_dim=args.decoder_embed_dim,
                                      hidden_size=args.decoder_embed_dim,
                                      out_embed_dim=args.decoder_out_embed_dim,
                                      encoder_embed_dim = args.encoder_embed_dim,
                                      encoder_output_units=args.encoder_embed_dim,
                                      num_layers=1,
                                      attention=eval(args.decoder_attention),
                                      dropout_in = 0.1, dropout_out=0.1,
                                      pretrained_embed = None,
                                      )

        return FEncFairseqModel(docEncoder, docDecoder)

    def max_positions(self):
        """Maximum length supported by the model."""
        print("* max positions encoder {}".format(self.docEncoder.max_positions()))
        print("* max positions decoder {}".format(self.docDecoder.max_positions()))

        return (self.docEncoder.max_positions(), self.docDecoder.max_positions())

    def getNormalizedProbs_Keys(self, keyout):
        #This model does not use Keys
        return None

    def getLambda(self):
        return self.lambda_keyloss

class ExtendedEncoder(FairseqEncoder):
    """ apply the convolutional word sequence encoder to the input sequence  """
    def __init__(self, args, encoder):
        super().__init__(encoder.dictionary)
        self.max_source_positions = args.max_source_positions
        self.encoder = encoder
        self.enc_embed_dim = args.encoder_embed_dim

    def forward(self, src_tokens, src_lengths, src_keywords):

        sout = self.encoder(src_tokens, src_lengths)

        return sout


    def reorder_encoder_out(self, encoder_out_dict, new_order):
        if encoder_out_dict['encoder_out'] is not None:
            encoder_out_dict['encoder_out'] = (
                encoder_out_dict['encoder_out'][0].index_select(0, new_order),
                encoder_out_dict['encoder_out'][1].index_select(0, new_order),
            )
        if encoder_out_dict['encoder_padding_mask'] is not None:
            encoder_out_dict['encoder_padding_mask'] = \
                encoder_out_dict['encoder_padding_mask'].index_select(0, new_order)

        return encoder_out_dict

    def max_positions(self):
        return self.encoder.max_positions()



class LDocumentDecoder(nn.Module):
    """LSTM -based document decoder"""
    def __init__(
        self, args, decoder, embed_dim=512, hidden_size=512, out_embed_dim=512,
        num_layers=1, dropout_in=0.1, dropout_out=0.1, attention=True,
        encoder_embed_dim=512, encoder_output_units=512, pretrained_embed=None,
    ):
        super().__init__()
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.hidden_size = hidden_size
        self.embed_dim = embed_dim
        self.decoder = decoder
        self.max_target_positions = args.max_target_positions
        self.sod_idx = decoder.dictionary.sod()
        self.padding_idx = decoder.dictionary.pad()
        self.encoder_output_units = encoder_output_units
        self.firstfeed = False #args.firstfeed
        self.hidemb = hasattr(args, 'hidemb') and args.hidemb
        self.normpos = hasattr(args, 'normpos') and args.normpos
        if hasattr(args, 'threshold_tgt_topicscore'):
            self.targetTopicThreshold = args.threshold_tgt_topicscore
        else:
            self.targetTopicThreshold = 0.1 #default
        print("*    Atte over hid+emb={} ; sentence-word position normalisation={}".format(self.hidemb, self.normpos))
        assert encoder_output_units == hidden_size, \
            'encoder_output_units ({}) != hidden_size ({})'.format(encoder_output_units, hidden_size)

        self.layers = nn.ModuleList([
            LSTMCell(
                #input_size=encoder_output_units + embed_dim if layer == 0 else hidden_size,
                #TODO: use this because for the moment we dont have input feeding (cat)
                input_size= embed_dim if layer == 0 else hidden_size,
                hidden_size=hidden_size,
            )
            for layer in range(num_layers)
        ])

        # Attn here is based on that defined on lstm.py module
        self.wordAttention = AttentionLayer(encoder_output_units, hidden_size) if attention else None

        if hidden_size != out_embed_dim:
            self.additional_fc = Linear(hidden_size, out_embed_dim, dropout=0.0)

        self.embed_sent_positions = SentencePositionalEmbedding(
            self.max_target_positions + 2, #padding
            embed_dim,
            self.padding_idx,
        )

    def forward(self, decoder_in_dict, encoder_out_dict,
                incremental_state=None, incr_doc_step=False, batch_idxs=None, new_incr_cached=None):

        prev_output_tokens = decoder_in_dict['prev_output_tokens']

        encoder_padding_mask = encoder_out_dict['encoder_padding_mask'] # [b x w]
        if encoder_padding_mask is not None:
            encoder_padding_mask = encoder_padding_mask.transpose(0, 1)
        srcbsz, srcdoclen, srcdim = encoder_out_dict['encoder_out'][0].size()  # [b x w x d]

        # summarise whole input for h0 decoder use, verbose but clearer
        src_summary_h0 =  encoder_out_dict['encoder_out'][0].mean(1) # [b x d]

        bsz, doclen, sentlen = prev_output_tokens.size() # these sizes are target ones

        start_doc = 0
        if incremental_state is not None:
            doclen = 1

        # get initial input embedding for document RNN decoder
        x = prev_output_tokens.data.new(bsz).fill_(self.sod_idx)
        x = self.decoder.embed_tokens(x)

        ## Decode sentence states ##

        # initialize previous states (or get from cache during incremental generation)
        cached_state_rnn = utils.get_incremental_state(self, incremental_state, 'cached_state_rnn')
        if incr_doc_step and cached_state_rnn is not None:
            # doing the fist step of the ith (i>1) sentence in incremental generation
            prev_hiddens, prev_cells, input = cached_state_rnn
            outs = [input]

        elif incremental_state is not None \
                and new_incr_cached is not None: # doing subsequents steps of a sentence in incremental generation
            bidxs, old_bsz, reorder_state = batch_idxs
            if reorder_state is not None: # need to do this when some hypotheses have been finished when generating
                # reducing decoding to lower nb of hypotheses
                new_incr_cached = new_incr_cached.index_select(0, reorder_state)
                outs = [new_incr_cached]
            else:
                outs = [new_incr_cached]
        else:
            # first state of first sentence in incremental generation or
            # or first comming here to generate the whole sentence in trainin/scoring
            # previous is h0 with encoder output summary
            outs = []
            encoder_hiddens_cells =  src_summary_h0 # [b x d]
            prev_hiddens = []
            prev_cells = []
            for i in range(len(self.layers)):
                prev_hiddens.append(encoder_hiddens_cells)
                prev_cells.append(encoder_hiddens_cells)
            input = x

        # attn of document decoder over input aggregated units (e.g. encoded sequence of paragraphs)
        attn_scores = x.data.new(srcdoclen, doclen, bsz).zero_()

        if (incremental_state is not None and incr_doc_step) \
                or incremental_state is None:
            for j in range(start_doc, doclen):
                for i, rnn in enumerate(self.layers):
                    # recurrent cell
                    hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))

                    # hidden state becomes the input to the next layer
                    input = hidden

                    # save state for next time step
                    prev_hiddens[i] = hidden
                    prev_cells[i] = cell

                    # apply attention using the last layer's hidden state (sentence vector)
                if self.wordAttention is not None:

                    # inputs to attention are of the form
                    # input: bsz x input_embed_dim
                    # source_hids: srclen x bsz x output_embed_dim
                    # either attend to the input representation by the cnn encoder or to its combination with the input embeddings
                    if self.hidemb:
                        attn_h, attn_scores_out = self.wordAttention(hidden, \
                                                            encoder_out_dict['encoder_out'][1].transpose(0, 1),\
                                                            encoder_padding_mask)
                    else:
                        attn_h, attn_scores_out = self.wordAttention(hidden, \
                                                                     encoder_out_dict['encoder_out'][0].transpose(0, 1), \
                                                                     encoder_padding_mask)

                    out = attn_h # [b x d]
                else:
                    out = hidden

                # input to next time step
                input = out
                new_incr_cached = out.clone()

                # save final output
                if incremental_state is not None:
                    outs.append(out)
                else:
                    outs.append(out.unsqueeze(1))

        ## Decode sentences ##

        # When training/validation, make all sentence s_t decoding steps in parallel here
        sent_states = None
        if incremental_state is not None:
            dec_outs = x.data.new(bsz, doclen, sentlen, len(self.decoder.dictionary)).zero_()
            # decode by sentence s_j
            for j in range(doclen):

                sp = self.embed_sent_positions(decoder_in_dict['sentence_position'])

                dec_out, word_atte_scores = self.decoder(
                    prev_output_tokens[:, j, :], outs[j], sp, encoder_out_dict, incremental_state,
                    firstfeed=self.firstfeed, normpos=self.normpos)
                # prev_output_tokens is [ b x s x w ], at each time step decode sentence j [b x w]
                # dec_out is [b x w x vocabulary]
                if j == 0:
                    dec_outs = dec_out
                else:
                    dec_outs = torch.cat((dec_outs, dec_out), 1)
                    # dec_outs is [bxs x w x  vocabulary], dim=0
                    # dec_outs is [b x s*w x  vocabulary], dim=1

        else:
            # decode everything in parallel
            sent_states = torch.cat(outs, dim=1).view(bsz*doclen, -1)
            ys = prev_output_tokens.view(bsz*doclen, -1)
            sp = make_sent_positions(prev_output_tokens, self.padding_idx).view(bsz*doclen)
            sp = self.embed_sent_positions(sp)

            # Replicate encoder_out_dict for the new nb of batches to do all in parallel

            ebsz, esrclen, edim = encoder_out_dict['encoder_out'][0].size()
            new_enc_out_dict = {}
            #repeat input for each target
            new_enc_out_dict['encoder_out'] = (
                encoder_out_dict['encoder_out'][0].view(ebsz, 1, esrclen, edim).expand(ebsz, doclen, esrclen, edim)
                                        .contiguous().view(ebsz*doclen, esrclen, edim),
                encoder_out_dict['encoder_out'][1].view(ebsz, 1, esrclen, edim).expand(ebsz, doclen, esrclen, edim)
                                        .contiguous().view(ebsz*doclen, esrclen, edim)
            )

            new_enc_out_dict['encoder_padding_mask'] = None
            if encoder_out_dict['encoder_padding_mask'] is not None:
                new_enc_out_dict['encoder_padding_mask'] = encoder_out_dict['encoder_padding_mask']\
                                                    .view(ebsz, 1, esrclen).expand(ebsz, doclen, esrclen)\
                                                    .contiguous().view(ebsz*doclen, -1)

            #decode all target sentences of all documents in parallel
            dec_out, word_atte_scores = self.decoder(ys, sent_states, sp, new_enc_out_dict,
                                                     firstfeed=self.firstfeed, normpos=self.normpos)
            dec_outs = dec_out.view(bsz, doclen*sentlen, len(self.decoder.dictionary))

        if incremental_state is not None and incr_doc_step:
            # only if we moved to the next document sentence
            # cache previous states (no-op except during incremental generation)
               utils.set_incremental_state(
               self, incremental_state, 'cached_state_rnn', (prev_hiddens, prev_cells, out))

        # srclen x tgtlen x bsz -> bsz x tgtlen x srclen
        attn_scores = attn_scores.transpose(0, 2)

        return dec_outs, (attn_scores, word_atte_scores), new_incr_cached, None # topic label predictions are None here


    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return (self.max_target_positions, self.decoder.max_positions())

    def getCached(self, incremental_state, key):
        x = utils.get_incremental_state(self, incremental_state, key)
        return x

    def setCached(self, incremental_state, key, value):
        utils.set_incremental_state(self, incremental_state, key, value)

    def reorder_incremental_state(self, incremental_state, new_order):

        def apply_reorder_incremental_state(module):
            if module != self and hasattr(module, 'reorder_incremental_state'):
                module.reorder_incremental_state(
                    incremental_state,
                    new_order,
                )
        self.apply(apply_reorder_incremental_state)

        # document decoder
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state_rnn')
        if cached_state is None:
            return

        def reorder_state(state):
            if isinstance(state, list):
                return [reorder_state(state_i) for state_i in state]
            return state.index_select(0, new_order)

        new_state = tuple(map(reorder_state, cached_state))

        utils.set_incremental_state(self, incremental_state, 'cached_state_rnn', new_state)


    def upgrade_state_dict(self, state_dict):
        return state_dict

    def get_normalized_probs(self, net_output, log_probs, _):
        raise NotImplementedError
    # here should call to the set decoder or not implemented

    def getTargetTopicThreshold(self):
        return self.targetTopicThreshold


class CondFConvDecoder(FConvDecoder):
    """Based on Convolutional decoder extends with sentence positions and sentence vectors"""

    def forward(self, prev_output_tokens, h0, sent_pos_emb, encoder_out_dict=None, incremental_state=None, firstfeed=False, normpos=False):
        if encoder_out_dict is not None:
            encoder_out = encoder_out_dict['encoder_out']
            encoder_padding_mask = encoder_out_dict['encoder_padding_mask']

            # split and transpose encoder outputs
            encoder_a, encoder_b = self._split_encoder_out(encoder_out, incremental_state)

        if self.embed_positions is not None:
            pos_embed = self.embed_positions(prev_output_tokens, incremental_state)
            if incremental_state is not None:
                pos_embed += sent_pos_emb
            else:
                pos_embed += sent_pos_emb.unsqueeze(1).repeat(1, pos_embed.size(1), 1) #debuged this expansion is ok
            if normpos:
                pos_embed *= math.sqrt(self.normalization_constant)
        else:
            pos_embed = 0

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
        x = self._embed_tokens(prev_output_tokens, incremental_state) #[b x w x d]

        # embed tokens and combine with positional embeddings
        x += pos_embed
        x = F.dropout(x, p=self.dropout, training=self.training)
        target_embedding = x

        # project to size of convolution
        x = self.fc1(x)

        # B x T x C -> T x B x C
        x = self._transpose_if_training(x, incremental_state)

        # temporal convolutions
        avg_attn_scores = None
        num_attn_layers = len(self.attention)
        residuals = [x]
        ln = 0
        for proj, conv, attention, res_layer in zip(self.projections, self.convolutions, self.attention,
                                                    self.residuals):
            if res_layer > 0:
                residual = residuals[-res_layer]
                residual = residual if proj is None else proj(residual)
            else:
                residual = None

            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, incremental_state)
            x = F.glu(x, dim=2)

            # attention
            if attention is not None:
                x = self._transpose_if_training(x, incremental_state)

                # x [ b x w x dim ]
                # target_embedding [ b x w x dim ]
                # encoder_a [ b x dim x s*w ]
                # encoder_b [ b x s*w x dim ]
                # encoder_padding_mask [ b x s x w ]


                if (ln == 0 and firstfeed) or (not firstfeed) :
                    q = (x + h0.unsqueeze(1).repeat(1, x.size(1), 1)) * math.sqrt(self.normalization_constant)
                else:
                    q = x

                x, attn_scores = attention(q, target_embedding, (encoder_a, encoder_b), encoder_padding_mask)
                attn_scores = attn_scores / num_attn_layers
                if avg_attn_scores is None:
                    avg_attn_scores = attn_scores
                else:
                    avg_attn_scores.add_(attn_scores)

                x = self._transpose_if_training(x, incremental_state)

            # residual
            if residual is not None:
                x = (x + residual) * math.sqrt(self.normalization_constant)
            residuals.append(x)
            ln +=1

        # T x B x C -> B x T x C
        x = self._transpose_if_training(x, incremental_state)

        # project back to size of vocabulary if not using adaptive softmax
        if self.fc2 is not None and self.fc3 is not None:
            x = self.fc2(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.fc3(x)

        return x, avg_attn_scores

    def upgrade_state_dict(self, state_dict):
        if state_dict.get('docDecoder.decoder.version', torch.Tensor([1]))[0] < 2:
            # old models use incorrect weight norm dimension
            for i, conv in enumerate(self.convolutions):
                # reconfigure weight norm
                nn.utils.remove_weight_norm(conv)
                self.convolutions[i] = nn.utils.weight_norm(conv, dim=0)
            state_dict['docDecoder.decoder.version'] = torch.Tensor([1])
        return state_dict


def SentencePositionalEmbedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, 0, 0.1)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m

def make_sent_positions(tensor, padding_idx, incremental_state=None, current_sent=None):
    """Replace non-padding symbols with their position numbers.

    Position numbers begin at padding_idx+1.
    """
    max_pos = padding_idx + 1 + tensor.size(1)
    if not hasattr(make_sent_positions, 'range_buf_sent'):
        make_sent_positions.range_buf_sent = tensor.new()
    make_sent_positions.range_buf_sent = make_sent_positions.range_buf_sent.type_as(tensor)
    if make_sent_positions.range_buf_sent.numel() < max_pos:
        torch.arange(padding_idx + 1, max_pos, out=make_sent_positions.range_buf_sent)
    mask = tensor.ne(padding_idx).sum(2)
    mask = mask.ne(0)

    positions = make_sent_positions.range_buf_sent[:mask.size(1)].expand_as(mask)

    if incremental_state is not None:
        positions = tensor.new(mask.size()).fill_(1).masked_fill_(mask, current_sent)
        return positions
    else:
        return tensor.new(mask.size()).fill_(1).masked_scatter_(mask, positions[mask])


@register_model_architecture('fconv_fatte_nokey', 'fconv_fatte_nokey_nokey')
def base_architecture(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_layers = getattr(args, 'encoder_layers', '[(512, 3)] * 20')
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_layers = getattr(args, 'decoder_layers', '[(512, 3)] * 20')
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 256)
    args.decoder_attention = getattr(args, 'decoder_attention', 'True')
    args.share_input_output_embed = getattr(args, 'share_input_output_embed', False)
    args.normalization_constant = getattr(args, 'normalization_constant', 0.5)


@register_model_architecture('fconv_fatte_nokey', 'fconv_fatte_nokey_wikicatsum_big')
def fconv_wikicatsum(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    layers = '[(850, 6)] * 3'
    layers += ' + [(850, 1)] * 1'
    layers += ' + [(850, 5)] * 4'
    layers += ' + [(850, 1)] * 1'
    layers += ' + [(850, 4)] * 3'
    layers += ' + [(1024, 4)] * 1'
    layers += ' + [(2048, 4)] * 1'
    args.encoder_layers = getattr(args, 'encoder_layers', layers)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_layers = getattr(args, 'decoder_layers', '[(512, 3)] * 6')
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 256)
    args.decoder_attention = getattr(args, 'decoder_attention', 'True')
    args.share_input_output_embed = getattr(args, 'share_input_output_embed', False)
    args.normalization_constant = getattr(args, 'normalization_constant', 0.5)


@register_model_architecture('fconv_fatte_nokey', 'fconv_fatte_nokey_iwslt_de_en')
def fconv_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.encoder_layers = getattr(args, 'encoder_layers', '[(256, 3)] * 4')
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
    args.decoder_layers = getattr(args, 'decoder_layers', '[(256, 3)] * 3')
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 256)
    base_architecture(args)

@register_model_architecture('fconv_fatte_nokey', 'fconv_fatte_nokey_wikicatsum')
def fconv_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.encoder_layers = getattr(args, 'encoder_layers', '[(256, 3)] * 4')
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
    args.decoder_layers = getattr(args, 'decoder_layers', '[(256, 3)] * 3')
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 256)
    base_architecture(args)