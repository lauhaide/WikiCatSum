#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

# we added decoding constraints along the lines:
# https://arxiv.org/pdf/1705.04304.pdf
# http://david.grangier.info/papers/2018/grangier-quickedit-editing-texts-2018.pdf

# extension for hierarchical decoding

import torch
import os

from fairseq import bleu, data, options, progress_bar, tasks, tokenizer, utils
from fairseq.meters import StopwatchMeter, TimeMeter
from fairseq.sequence_generator import SequenceGenerator
from fairseq.sequence_generatorWCS import SequenceGeneratorWCS
from fairseq.sequence_generatorWCSSepahypo import SequenceGeneratorWCSSepahypo
from fairseq.sequence_scorer import SequenceScorer

def make_html_safe(s):
    """Rouge use html, has to make output html safe"""
    return s.replace("<", "&lt;").replace(">", "&gt;")

def processFlatHypo(sample_id, src_tokens, target_tokens, hypos,
                    src_str, align_dict, tgt_dict, remove_bpe, has_target, target_str):
    """Not used"""
    for i, hypo in enumerate(hypos[:min(len(hypos), args.nbest)]):
        hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
            hypo_tokens=hypo['tokens'].int().cpu(),
            src_str=src_str,
            alignment=hypo['alignment'].int().cpu(),
            align_dict=align_dict,
            tgt_dict=tgt_dict,
            remove_bpe=remove_bpe,
        )

        if not args.quiet:
            print('H-{}\t{}\t{}'.format(sample_id, hypo['score'], hypo_str))
            print('P-{}\t{}'.format(
                sample_id,
                ' '.join(map(
                    lambda x: '{:.4f}'.format(x),
                    hypo['positional_scores'].tolist(),
                ))
            ))
            print('A-{}\t{}'.format(
                sample_id,
                ' '.join(map(lambda x: str(utils.item(x)), alignment))
            ))

        # Score only the top hypothesis
        if has_target and i == 0:
            if align_dict is not None or args.remove_bpe is not None:
                # Convert back to tokens for evaluation with unk replacement and/or without BPE
                target_tokens = tokenizer.Tokenizer.tokenize(
                    target_str, tgt_dict, add_if_not_exist=True)

        # write files for ROUGE
        with open(os.path.join(args.decode_dir, "{}.dec".format(sample_id)), 'w') as f:
            f.write(make_html_safe(hypo_str))
            f.close()



def main(args):
    assert args.path is not None, '--path required for generation!'
    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'

    if args.max_tokens is None and args.max_sentences is None:
        args.max_tokens = 12000
    print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)
    print('| {} {} {} examples'.format(args.data, args.gen_subset, len(task.dataset(args.gen_subset))))

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    models, _ = utils.load_ensemble_for_inference(args.path.split(':'), task)

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(beamable_mm_beam_size=None if args.no_beamable_mm else args.beam)
        if args.fp16:
            model.half()

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    ignoredIndices = []
    if args.outindices:
        f=open(args.outindices, 'r')
        for line in f.readlines():
            ignoredIndices.append(int(line.strip()))
    print("{} indices to be ignored from validation set.".format(len(ignoredIndices)))

    # Load dataset (possibly sharded)
    itr = data.EpochBatchIterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=models[0].max_positions(),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=8,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        savedir=os.path.join(args.decode_dir, "valid_"),
        ignoredIndices=ignoredIndices,
    ).next_epoch_itr(shuffle=False)

    # Initialize generator
    gen_timer = StopwatchMeter()
    if args.score_reference:
        translator = SequenceScorer(models, task.target_dictionary)
    elif args.sepahypo:
        translator = SequenceGeneratorWCSSepahypo(
            models, task.target_dictionary, beam_size=args.beam,
            stop_early=(not args.no_early_stop), normalize_scores=(not args.unnormalized),
            len_penalty=args.lenpen, unk_penalty=args.unkpen,
            sampling=args.sampling, sampling_topk=args.sampling_topk, minlen=args.min_len,
            maxlen=None, context=args.context, ngram=args.ngram, naive=args.naive,
            num_topics=args.num_topics, flatenc=args.flatenc, flatten_source=args.flatten_source,
            cov_penalty=args.covpen, keystop=args.keystop,
        )
    elif args.flatdec:
        translator = SequenceGenerator(
            models, task.target_dictionary, beam_size=args.beam,
            stop_early=(not args.no_early_stop), normalize_scores=(not args.unnormalized),
            len_penalty=args.lenpen, unk_penalty=args.unkpen,
            sampling=args.sampling, sampling_topk=args.sampling_topk, minlen=args.min_len,
            flatdec=True,
        )
    else:
        translator = SequenceGeneratorWCS(
            models, task.target_dictionary, beam_size=args.beam,
            stop_early=(not args.no_early_stop), normalize_scores=(not args.unnormalized),
            len_penalty=args.lenpen, unk_penalty=args.unkpen,
            sampling=args.sampling, sampling_topk=args.sampling_topk, minlen=args.min_len,
            maxlen=None, context=args.context, ngram=args.ngram, num_topics=args.num_topics,
            flatenc=args.flatenc, dechatt=args.dechatt, flatten_source=args.flatten_source,
        )

    if use_cuda:
        translator.cuda()

    # Generate and compute BLEU score
    scorer = bleu.Scorer(tgt_dict.pad(), tgt_dict.eos(), tgt_dict.unk())
    num_sentences = 0
    has_target = True
    outlog = open(args.decode_dir+'/out.log','w', encoding='utf8')
    print("* Generating target texts of max length proportional to b: {} (ax+b)".format(args.max_len_b))
    with progress_bar.build_progress_bar(args, itr) as t:
        if args.score_reference:
            translations = translator.score_batched_itr(t, cuda=use_cuda, timer=gen_timer)
        else:
            translations = translator.generate_batched_itr(
                t, maxlen_a=args.max_len_a, maxlen_b=args.max_len_b,
                cuda=use_cuda, timer=gen_timer, prefix_size=args.prefix_size,
            )

        wps_meter = TimeMeter()
        for sample_id, src_tokens, target_tokens, hypos in translations: # for each batch
            # Process input and ground truth
            has_target = target_tokens is not None
            target_tokens = target_tokens.int().cpu() if has_target else None

            # Either retrieve the original sentences or regenerate them from tokens.
            target_str = None
            if align_dict is not None and args.raw_text:
                src_str = task.dataset(args.gen_subset).src.get_original_text(sample_id)
                target_str = task.dataset(args.gen_subset).tgt.get_original_text(sample_id)
            else:
                src_str = src_dict.string(src_tokens, args.remove_bpe)
                if has_target and args.target_raw_text:
                    target_str_tok = tgt_dict.string(target_tokens, args.remove_bpe, escape_unk=True)
                    target_str = task.dataset(args.gen_subset).get_target_original_text(sample_id)

            # Process top predictions
            if args.flatdec:
                processFlatHypo(sample_id, src_tokens, target_tokens, hypos,
                                src_str, align_dict, tgt_dict, args.remove_bpe, has_target, target_str)
            else:
                for j in range(min(len(hypos), args.nbest)): # for each beam
                  doc_hypo_tokens = []
                  doc_hypo_str = []
                  doc_target_str = []

                  for i in range(len(hypos[j]['beam'])): # for each sentence of the beam
                    hypo = hypos[j]['beam'][i]
                    hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                        hypo_tokens=hypo['tokens'].int().cpu(),
                        src_str=src_str,
                        alignment=hypo['alignment'].int().cpu(),
                        align_dict=align_dict,
                        tgt_dict=tgt_dict,
                        remove_bpe=args.remove_bpe,
                    )

                    if not args.quiet:
                        print('H({})-{}\t{}\t{}'.format(j, sample_id, hypo['score'], hypo_str))
                        print('P({})-{}\t{}'.format(j,
                            sample_id,
                            ' '.join(map(
                                lambda x: '{:.4f}'.format(x),
                                hypo['positional_scores'].tolist(),
                            ))
                        ))
                        print('A({})-{}\t{}'.format(j,
                            sample_id,
                            ' '.join(map(lambda x: str(utils.item(x)), alignment))
                        ))

                    subhypo = False
                    tokens_curhypo = set(hypo_str.split())
                    for hyp in doc_hypo_str:
                        tokens_hyp = set(hyp.split())

                        # if its contained in previous sentence hypothesis
                        if hypo_str.strip()[0:-1] in hyp:
                            subhypo = True
                            break

                        shorter = len(tokens_curhypo)

                        # if it overlaps on more than 80% of its tokens
                        shorter = round(shorter * 0.8)
                        if len(tokens_curhypo.intersection(tokens_hyp)) >= shorter:
                            subhypo = True

                    if not (hypo_str in doc_hypo_str or  subhypo):
                        doc_hypo_str.append(hypo_str)
                    else:
                        print("repeated on {} / {}".format(sample_id, i))
                        print(hypo_str)

                    if has_target and i == 0:
                        doc_hypo_tokens.append(hypo_tokens)

                #write files for ROUGE
                with open(os.path.join(args.decode_dir,"{}.dec".format(sample_id)),'w') as f:
                      f.write(make_html_safe(" ".join(doc_hypo_str).replace(tgt_dict.eod_word, "").strip()))
                      f.close()

                #TODO: call scorer for BLEU

                if target_str:
                    doc_target_str.append(target_str)
                    with open(os.path.join(args.reference_dir,"{}.ref".format(sample_id)),'w') as f:
                        f.write(make_html_safe(" ".join(doc_target_str)))
                        f.close()
                    with open(os.path.join(args.reference_dir + "_fromdict", "{}.ref".format(sample_id)),'w') as f:
                        f.write(make_html_safe(target_str_tok))
                        f.close()
                outlog.write("[{}] ".format(sample_id) + " ".join(doc_hypo_str).replace(tgt_dict.eod_word, "").strip() + "\n")

            wps_meter.update(src_tokens.size(0))
            t.log({'wps': round(wps_meter.avg)})
            num_sentences += 1

    outlog.close()

    print('| Translated {} sentences ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)'.format(
        num_sentences, gen_timer.n, gen_timer.sum, num_sentences / gen_timer.sum, 1. / gen_timer.avg))

def generate_from_script(list_args):
    parser = options.get_generation_parser()
    group = parser.add_argument_group('Generation output')
    group.add_argument('--decode-dir', metavar='DIR', default='outputs',
                   help='path to save predictions')
    group.add_argument('--reference-dir', metavar='DIR', default='outputs/reference/valid',
                   help='path to save predictions')
    group.add_argument('--usekeys', action='store_true',
                   help='whether to use target key prediction')
    group.add_argument('--context', action='store_true',
                   help='whether to use previous sentences as context for current sentence decoding')
    group.add_argument('--ngram', type=int, default=0,
                   help='whether to use hard constrains on ngram repetition when decoding')
    group.add_argument('--sepahypo', action='store_true',
                   help='decode sentence hypothesis independently. sort best for each sentence.')
    group.add_argument('--naive', action='store_true',
                   help='decode sentence hypothesis independently. sort best for each sentence.')
    parser.add_argument('--outindices', required=False,type=str,
                            help='load set of indices that were out for a category dataset.')
    parser.add_argument('--covpen', type=float, default=0, metavar='D',
                        help='coverage penalty (Gehrmann et al. 2018).')
    group.add_argument('--keystop', action='store_true',
                   help='whether to use topic prediction to spot EndOfDocumet. Makes only sense '
                        'with models using topic-key-prediction')

    args = options.parse_args_and_arch(parser, list_args)

    if not os.path.isdir(args.decode_dir):
        os.mkdir(args.decode_dir)

    main(args)


if __name__ == '__main__':
    parser = options.get_generation_parser()
    group = parser.add_argument_group('Generation output')
    group.add_argument('--decode-dir', metavar='DIR', default='outputs',
                   help='path to save predictions')
    group.add_argument('--reference-dir', metavar='DIR', default='outputs/reference/valid',
                   help='path to save predictions')
    group.add_argument('--usekeys', action='store_true',
                   help='whether to use target key prediction')
    group.add_argument('--context', action='store_true',
                   help='whether to use previous sentences as context for current sentence decoding')
    group.add_argument('--ngram', type=int, default=0,
                   help='whether to use hard constrains on ngram repetition when decoding')
    group.add_argument('--sepahypo', action='store_true',
                   help='decode sentence hypothesis independently. sort best for each sentence.')
    group.add_argument('--naive', action='store_true',
                   help='decode sentence hypothesis independently. sort best for each sentence.')
    parser.add_argument('--outindices', required=False,type=str,
                            help='load set of indices that were out for a category dataset.')
    parser.add_argument('--covpen', type=float, default=0, metavar='D',
                        help='coverage penalty (Gehrmann et al. 2018).')
    group.add_argument('--keystop', action='store_true',
                   help='whether to use topic prediction to spot EndOfDocumet. Makes only sense '
                        'with models using topic-key-prediction')


    args = options.parse_args_and_arch(parser)

    if not os.path.isdir(args.decode_dir):
        os.mkdir(args.decode_dir)

    main(args)
