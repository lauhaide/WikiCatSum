""" Evaluate the baselines ont ROUGE/METEOR"""
import argparse
import json
import os
from os.path import join, exists

from evaluation.evaluate import eval_meteor, eval_rouge


try:
    _DATA_DIR = os.environ['REFDIR']
except KeyError:
    print('please use environment variable to specify data directories')

def doEval(decode_dir, metric):
    dec_dir = decode_dir #join(args.decode_dir, 'output')
    #with open(join(args.decode_dir, 'log.json')) as f:
    #    split = json.loads(f.read())['split']
    ref_dir = _DATA_DIR #join(_DATA_DIR, 'refs', split)
    assert exists(ref_dir)

    if metric=='rouge':
        dec_pattern = r'(\d+).dec'
        ref_pattern = '#ID#.ref'
        output = eval_rouge(dec_pattern, dec_dir, ref_pattern, ref_dir)
    else:
        dec_pattern = '[0-9]+.dec'
        ref_pattern = '[0-9]+.ref'
        output = eval_meteor(dec_pattern, dec_dir, ref_pattern, ref_dir)

    return output

def main(args):
    metric = 'rouge' if args.rouge else 'meteor'
    output = doEval(args.decode_dir, metric)
    print(output)
    refname = _DATA_DIR.split('/')
    refname = refname[len(refname)-1]
    fileout = open(join(args.decode_dir, '{}_{}.txt'.format(metric, refname)), 'w')
    fileout.write(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate the output files for the RL full models')

    # choose metric to evaluate
    metric_opt = parser.add_mutually_exclusive_group(required=True)
    metric_opt.add_argument('--rouge', action='store_true',
                            help='ROUGE evaluation')
    metric_opt.add_argument('--meteor', action='store_true',
                            help='METEOR evaluation')

    parser.add_argument('--decode_dir', action='store', required=True,
                        help='directory of decoded summaries')

    args = parser.parse_args()
    main(args)
