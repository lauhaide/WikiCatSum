"""Laura Perez"""

from __future__ import division, unicode_literals
import os
import re
import argparse
import shutil

from my_generate2 import generate_from_script
from my_generateSingle import generate_from_script_single
from evaluation.eval_full_model import doEval


def run_generation(fconv, ARGS_LIST, MODEL, DECODEDIR, data_dir):
    if fconv:
        generate_from_script_single([data_dir, '--path', MODEL, '--decode-dir', DECODEDIR, '--gen-subset', 'valid'] + ARGS_LIST )
    else:
        generate_from_script([args.data_dir, '--path', MODEL, '--decode-dir', DECODEDIR, '--gen-subset', 'valid'] + ARGS_LIST )



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Evaluate rouge on different epochs')

    # choose metric to evaluate
    parser.add_argument('--data-dir', help='binary files directory')
    parser.add_argument('--model-dir', help='directory where model epochs are saved')
    parser.add_argument('--reference-dir', action='store', required=True,
                        help='directory where to find references')
    parser.add_argument('--fconv', action='store_true',
                        help='Single-sequence encoder decoder')
    parser.add_argument('--outpath', help='Base directory where to store system outputs')

    args = parser.parse_args()

    ARG_LIST = os.getenv('ARG_LIST').split()

    def doFile(filename, model_dir):
        epoch = filename.split("checkpoint")[1].split(".pt")[0]
        skips = [str(i) for i in range(1,11)]
        if epoch in skips :
            model_name = os.path.join(model_dir, filename)
            if os.path.isfile(model_name):
                print("removing model epochs...",filename)
                os.remove(model_name) #remove the fist epoch files
        return epoch not in (skips + ["_last"] + ["_best"] )

    outpath = args.outpath
    f = open(os.path.join(args.model_dir, 'best_rouge.txt'), 'w')
    max_rouge_1 = 0.
    max_rouge_2 = 0.
    max_rouge_L = 0.
    best_model = ''
    for filename in os.listdir('./'+str(args.model_dir)):
        if filename.endswith(".pt") and doFile(filename, args.model_dir):
            DECODEDIR = os.path.join(outpath,args.model_dir.split('checkpoints/')[1])
            DECODEDIR = DECODEDIR +'_'+ filename.split('.pt')[0]
            MODEL = os.path.join(args.model_dir, filename)
            print(MODEL)
            run_generation(args.fconv, ARG_LIST, MODEL, DECODEDIR, args.data_dir)

            output = doEval(DECODEDIR, 'rouge')

            curr_rouge_1 = float(re.search('ROUGE-1 Average_F: (.+?) \(', output).group(1))
            curr_rouge_2 = float(re.search('ROUGE-2 Average_F: (.+?) \(', output).group(1))
            curr_rouge_L = float(re.search('ROUGE-L Average_F: (.+?) \(', output).group(1))

            print(filename, "{} {} {}".format(curr_rouge_1, curr_rouge_2, curr_rouge_L))

            if curr_rouge_1 > max_rouge_1 and curr_rouge_2 > max_rouge_2 and curr_rouge_L > max_rouge_L:
                f.write(output)
                max_rouge_1 = curr_rouge_1
                max_rouge_2 = curr_rouge_2
                max_rouge_L = curr_rouge_L
                if best_model != '':
                    # dont remove the model as we may want to run again
                    shutil.rmtree(best_model_outdir)
                best_model = MODEL
                best_model_outdir = DECODEDIR
            else:
                # dont remove the model as we may want to run again
                shutil.rmtree(DECODEDIR, ignore_errors=True)

    results = 'The best model is:\n ' + best_model + \
        ' with Average_F rouge:\n ROUGE-1: {} ROUGE-2: {} ROUGE-L:{}'.format(max_rouge_1, max_rouge_2, max_rouge_L)
    print(results)
    f.write("\n\n"+results)
