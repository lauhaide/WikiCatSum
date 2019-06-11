
import numpy as np
import argparse
import os
from os.path import join, exists

from fairseq.data.indexed_dataset import RawTextDataset
from rouge import _get_ngrams

try:
    _DATA_DIR = os.environ['DATADIR']
except KeyError:
    print('please use environment variable to specify data directories')


missingTok = 'UNK'
unkConv = '&lt;unk&gt;'

def loadOutputs(dir):
    d = {}
    for filename in os.listdir(dir):
        if filename.endswith(".dec"):
            f = open(os.path.join(dir, filename),'r')
            d[int(filename.split(".dec")[0])] = f.read().strip().replace(unkConv, missingTok)
    print("Finish reading system outputs...")
    return d


from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
stop_words.remove('own')

def metrics(input, eval_sentence, ref_sentence, n=1):
  """Computes Abstraction and Copy precision & recall scores for an input set of paragraphs, generated summary, and gold .

  Args:
    input: The input sequence of paragraphs
    eval_sentences: The generated summary
    ref_sentences: The gold summary
    n: Size of ngram.  Defaults to 1.

  Returns:
    recall & precision scores
  """
  def remove_stopwords(texts):
      return [word for word in texts if word not in stop_words]

  eval_ngrams = _get_ngrams(n, remove_stopwords(eval_sentence.split()))
  ref_ngrams = _get_ngrams(n, remove_stopwords(ref_sentence.split()))
  in_ngrams = _get_ngrams(n, remove_stopwords(input.split()))

  # abstraction
  evalInputComplement = eval_ngrams - in_ngrams
  refInputComplement = ref_ngrams - in_ngrams
  overlapping_onlyToRef = refInputComplement.intersection(evalInputComplement)

  abs_p = len(overlapping_onlyToRef) / len(evalInputComplement) if evalInputComplement else 0.0
  abs_r = len(overlapping_onlyToRef) / len(refInputComplement) if refInputComplement else 0.0

  # copy
  eval_overlapping_ngrams = eval_ngrams.intersection(in_ngrams)
  ref_overlapping_ngrams = ref_ngrams.intersection(in_ngrams)
  all_overlapping_ngrams = ref_overlapping_ngrams.intersection(eval_ngrams)

  copy_p = len(all_overlapping_ngrams) / len(eval_overlapping_ngrams) if eval_overlapping_ngrams else 0.0
  copy_r = len(all_overlapping_ngrams) / len(ref_overlapping_ngrams) if ref_overlapping_ngrams else 0.0

  return abs_p, abs_r, copy_p, copy_r


def doEval(decode_dir, dataset):
    """Computes Abstraction and Copy F1 score"""

    data_dir = _DATA_DIR
    assert exists(data_dir)

    srcTexts = RawTextDataset(os.path.join(data_dir, dataset + ".src"))
    refTexts = RawTextDataset(os.path.join(data_dir, dataset + ".tgt"))
    sysTexts = loadOutputs(decode_dir)
    print("Extra statistics for {} ({} cases)".format(dataset, len(sysTexts.keys())))

    absPrecision = []
    absRecall = []
    absF1 = []
    copyPrecision = []
    copyRecall = []
    copyF1 = []
    for instance in sysTexts.keys():
        srci = srcTexts.get_original_text(instance)
        tgti = refTexts.get_original_text(instance)

        # compute metrics
        abs_p, abs_r, copy_p, copy_r = metrics(srci, sysTexts[instance], tgti)
        absPrecision.append(abs_p)
        absRecall.append(abs_r)
        copyPrecision.append(copy_p)
        copyRecall.append(copy_r)

        absF1.append(2.0 * ((abs_p * abs_r) / (abs_p + abs_r + 1e-8)))
        copyF1.append(2.0 * ((copy_p * copy_r) / (copy_p + copy_r + 1e-8)))

    out = []
    out.append("Abstraction precision: {}".format(np.mean(absPrecision, dtype=np.float32)))
    out.append("Abstraction recall: {}".format(np.mean(absRecall, dtype=np.float32)))
    out.append("Abstraction F1: {}".format(np.mean(absF1, dtype=np.float32)))
    out.append("Copy precision: {}".format(np.mean(copyPrecision, dtype=np.float32)))
    out.append("Copy recall: {}".format(np.mean(copyRecall, dtype=np.float32)))
    out.append("Copy F1: {}".format(np.mean(copyF1, dtype=np.float32)))
    return "\n".join(out)

def main(args):
    output = doEval(args.decode_dir, args.dataset)
    print(output)
    fileout = open(join(args.decode_dir, 'extra_metrics.txt'), 'w')
    fileout.write(output)
    fileout.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate the output files with additional metrics')

    parser.add_argument('--decode-dir', action='store', required=True,
                        help='directory of decoded summaries')
    parser.add_argument('--dataset', default='test', help='partition to run on')

    args = parser.parse_args()
    main(args)
