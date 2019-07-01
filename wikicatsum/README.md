# WikiCatSum
Abstractive Summarisation, generating Wikipedia lead sections.


## Dataset

The WikiCatSum dataset can be downloaded from [this](https://datashare.is.ed.ac.uk/handle/10283/3368) repository. In the paper we used the *Company*, *Film*, and *Animal* domains.

We hope to extract additional domains: Athlete, Building, Musical-Work


## Topic Templates from WikiCatSum summaries

Creating sentence-level LDA topic models of WikiCatSum summaries.

You can use the following to perform grid search on the number of topics [10, ..., 90] every ten steps:
```
python BuildLDATopics.py \
--corpus wikicatsum/dataset/film_tok_min5_L7.5k/train.bow.tgt \
--outdir wikicatsum/dataset/film_tok_min5_L7.5k-bin \
--sent-level --search-num-topics --bowpreproc 
```

Example command to generating topic model:
```
python BuildLDATopics.py --num-topics 30 --sent-level --bowpreproc \
--corpus wikicatsum/dataset/animal_tok_min5_L7.5k/train.bow.tgt \
--outdir wikicatsum/dataset/animal_tok_min5_L7.5k-bin/ \
--savemodel animal.30.TLDA
```

