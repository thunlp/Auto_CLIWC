# Auto LIWC
The code for **Chinese LIWC Lexicon Expansion via Hierarchical Classification
of Word Embeddings with Sememe Attention** (AAAI18).

## Datasets
This folder `datasets` contains two datasets.

1. `HowNet.txt` is an Chinese knowledge base with annotated word-sense-sememe information.
2. `sc_liwc.dic` is the Chinese LIWC lexicon. This is revised version of
the original C-LIWC file. Because the original contains part of speech (POS)
categories such as _verb_, _adverb_, and _auxverb_, we believe it is
more accurate to utilize POS tagging programs when conducting text analysis
in a given text, so we delete POS categories in our experiment. Furthermore,
the hierarchical structure is slightly different from the original English
version of LIWC, so we altered the hierarchical structure based on the English LIWC.

Please note that the above datasets files are for academic and educational use **only**.
They are **not** for commercial use. If you have any questions, please contact us first
before downloading the datasets.

Due to the large size of the embedding file, we can only release the code for training
the word embeddings. Please see `word2vec.py` for details.

## Run
Run the following command for training and testing:

`python3 train_liwc.py`

If the datasets are in a different folder, please change the path
[here](https://github.com/thunlp/Auto_CLIWC).

The current code generates different training and testing set every time.
To reproduce the results in the paper, you can load `train.bin` and `test.bin`
located in `bin_data` using `pickle`.

## Dependencies

- Tensorflow == 1.4.0
- Scipy == 0.19.0
- Numpy == 1.13.1
- Scikit-learn == 0.18.1
- Gensim == 2.0.0

## Cite
If you use the code, please cite this paper:

_Xiangkai Zeng, Cheng Yang, Cunchao Tu, Zhiyuan Liu, Maosong Sun.
Chinese LIWC Lexicon Expansion via Hierarchical Classification of Word
Embeddings with Sememe Attention. The 32nd AAAI Conference on Artificial
Intelligence (AAAI 2018)._