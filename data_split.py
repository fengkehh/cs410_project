from base import *
import math
import metapy

# Data splitter: create two sets of data from one large set of data specified by config file at config_path using random index, one set with guaranteed size of roughly and at least frac
# @param config_path: path to the corpus config file
# @frac: float indicating the fraction of data points contained in one of the sets
# @return: (set1, set2), set1 follows frac, set2 is just the complement. Make set1 bigger if doesn't divide perfectly.
# def data_split(config_path, frac):
#     config = parse_config(config_path)
#     corpus_path = config['prefix'] + '/' + config['dataset'] + '/'
#     corpus = read_corpus(corpus_path)
#     n = len(corpus)
#     index_data = range(n)
#     n_set1 = math.ceil(n*frac)
#     set1_index = sample(index_data, n_set1, replace = False)
#     set2_index = complement(index_data, set1_index)
#     set1 = corpus[set1_index]
#     set2 = corpus[set2_index]


# Dirty tricks with cv fold to prepare data for testing

# First make just two folds, then make 10 on the training set
config_path = './apnews/apnews_config.toml'
inv_idx = metapy.index.make_inverted_index(config_path)
index_data = range(inv_idx.num_docs())
cv_folds = gen_cv_folds(index_data, 2)
gen_data_folds(config_path, cv_folds)
# TODO: delete redundant training fold (just keep the test fold)
# further split the test set in the first fold into ten test folds
config_path2 = './apnews/apnews/resampled/fold_1/test_fold.toml'
inv_idx2 = metapy.index.make_inverted_index(config_path2)
index_data2 = range(inv_idx2.num_docs())
cv_folds2 = gen_cv_folds(index_data2, 10)
gen_data_folds(config_path2, cv_folds2)


