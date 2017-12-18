from base import *
from glob import glob
from eval import evaluate
from operator import add
import metapy

# Number of desired training sets
k = 10

# Original set config
config_path = './apnews/apnews_config.toml'

# Tricks with cv fold to prepare data for testing
# First make just two folds, fold1 test fold is the training set, fold2 test fold is the test set.
# Then further split the training set into 10 disjoint sets for actual testing.
inv_idx = metapy.index.make_inverted_index(config_path)
index_data = range(inv_idx.num_docs())
cv_folds = gen_cv_folds(index_data, 2)
gen_data_folds(config_path, cv_folds)
# further split the test set in the first fold into ten test folds
config_path2 = './apnews/apnews/resampled/fold_1/test_fold.toml'
inv_idx2 = metapy.index.make_inverted_index(config_path2)
index_data2 = range(inv_idx2.num_docs())
cv_folds2 = gen_cv_folds(index_data2, k)
gen_data_folds(config_path2, cv_folds2)
# Delete redundant training folds (just keep the test folds)
resampled_dir = './apnews/apnews/resampled/'
for root, dirnames, filenames in os.walk(resampled_dir):
    curr_basedir = os.path.basename(root)
    if curr_basedir == 'train':
        rmtree(root)

    for file in filenames:
        if file == 'train_fold.toml':
            os.remove(root + '/' + file)