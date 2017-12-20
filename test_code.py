import metapy
from base import *
from eval import *

#config_path = './fake_data/fake_config.toml'
config_path = './cranfield/cranfield_config.toml'
# data_dir = './cranfield/'
# qrel_filename = 'cranfield-qrels.txt'

fidx = metapy.index.make_forward_index(config_path)
index = fidx.docs()

folds = gen_cv_folds(index, k = 2)

# Generate
gen_data_folds(config_path, folds)
fold_dir = './cranfield/cranfield/resampled/fold_1/'
test_config_path = fold_dir + 'test_fold.toml'
train_config_path = fold_dir + 'train_fold.toml'
# Try out evaluate() on cranfield_set
evaluate(test_config_path = test_config_path,
         train_config_path = train_config_path,
         ranker = BM25_ranker,
         model_params = {'k':1, 'b':0.5},
         result_path = fold_dir + 'results.txt',
         cutoff = 10)
