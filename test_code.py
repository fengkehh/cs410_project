import metapy
from base import *
from eval import *

#config_path = './fake_data/fake_config.toml'
config_path = './cranfield_config.toml'
data_dir = './cranfield/'
qrel_filename = 'cranfield-qrels.txt'

fidx = metapy.index.make_forward_index(config_path)
index = fidx.docs()

folds = gen_cv_folds(index, k = 2)

# Generate
gen_data_folds(config_path, folds)

# Try out evaluate() on cranfield_set
evaluate(test_config_path = data_dir + 'resampled/fold_1/test_fold.toml', train_config_path = data_dir + 'resampled/fold_2/train_fold.toml', model_params = {'k':1, 'b':0.5}, result_path = './cranfield/resampled/results.txt', mapped = True, qrel_path = data_dir + qrel_filename, cutoff = 10)
