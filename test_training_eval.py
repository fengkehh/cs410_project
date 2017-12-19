from eval import evaluate
from eval_metapy import *

# This script computes overfitted model performance using training set evaluation
root = './apnews/apnews/resampled/fold_1/test/resampled/'
for dirname in next(os.walk(root))[1]:
    dirpath = root + dirname + '/'
    print('Processing ' + dirpath)
    config_file = dirpath + 'test_fold.toml'
    evaluate(test_config_path=config_file,
             train_config_path=config_file,
             ranker = BM25_ranker,
             model_params={'k': 1, 'b': 0.5},
             result_path=dirpath + '/train_eval_result.txt', cutoff=10)