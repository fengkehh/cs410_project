from eval import evaluate
from eval_metapy import *

# This script computes true model performance using test set evaluation
root = './apnews/apnews/resampled/fold_1/test/resampled/'
test_config_file = './apnews/apnews/resampled/fold_2/test_fold.toml'
for dirname in next(os.walk(root))[1]:
    dirpath = root + dirname + '/'
    train_config_file = dirpath + 'test_fold.toml'
    evaluate(test_config_path=test_config_file,
             train_config_path=train_config_file,
             ranker = BM25_ranker,
             model_params={'k': 1, 'b': 0.5},
             result_path= dirpath + 'test_eval_result.txt', cutoff=10)