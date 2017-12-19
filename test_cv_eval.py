from base import *
from glob import glob
from eval import evaluate
from eval_metapy import *
import metapy

# flag to indicate whether to carry out initial data split and resampling
make_cache = True # flag to indicate if inverted index cache should be made
compute_resampled_evals = True # flag to indicate if resampled evaluations should be computed

# CV number of folds
k = 5

# Original set config
config_path = './apnews/apnews_config.toml'

# Walk through each set for training, generate cv sets and generate evaluations
# root folder containing all training sets folders.
root = './apnews/apnews/resampled/fold_1/test/resampled/'
if compute_resampled_evals:
    for dirname in next(os.walk(root))[1]:
        dirpath = root + dirname + '/'
        config_files = glob(dirpath + '*.toml')
        if len(config_files) != 1:
            print('Multiple config files detected in training set directory! Aborting...')
        else:
            if make_cache:
                config_file_path = config_files[0]
                # generate CV folds
                inv_idx = metapy.index.make_inverted_index(config_file_path)
                index_data = range(inv_idx.num_docs())
                folds = gen_cv_folds(index_data, k = k)
                gen_data_folds(config_file_path, folds)

            # Generate evaluations from the CV folds
            # path pointing to the resampled folders containing the folds
            folds_root_path = dirpath + '/test/resampled/'
            for fold in next(os.walk(folds_root_path))[1]:
                print('Processing ' + folds_root_path + fold)
                test_fold_config_file = folds_root_path + fold + '/test_fold.toml'
                train_fold_config_file = folds_root_path + fold + '/train_fold.toml'

                result_path = folds_root_path + fold + '/result.txt'
                evaluate(test_config_path = test_fold_config_file,
                         train_config_path = train_fold_config_file,
                         ranker = BM25_ranker,
                         model_params = {'k':1, 'b':0.5},
                         result_path = result_path, cutoff = 10)

# Walk through each training set and compute the arithmetic mean to get the final CV results.
for set in next(os.walk(root))[1]:
    result = numpy.array([])
    first_fold = True
    for fold in next(os.walk(root + set + '/test/resampled/'))[1]:
        fold_result_path = root + set + '/test/resampled/' + fold + '/result.txt'
        if first_fold:
            result = numpy.loadtxt(fold_result_path)
            first_fold = False
        else:
            result = result + numpy.loadtxt(fold_result_path)
    result = numpy.divide(result,k) # Compute average
    ind = 0
    fid = file_open(root + set + '/cv_result.txt', 'w')
    for ind in range(len(result)):
        fid.write(str(result[ind]) + '\n')
    fid.close()