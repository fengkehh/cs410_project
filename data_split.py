from base import *
from glob import glob
from eval import evaluate
import metapy

# flag to indicate whether to carry out initial data split and resampling
prepare_data = True
make_resamples = True
# Original set config
config_path = './apnews/apnews_config.toml'

if prepare_data:
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
    cv_folds2 = gen_cv_folds(index_data2, 10)
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

# Walk through each set for training, generate cv sets and generate evaluations
# root folder containing all training sets folders.
root = './apnews/apnews/resampled/fold_1/test/resampled/'

for dirname in next(os.walk(root))[1]:
    dirpath = root + dirname + '/'
    config_files = glob(dirpath + '*.toml')
    if len(config_files) != 1:
        print('Multiple config files detected in training set directory! Aborting...')
    else:
        if make_resamples:
            config_file_path = config_files[0]
            # generate CV folds
            inv_idx = metapy.index.make_inverted_index(config_file_path)
            index_data = range(inv_idx.num_docs())
            folds = gen_cv_folds(index_data, k = 10)
            gen_data_folds(config_file_path, folds)

        # Generate evaluations from the CV folds
        # path pointing to the resampled folders containing the folds
        folds_root_path = dirpath + '/test/resampled/'
        for fold in next(os.walk(folds_root_path))[1]:
            test_fold_config_file = folds_root_path + fold + '/test_fold.toml'
            train_fold_config_file = folds_root_path + fold + '/train_fold.toml'

            evaluate(test_config_path = test_fold_config_file,
                     train_config_path = train_fold_config_file,
                     model_params = {'k':1, 'b':0.5},
                     result_path = folds_root_path + fold + '/result.txt', cutoff = 10)