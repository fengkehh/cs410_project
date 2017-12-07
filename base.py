from base_helpers import *
from shutil import copy2, rmtree
from math import floor


# Return a numpy array composed of randomly sampled lists of index.
#
# @param index_data: A full list of index of the ordered, iterable object. Design decision: first index is 1!
#
# @param n: a positive integer indicating the number of documents to be sampled from the full list. If n <= 0
# sample the
# full index
#
# @param replace: boolean indicating whether sample uses replacement. if True, ignore n (use n = index_data or
# len(index_data) instead)
#
# @return: A 1 by n numpy array composed of n indices randomly sampled from index_data.
def sample(index_data, n, replace=False):
    if n <= 0:
        n = len(index_data)

    result = numpy.sort(numpy.random.choice(index_data, n, replace))

    return result


# Generate a set containing the training folds of a k cross-validation
#
# @param index_data: a list containing the initial full index of the data. Design decision: first fold index is 1!
#
# @param k: the number of folds for cross validation
#
# @return An dict with key = fold id (int 1 to k) and value = numpy array corresponding to the fold (test set).
def gen_cv_folds(index_data, k):
    folds = dict()
    n = floor(len(index_data)/k) # number of samples to draw from curr_index
    curr_index = index_data

    for i in range(k):
        if i != k - 1: # not generating the last fold
            test_fold = sample(curr_index, n, replace = False)
            curr_index = complement(curr_index, test_fold) # set curr_index to the rest of curr_index that hasn't
            # been chosen
        else: # generating the last fold
            test_fold = curr_index

        # training fold is elements not picked by sample().
        folds[i + 1] = complement(index_data, test_fold)

    return folds


# Generate a set containing the training folds of a bootstrap.
def gen_boot_folds(index_data, k):
    folds = dict()
    n = len(index_data)

    # pick bootstramp training fold directly using sample().
    for i in range(k):
        folds[i+1] = sample(index_data, n, replace = True)

    return folds


# Generate resampled corpuses using a given fold indices and save them under the directory specified by user.
#
# @param config_path: a string pointing to the config file of the original corpus
#
# @param full_config: full configuration file in an OrderedDict
#
# @param folds: An dict() containing numpy arrays of indices of the documents inside each fold.
def gen_data_folds(config_path, folds):
    # Parse original config
    orig_config = parse_config(config_path)
    set_name = orig_config['dataset']
    orig_data_dir = orig_config['prefix'] + '/' + set_name + '/' # dir path to dataset.dat
    orig_corpus_path = orig_data_dir + set_name + '.dat'
    corpus = read_corpus(orig_corpus_path)
    target_dir = orig_data_dir + 'resampled/'
    if (os.path.exists(target_dir)):
        # target directory is dirty. Delete!
        rmtree(target_dir)

    full_index = range(len(corpus)) # list containing the full index of the corpus

    for i in folds:
        # Generate each training set corpus and test set corpus
        trainfold_index = folds[i]
        testfold_index = complement(full_index, trainfold_index)
        trainfold_corpus = corpus[trainfold_index]
        testfold_corpus = corpus[testfold_index]
        # Caching corpuses on disk
        fold_dir = target_dir + 'fold_' + str(i) + '/' # dir path to each fold

        trainfold_dirpath = fold_dir + 'train/' # train fold dir path
        testfold_dirpath = fold_dir + 'test/'  # test  fold dir path
        write_corpus(trainfold_corpus, trainfold_dirpath + 'train.dat')
        write_corpus(testfold_corpus, testfold_dirpath + 'test.dat')
        # Judgement mapping
        full_qrel_path = orig_config['query-judgements'] # path to the original query judgment
        qrel_mapper(full_qrel_path, trainfold_index, trainfold_dirpath)
        qrel_mapper(full_qrel_path, testfold_index, testfold_dirpath)
        # Config generation - config to be saved under orig_data_dir/fold_i/
        stopwords_path = orig_config['stop-words']  # abs path for stopwords file
        # test set config
        test_config = orig_config.copy()
        test_config['prefix'] = fold_dir # config to be saved under "orig_data_dir/fold_i/"
        test_config['stop-words'] = stopwords_path # setting stopwords file path & name
        test_config['dataset'] = 'test'
        test_config['query-judgements'] = testfold_dirpath + 'qrels-sampled.txt' # setting judgement file path & name
        test_config['index'] = testfold_dirpath + 'idx'
        write_config(test_config, fold_dir + 'test_fold.toml') # write configuration file
        copy2(orig_data_dir + orig_config['query-runner']['query-path'], testfold_dirpath) # copy query file over
        copy2(orig_data_dir + 'line.toml', testfold_dirpath) # copy over corpus config file
        # training set config
        train_config = test_config
        train_config['dataset'] = 'train'
        train_config['query-judgements'] = trainfold_dirpath + 'qrels-sampled.txt'
        train_config['index'] = trainfold_dirpath + 'idx'
        write_config(train_config, fold_dir + 'train_fold.toml')
        copy2(orig_data_dir + orig_config['query-runner']['query-path'], trainfold_dirpath)
        copy2(orig_data_dir + 'line.toml', trainfold_dirpath)  # copy over corpus config file