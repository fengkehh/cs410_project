from base_helpers import *
from shutil import copy2
from math import floor


# Return a numpy array composed of randomly sampled lists of index.
#
# @param index_data: A full list of index of the ordered, iterable object.
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


# Generate an ordered set containing the folds of a k cross-validation
#
# @param index_data: a list containing the initial full index of the data.
#
# @param k: the number of folds for cross validation
#
# @return An OrderedDict with key = fold id (int 1 to k) and value = numpy array corresponding to the fold (test set).
def gen_cv_folds(index_data, k):
    folds = OrderedDict()
    n = floor(len(index_data)/k) # number of samples to draw from curr_index
    curr_index = index_data

    for i in range(k):
        if i < k - 1: # not generating the last fold
            folds[i+1] = sample(curr_index, n, replace = False)
            curr_index = complement(curr_index,folds[i+1]) # set curr_index to the rest of curr_index that hasn't been
            # chosen
        else: # generating the last fold
            folds[i] = numpy.sort(curr_index)

    return curr_index


# Generate resampled corpuses using a given fold indices and save them under the directory specified by user.
#
# @param config_path: a string pointing to the config file of the original corpus
#
# @param full_config: full configuration file in an OrderedDict
#
# @param folds: An OrderedDict containing numpy arrays of indices of the documents inside each fold.
def gen_folds(config_path, folds):
    # Parse original config
    orig_config = parse_config(config_path)
    set_name = orig_config['dataset']
    orig_data_dir = orig_config['prefix'] + '/' + set_name + '/' # dir path to dataset.dat
    orig_corpus_path = orig_data_dir + set_name + '.dat'
    corpus = read_corpus(orig_corpus_path)

    full_index = range(len(corpus)) # list containing the full index of the corpus

    for i in folds.keys():
        # Generate each inFold corpus and outFold corpus
        # Definition: inFold is the test set, outFold is the training set.
        infold_index = folds[i]
        outfold_index = complement(full_index, infold_index)
        infold_corpus = corpus[infold_index]
        outfold_corpus = corpus[outfold_index]
        # Caching corpuses on disk
        fold_dir = orig_data_dir + 'fold_' + str(i) + '/' # dir path to each fold
        infold_dirpath = fold_dir + 'in/' # in fold dir path
        outfold_dirpath = fold_dir + 'out/' # out fold dir path
        write_corpus(infold_corpus, infold_dirpath + 'in.dat')
        write_corpus(outfold_corpus, outfold_dirpath + 'out.dat')
        # Judgement mapping
        full_qrel_path = orig_config['query-judgements'] # path to the original query judgment
        qrel_mapper(full_qrel_path, infold_index, infold_dirpath)
        qrel_mapper(full_qrel_path, outfold_index, outfold_dirpath)
        # Config generation - config to be saved under orig_data_dir/fold_i/
        stopwords_path = os.path.abspath(orig_config['stop-words'])  # abs path for stopwords file
        # in-fold (testing) config
        in_config = orig_config
        in_config['prefix'] = fold_dir # config to be saved under "orig_data_dir/fold_i/"
        in_config['stop-words'] = stopwords_path  # setting stopwords file path & name
        in_config['dataset'] = 'in'
        in_config['query-judgements'] = infold_dirpath + 'qrels-sampled.txt' # setting judgement file path & name
        write_config(in_config, fold_dir + 'in_fold.toml') # write configuration file
        copy2(orig_data_dir + orig_config['query-path'], infold_dirpath) # copy query file over
        # out-fold (training)
        out_config = in_config
        out_config['dataset'] = 'out'
        out_config['query-judgements'] = outfold_dirpath + 'qrels-sampled.txt'
        write_config(in_config, fold_dir + 'out_fold.toml')
        copy2(orig_data_dir + orig_config['query-path'], outfold_dirpath)