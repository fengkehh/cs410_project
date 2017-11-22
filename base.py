from base_helpers import *
from shutil import copy2


# Return a 2D array composed of randomly sampled lists of index.
#
# @param index_data: A full list of index of the ordered, iterable object.
#
# @param n: a positive integer indicating the number of documents to be sampled from the full list. If n <= 0 sample the
# full index
#
# @param k: the number of times sampling should be done. Each sampled list of index is stored on a separate ROW of
# the resulting 2D array.
#
# @param replace: boolean indicating whether sample uses replacement. if True, ignore n (use n = index_data or
# len(index_data) instead)
#
# @return: A k by n 2D array composed of k randomly sampled lists of index, each of size n.
def sample(index_data, n, k = 1, replace = False):
    if n <= 0:
        n = len(index_data)

    result = numpy.zeros((k, n))
    for i in range(k):
        result[i, 0:n] = numpy.sort(numpy.random.choice(index_data, n, replace))

    return result


# Generate resampled corpuses using a given fold indices and save them under the directory specified by user.
#
# @param config_path: a string pointing to the config file of the original corpus
#
# @param full_config: full configuration file in an OrderedDict
#
# @param folds: A k x n 2D numpy array containing indices of the documents inside each fold.
def gen_folds(config_path, folds):
    # Parse original config
    orig_config = parse_config(config_path)
    set_name = orig_config['dataset']
    orig_data_dir = orig_config['prefix'] + '/' + set_name + '/' # dir path to dataset.dat
    orig_corpus_path = orig_data_dir + set_name + '.dat'
    corpus = read_corpus(orig_corpus_path)

    full_index = range(len(corpus)) # list containing the full index of the corpus
    (k, n) = folds.shape
    for i in range(k):
        # Generate each inFold corpus and outFold corpus
        infold_index = folds[i,:]
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
        # TODO: config generation - config to be saved under orig_data_dir/fold_i/
        stopwords_path = os.path.abspath(orig_config['stop-words'])  # abs path for stopwords file
        # in-fold (training) config
        in_config = orig_config
        in_config['prefix'] = fold_dir # config to be saved under "orig_data_dir/fold_i/"
        in_config['stop-words'] = stopwords_path  # setting stopwords file path & name
        in_config['dataset'] = 'in'
        in_config['query-judgements'] = infold_dirpath + 'qrels-sampled.txt' # setting judgement file path & name
        write_config(in_config, fold_dir + 'in_fold.toml') # write configuration file
        copy2(orig_data_dir + orig_config['query-path'], infold_dirpath) # copy query file over
        # out-fold (testing)
        out_config = in_config
        out_config['dataset'] = 'out'
        out_config['query-judgements'] = outfold_dirpath + 'qrels-sampled.txt'
        write_config(in_config, fold_dir + 'out_fold.toml')
        copy2(orig_data_dir + orig_config['query-path'], outfold_dirpath)