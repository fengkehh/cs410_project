from base_helpers import *


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
#
# @param dirpath: a string containing the path to the parent directory for the resampled corpuses to be saved.
def gen_folds(config_path, folds, dirpath):
    # Parse original config
    orig_config = parse_config(config_path)
    set_name = orig_config['dataset']
    orig_data_dir = './' + set_name + '/'
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
        infold_dirpath = dirpath + '/fold' + str(i) + '/in/'
        outfold_dirpath = dirpath + '/fold' + str(i) + '/out/'
        write_corpus(infold_corpus, infold_dirpath + 'in.dat')
        write_corpus(outfold_corpus, outfold_dirpath + 'out.dat')
        # Query mapping
        full_qrel_path = orig_data_dir + set_name + '-qrels.txt'
        qrel_mapper(full_qrel_path, infold_index, infold_dirpath)
        qrel_mapper(full_qrel_path, outfold_index, outfold_dirpath)
        # TODO: config generation