### sample(index_data, n, k=1, replace = False)

Return a 2D array composed of randomly sampled lists of index.

@param index_data: A full list of index of the ordered, iterable object.

@param n: a positive integer indicating the number of documents to be sampled from the full list. If n <= 0 use the full index.

@param k: the number of times sampling should be done. Each sampled list of index is stored on a separate ROW of
the resulting 2D array.

@param replace: boolean indicating whether sample uses replacement. if True, ignore n (use n = index_data or
index_data.size() instead)

@return: A k by n 2D array composed of k randomly sampled lists of index, each of size n.

**Implementation**

Wrapper for [numpy.random.choice()](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.random.choice.html).

-----

### gen_folds(corpus, folds, dirpath)

Generate resampled corpuses using a given fold indices and save them under the directory specified by user.

@param corpus: A numpy array of strings containing all documents in the full corpus.

@param folds: A k x n 2D numpy array containing indices of the documents inside each fold.

@param dirpath: a string containing the path to the parent directory for the resampled corpuses to be saved.

**Implementation**

Helper functions:

1. complement(full, sub): Given a full 1D list or array and a subset of said list, return the complement.

2. write_corpus(strings, fullpath): Write a list (or numpy array) of strings into a corpus file from a user supplied filepath. 

Use complement() to generate out of fold index. Use numpy index array access and write_corpus() to write in fold and out of fold corpuses. 
 
-----

### eval(ranker, queries, corpus)

Return a list of evaluation scores of the retrieved documents from the given corpus using a given list of queries
@param ranker: a ranker with given trained hyper parameters

@param queries: a list of queries

@param corpus: a labeled corpus to compute evaluation scores for the retrievals
