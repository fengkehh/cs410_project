### sample(index_data, n, replace = False)

Return a 2D array composed of randomly sampled lists of index.

@param index_data: A full list of index of the ordered, iterable object.

@param n: a positive integer indicating the number of documents to be sampled from the full list. If n <= 0 use the full index.

@param k: the number of times sampling should be done. Each sampled list of index is stored on a separate ROW of
the resulting 2D array.

@param replace: boolean indicating whether sample uses replacement. if True, ignore n (use n = index_data or
index_data.size() instead)

@return: A k by n 2D array composed of k randomly sampled lists of index, each of size n.

**Implementation**

Wrapper for [numpy.random.choice()](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.random.choice.html)

-----
### genSamples(corpus, folds, pathname)

Split a given corpus of text into folds (both infold and outfold) and save each fold into subfolders specified by pathname as a resampled corpus

@param corpus: a string representing the path to the file containing the full corpus to be resampled

@param folds: a list of list of docIDs, each sublist contains docIDs in that particular fold. If there are k resamples, then folds will be a list of k elements, each element being a list of the docIDs in the fold of that resample.

@param pathname: a string representing the path to the directory that will contain the resampled corpuses

-----
### eval(ranker, queries, corpus)

Return a list of evaluation scores of the retrieved documents from the given corpus using a given list of queries
@param ranker: a ranker with given trained hyper parameters

@param queries: a list of queries

@param corpus: a labeled corpus to compute evaluation scores for the retrievals
