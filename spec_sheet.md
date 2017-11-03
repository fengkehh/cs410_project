**sample(full_list, n, replace = False)**

Return a randomly sampled list of document IDs from a given list

@param full_list: a list containing all of the docIDs to choose from

@param n: an integer indicating the number of documents to be sampled from the full list

@param replace: boolean indicating whether sample uses replacement. if True, ignore n 
(use n = full_list.size() instead)

-----
**genSamples(corpus, folds, pathname)**

Split a given corpus of text into folds (both infold and outfold) and save each fold into subfolders specified by pathname as a resampled corpus

@param corpus: a string representing the path to the file containing the full corpus to be resampled

@param folds: a list of list of docIDs, each sublist contains docIDs in that particular fold. If there are k resamples, then folds will be a list of k elements, each element being a list of the docIDs in the fold of that resample.

@param pathname: a string representing the path to the directory that will contain the resampled corpuses

-----
**eval(ranker, queries, corpus)**

Return a list of evaluation scores of the retrieved documents from the given corpus using a given list of queries
@param ranker: a ranker with given trained hyper parameters

@param queries: a list of queries

@param corpus: a labeled corpus to compute evaluation scores for the retrievals
