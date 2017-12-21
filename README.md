# Document-Domain Resampling Framework ReadMe

This is a framework meant to support the development and deployment of document-domain resampling in the context of text retrieval and natural language processing.

A video presentation is also available [here](https://www.youtube.com/watch?v=MfIvKtaTc-k&feature=youtu.be) for those interested.

The framework is organized into three layers:

1. **Resampling:** tools for data partitioning and generation of support files (config, docID mapping, etc). Core files: `base.py`, `base_helpers.py`.

2. **Evaluation:** facilitate model evaluation using the train/test split methodology. Supports modular ranker. Currently a BM25 ranker and logarithmic NDCG are implemented by interfacing with the [MeTA toolkit](https://meta-toolkit.org/). Core files: `eval.py`,`eval_metapy.py`

3. **Testing:** template codes showing example use cases and to generate statistics to study method efficacy.

## Input Data Structure

Input data is assumed to be organized in the line corpus format (ie: single text files with each line representing a document). Both judgments and query are to be provided in separate text files. Judgments are stored as a mapping between query ID and document ID per line. 

In general, a data corpus is expected to have:
 
 1. A `root` folder that contains the configuration toml and other files such as the collection of stopwords.
 
 2. The configuration toml should have the basic entries pointing to the correct paths.
  
 3. A `root/dataset/` sub-folder that contains the actual corpus data files (corpus-format.toml, corpus.dat, query-relevance.txt, queries.txt...etc). 
 
An example dataset (`./cranfield/`) along with its configuration file is included in the repository. It is recommended to look over the corpus configuration files to see how it is setup.

## Installation

Required python libraries: pytoml, numpy, metapy, pymp. All libraries can be installed through pip. ie:

`pip install pymp`

I recommend installing [Anaconda](https://www.anaconda.com/download/) first and then pytoml and pymp. The framework itself can be used as long as you can import the corresponding source files prior to invoking the framework functions. Due to the use of pymp the framework does NOT work on Windows! Theoretically it should work on OSX but it has not been tested. It is only guaranteed to work on Linux.

## Resampling Layer

Core files: `base.py`, `base_helpers.py`

The resampling layer is meant to be used as building blocks to construct various resampling methods. The most important functions are:

1. `sample(index_data, n, replace = False)`: given an index_data array for a corpus with j documents (ie: an array containing sequential integers from 0 to j-1), generate a sorted array consisting of n randomly sampled indices from index_data.

2. `complement(full, sub)`: given `full`: a full index array and `sub`: a subset of the previous array, generate a sorted array that is the complement of sub.

2. `gen_data_folds(config_path, folds)`: given `config_path`: a path to a configuration file and `folds`: a dictionary with keys 1:k, values = sampled indices to be included in the training folds, carry out test and train splits specified by folds using the data specified by the config file, along with the corresponding data structures and all modified support files.

It is perhaps best to illustrate this using an example. Suppose one wishes to create 2-fold CV folds for the included cranfield set. First we will generate each of the test fold index and find the train fold index to pass along using the sample() and complement() functions. Once all the resampled indices are generated we can translate the indices to actual resampled data using gen_data_folds():
    
    from base import *
    
    folds = dict() // fold dictionary
    full_index = range(1400) //cranfield corpus contains 1400 documents
    test_fold = sample(full_index, 700, replace = False) // Generate first test fold
    folds[1] = complement(full_index, test_fold) // Put first train fold into fold dictionary
    rest = complement(full_index, test_fold) // All the indices that haven't been included in test folds.
    test_fold = sample(rest, 700, replace = False) // Generate second test fold
    folds[2] = complement(full_index, test_fold) // Put second train fold into fold dictionary
    
    gen_data_folds('./cranfield/cranfield_config.toml', folds) // Generate actual resampled data structures.

Two common resampling methods have been written using the process above (k-fold cross validation in `gen_cv_folds(index_data, k)` and bootstrap in `gen_boot_folds(index_data, k)`). More complex resampling methods such as nested cross-validation can be built using the exact same logic.

## Evaluation Layer

Core files: `eval.py`,`eval_metapy.py`

The evaluation layer provides facility to evaluate text retrieval ranking models using existing, judged files with support of the train/test split methodology. The most important function is self explanatory:

`evaluate(test_config_path, train_config_path, ranker, model_params, result_path = '', cutoff = 10)`: This allows user to specify the config file path to the test/train corpus, ranker function to be used and its model parameters, file to write the evaluation results to and the NDCG cutoff value for text retrieval. 

By systematically walking through each fold and carry out evaluations using documents from the test fold while computing all required variables in the ranker using characteristics from the train fold, the user can generate evaluation statistics on each resampled data for aggregation. For example, let's say we want to estimate the performance of a BM25 retrieval model using the 2-fold CV resample we have just generated on the cranfield set:

    fold_dir = './cranfield/cranfield/resampled/fold_1/'
    test_config_path = fold_dir + 'test_fold.toml'
    train_config_path = fold_dir + 'train_fold.toml' 
    evaluate(test_config_path = test_config_path,
         train_config_path = train_config_path,
         ranker = BM25_ranker,
         model_params = {'k':1, 'b':0.5},
         result_path = fold_dir + 'results.txt',
         cutoff = 10)

## Testing Layer:

Core files: all files starting with `test`

The test layer contains code for a study on the efficacy of 5-fold CV on a fairly large dataset. They are not fundamentally relevant to the functionality of the resampling framework but may serve as an example on using the framework to do more complicated things. The associated report can be found under `Report/FinalReport.pdf`.  