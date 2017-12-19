from eval import *
from base import *
from ast import literal_eval
from eval_metapy import *


# BM25 grid search (parameter optimizer)
# This function conducts a grid search in the range defined by model_param_range by
# minimizing training set error.
# param_spec: {'k': (type, start, end, num), 'b' : (type, start, end, num)}
# type = string indicating linear or exponential grid ('lin' or 'exp')
def BM25_grid_search(train_config_path, param_spec, cutoff = 10):
    # Setting up all the relevant arguments before evaluate_cached
    # setup analyzers and document container
    analyzer = metapy.analyzers.load(train_config_path)
    doc_container = metapy.index.Document()
    query_container = metapy.index.Document()

    # load config
    test_config = parse_config(train_config_path)
    # Generate inverted index of both test and training set
    idx_test = metapy.index.make_inverted_index(train_config_path)
    # idx_train = metapy.index.make_inverted_index(train_config_path)
    # parse for documents in the test set
    test_data_dir = test_config['prefix'] + '/' + test_config['dataset'] + '/'
    test_corpus = read_corpus(test_data_dir + test_config['dataset'] + '.dat')
    # setting up document object and generate analyzed content
    documents = []
    for docID_new in range(len(test_corpus)):
        doc_str = test_corpus[docID_new]
        doc_container.content(doc_str)
        doc_analyzed = analyzer.analyze(doc_container)
        documents.append((docID_new, doc_analyzed))
    # Parse for queries
    queries = read_corpus(test_data_dir + test_config['query-runner']['query-path'])
    analyzed_queries = []
    for qID in range(len(queries)):
        # Generate analyzed query
        query_str = queries[qID]
        query_container.content(query_str)
        analyzed_queries.append(analyzer.analyze(query_container))

    dmap = dict()
    if not os.path.exists(test_data_dir + 'doc_map.txt'):  # no doc_map exists. No mapping required.
        # Doc map is identity.
        for i in range(len(test_corpus)):
            dmap[i] = i
    else:
        # test set is splitted from a full set. Full relevance is provided by user through qrel_path
        # must read doc_map.txt and generate docID mapper
        dmap = parse_dmap(test_data_dir + 'doc_map.txt')

    # parse the query relevance data and generate a list of all original doc_IDs included in this test set.
    qrel_path = test_config['query-judgments']
    qrel_dict = qrel_parse(qrel_path)
    orig_docIDs = [dmap[doc_ID] for doc_ID in range(idx_test.num_docs())]

    # determine optimized parameters
    param_grid = gen_grid(param_spec)
    allKeys = sorted(param_grid)
    param_combs = itertools.product(*(param_grid[Key] for Key in allKeys))
    n_combs = len(list(param_combs)) # inefficient hack to get the # of total combos
    param_combs = itertools.product(*(param_grid[Key] for Key in allKeys))
    param_dict = dict()
    curr_eval_max = 0
    max_k = 1
    max_b = 0.5
    count = 1
    last_report = 0
    for param in param_combs:
        for i in range(len(allKeys)):
            param_dict[allKeys[i]] = param[i]

        evaluations = evaluate_cached(test_config, documents, analyzed_queries, qrel_dict, dmap, orig_docIDs, idx_test,
                                      idx_test, BM25_ranker, param_dict, result_path = '', cutoff = 10)
        eval_score = numpy.mean(evaluations)
        if eval_score > curr_eval_max:
            curr_eval_max = eval_score
            max_k = param_dict['k']
            max_b = param_dict['b']
        if (count - last_report)/ n_combs >= 0.05:
            print(str(count/n_combs*100) + '%')
            last_report = count
        count += 1

    end_result = (max_k, max_b, curr_eval_max)
    base_path = os.path.dirname(train_config_path)
    fid = file_open(base_path + '/gs_result.txt', 'w')
    fid.write(str(end_result))
    fid.close()
    return end_result


# This section uses grid search on training set to find model parameters for each set
param_spec = {'k':('lin', 1, 2.5, 3), 'b':('lin', 0, 0.5, 3)}
root = './apnews/apnews/resampled/fold_1/test/resampled/'
for dirname in next(os.walk(root))[1]:
    dirpath = root + dirname + '/'
    print('Processing ' + dirpath)
    if not os.path.exists(dirpath + '/gs_result.txt'):
        config_file = dirpath + '/test_fold.toml'
        BM25_grid_search(config_file, param_spec, 10)
    else:
        print('GS result found. Skipping...')


# This section uses the GS result to generate training, cv and test evaluations from optimized BM25.
test_config_file = './apnews/apnews/resampled/fold_2/test_fold.toml'
for dirname in next(os.walk(root))[1]:
    dirpath = root + dirname + '/'
    print('Evaluating ' + dirpath)
    config_file = dirpath + '/test_fold.toml'
    fid = file_open(dirpath + '/gs_result.txt', 'r')
    gs_result = literal_eval(fid.readline())
    fid.close()
    model_params = {'k':gs_result[0], 'b':gs_result[1]}
    # train evaluation
    if not os.path.exists(dirpath + '/optimized_train_result.txt'):
        print('Training evaluation on ' + dirpath)
        evaluate(test_config_path=config_file,
                 train_config_path=config_file,
                 ranker=BM25_ranker,
                 model_params=model_params,
                 result_path=dirpath + '/optimized_train_result.txt', cutoff=10)
    else:
        print('Result found. Skipping.')
    # test evaluation
    print('Test evaluation on ' + dirpath)
    if not os.path.exists(dirpath + '/optimized_test_result.txt'):
        evaluate(test_config_path=test_config_file,
                 train_config_path=config_file,
                 ranker=BM25_ranker,
                 model_params=model_params,
                 result_path=dirpath + '/optimized_test_result.txt', cutoff=10)
    else:
        print('Result found. Skipping.')
    # CV evaluation
    # Generate evaluations from the CV folds
    # path pointing to the resampled folders containing the folds
    folds_root_path = dirpath + '/test/resampled/'
    for fold in next(os.walk(folds_root_path))[1]:
        print('CV evaluation on ' + folds_root_path + fold)
        test_fold_config_file = folds_root_path + fold + '/test_fold.toml'
        train_fold_config_file = folds_root_path + fold + '/train_fold.toml'

        result_path = folds_root_path + fold + '/opt_CV_result.txt'
        if not os.path.exists(result_path):
            evaluate(test_config_path=test_fold_config_file,
                     train_config_path=train_fold_config_file,
                     ranker=BM25_ranker,
                     model_params=model_params,
                     result_path=result_path, cutoff=10)
        else:
            print('Result found. Skipping.')

# Walk through each training set and compute the arithmetic mean to get the final CV results.
for set in next(os.walk(root))[1]:
    result = numpy.array([])
    first_fold = True
    for fold in next(os.walk(root + set + '/test/resampled/'))[1]:
        fold_result_path = root + set + '/test/resampled/' + fold + '/opt_CV_result.txt'
        if first_fold:
            result = numpy.loadtxt(fold_result_path)
            first_fold = False
        else:
            result = result + numpy.loadtxt(fold_result_path)
    result = numpy.divide(result, 5)  # Compute average
    ind = 0
    fid = file_open(root + set + '/optimized_CV_result.txt', 'w')
    for ind in range(len(result)):
        fid.write(str(result[ind]) + '\n')
    fid.close()