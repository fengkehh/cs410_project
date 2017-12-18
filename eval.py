from eval_metapy import *
from base_helpers import *

# Make predictions by generating a list of retrieved documents at cutoff from the test set, using model trained on the training set for each query
# @param test_config_path: string pointing to the test set configuration toml file
# @param train_config_path: string pointing to the train set configuration toml file
# @param model_params: dict() containing model parameters (only BM25 is implemented thus far)
# @param result_path: string pointing to the file where evaluation result is to be saved
# @param cutoff: n cutoff default to 10
# @return
def evaluate(test_config_path, train_config_path, model_params, result_path, cutoff = 10):
    k = model_params['k']
    b = model_params['b']
    # setup analyzers and document container
    analyzer = metapy.analyzers.load(test_config_path)
    doc_container = metapy.index.Document()
    query_container = metapy.index.Document()

    # load config
    test_config = parse_config(test_config_path)
    # train_config = parse_config(queries_path)
    # Generate inverted index of both test and training set
    idx_test = metapy.index.make_inverted_index(test_config_path)
    idx_train = metapy.index.make_inverted_index(train_config_path)
    # parse for documents in the test set
    test_data_dir = test_config['prefix'] + '/' + test_config['dataset'] + '/'
    test_corpus = read_corpus(test_data_dir + test_config['dataset'] + '.dat')
    # Parse for queries
    queries = read_corpus(test_data_dir + test_config['query-runner']['query-path'])
    dmap = dict()
    if not os.path.exists(test_data_dir + 'doc_map.txt'): # no doc_map exists. No mapping required.
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
    # open target file for writing result
    fid = file_open(result_path, 'w')
    # Do retrieval for each query
    for qID in range(len(queries)):
        # Generate analyzed query
        query_str = queries[qID]
        query_container.content(query_str)
        query_analyzed = analyzer.analyze(query_container)
        doc_scores = [] # going to be in [(doc_ID_test, score)]

        # stored qID in the relevance file may not start at 0!
        qID = qID + test_config['query-runner']['query-id-start']

        for docID_new in range(len(test_corpus)):
            # setting up document object and generate analyzed content
            doc_str = test_corpus[docID_new]
            doc_container.content(doc_str)
            doc_analyzed = analyzer.analyze(doc_container)
            document = (docID_new, doc_analyzed)
            # Compute BM25 score
            curr_score = BM25_score(document, query_analyzed, idx_test, idx_train, k, b)
            tuple = (docID_new, curr_score)
            doc_scores.append(tuple)

        doc_scores.sort(key = lambda tup: tup[1], reverse = True)
        # retrieved docs in the format [(doc_ID, ranker score),...]
        retrieved_docs_scores = doc_scores[0:cutoff]
        retrieved_docs_gains = []
        # Determine the gain from query relevance file
        for ind in range(len(retrieved_docs_scores)):
            gain_val = 0
            # find doc gain from the relevance file
            docID = retrieved_docs_scores[ind][0]
            # relevance file is using old docID, use dmap to find old docID
            docID_old = dmap[docID]

            if (qID in qrel_dict) and (docID_old in qrel_dict[qID]):
                # qrel lookup table contains an entry for the current query for the current docID
                #retrieved_docs[ind][1] = qrel_dict[qID][retrieved_docs[ind][0]]
                gain_val = qrel_dict[qID][docID_old]
            retrieved_docs_gains.append((docID_old, gain_val))

        # Compute nDCG for this qID and list of retrieved documents. Write to disk.
        qID_ndcg = ndcg(qID, retrieved_docs_gains, qrel_dict, orig_docIDs)
        fid.write(str(qID_ndcg) + "\n")

    fid.close()
