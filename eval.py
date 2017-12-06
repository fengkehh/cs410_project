from eval_metapy import *
from base_helpers import *

# Make predictions by generating a list of retrieved documents at cutoff from the test set, using model trained on the training set for each query
# @param test_config_path: string pointing to the test set configuration toml file
# @param train_config_path: string pointing to the train set configuration toml file
# @param queries_path: string pointing to the .dat file containing the queries
# @param model_params: dict() containing model parameters (only BM25 is implemented thus far)
# @param result_path: string pointing to the file where evaluation result is to be saved
# @param cutoff: n cutoff default to 10
# @return
def evaluate(test_config_path, train_config_path, queries_path, model_params, result_path, cutoff = 10):
    k = model_params['k']
    b = model_params['b']
    # setup analyzers and document container
    analyzer = metapy.analyzers.load(test_config_path)
    doc_container = metapy.index.Document()
    query_container = metapy.index.Document()

    # load config
    test_config = parse_config(test_config_path)
    train_config = parse_config(queries_path)
    # Generate inverted index of both test and training set
    idx_test = metapy.index.make_inverted_index(test_config_path)
    idx_train = metapy.index.make_inverted_index(train_config_path)
    # parse for documents in the test set
    test_corpus = read_corpus(test_config['prefix'] + '/' + test_config['dataset'] + '/' + test_config['dataset'] + '.dat')
    # Parse for queries
    queries = read_corpus(test_config['prefix'] + '/' + test_config['dataset'] + '/' + test_config['query-runner']['query-path'])
    qrel_path = test_config['query-judgments']
    # parse the query relevance data
    qrel_dict = qrel_parse(qrel_path)
    # open target file for writing result
    fid = file_open(result_path, 'w')
    # Do retrieval for each query
    for qID in range(len(queries)):
        # Generate analyzed query
        query_str = queries[qID]
        query_container.content(query_str)
        query_analyzed = analyzer.analyze(query_container)
        doc_scores = [] # going to be in [(doc_ID_test, score)]

        for docID_test in range(len(test_corpus)):
            # setting up document object and generate analyzed content
            doc_str = test_corpus[docID_test]
            doc_container.content(doc_str)
            doc_analyzed = analyzer.analyze(doc_container)
            document = (docID_test, doc_analyzed)
            # Compute BM25 score
            curr_score = BM25_score(document, query_analyzed, idx_test, idx_train, k, b)
            tuple = (docID_test,curr_score)
            doc_scores.append(tuple)

        doc_scores.sort(key = lambda tup: tup[1], reverse = True)
        # retrieved docs in the format [(doc_ID, ranker score),...]
        retrieved_docs = doc_scores[0:cutoff]
        # Determine the gain from query relevance file
        for ind in range(len(retrieved_docs)):
            # change ranker score to gain
            if retrieved_docs[ind][1] in qrel_dict[qID]:
                # qrel lookup table contains an entry for the current query for the current docID
                retrieved_docs[ind][1] = qrel_dict[qID][retrieved_docs[ind][0]]
            else:
                # document is completely irrelevant. set gain to 0
                retrieved_docs[ind][1] = 0

        # Compute nDCG for this qID and list of retrieved documents. Write to disk.
        qID_ndcg = ndcg(qID, retrieved_docs, qrel_dict)
        fid.write(str(qID_ndcg) + "\n")

    fid.close()