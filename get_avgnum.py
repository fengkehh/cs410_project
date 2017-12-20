import os, numpy
from base_helpers import *
from operator import add

result_folder = './results'

result_subdirs = ['/set1', '/set2', '/set3', '/set4', '/set5']

def get_totnum(dmap, qrel_dict, n_queries):
    sum_rel_num = [0]*n_queries
    for qID in range(len(queries)):
        if qID not in qrel_dict:
            sum_rel_num[qID] = 0
        else:
            for new_docID in dmap:
                if dmap[new_docID] in qrel_dict[qID]:
                    sum_rel_num[qID] += 1
    return sum_rel_num

root = './apnews/apnews/resampled/fold_1/test/resampled/'
# Compute the average number of relevant files included in the CV folds of each set
qrel_dict = qrel_parse('./apnews/apnews/resampled/fold_1/test/orig_qrels.txt')
queries = read_corpus('./apnews/apnews/resampled/fold_1/test/queries.txt')
n_queries = len(queries)
for setname in next(os.walk(root))[1]:
    setpath = root + setname + '/'
    folds_root_path = setpath + '/test/resampled/'
    sum_rel_num = [0] * len(queries)
    for fold in next(os.walk(folds_root_path))[1]:
        # Find the number of relevant files included in the test fold
        dmap = parse_dmap(folds_root_path + fold + '/test/doc_map.txt')
        fold_result = get_totnum(dmap, qrel_dict, n_queries)
        sum_rel_num = map(add, sum_rel_num, fold_result)
    avg_rel_num = numpy.divide(list(sum_rel_num), 5)
    fid = file_open(setpath + 'fold_avgnum_rel_docs.txt', 'w')
    for avg in avg_rel_num:
        fid.write(str(avg) + '\n')
    fid.close()

# Compute the average number of relevant files included in each training set
for setname in next(os.walk(root))[1]:
    setpath = root + setname + '/'
    dmap = parse_dmap(setpath + 'test/doc_map.txt')
    avg_rel_num = get_totnum(dmap, qrel_dict, n_queries)
    fid = file_open(setpath + 'set_avgnum_rel_docs.txt', 'w')
    for avg in avg_rel_num:
        fid.write(str(avg) + '\n')
    fid.close()

# Compute the average number of relevant files included in the test set
testset_path = './apnews/apnews/resampled/fold_2/'
dmap = parse_dmap(testset_path + 'test/doc_map.txt')
avg_rel_num = get_totnum(dmap, qrel_dict, n_queries)
fid = file_open(testset_path + 'test_avgnum_rel_docs.txt', 'w')
for avg in avg_rel_num:
    fid.write(str(avg) + '\n')
fid.close()
