import numpy
from collections import OrderedDict
from os import *


# Helper function. Given a fullpath and mode, open this file for read or write. If it's write mode and directory doesn't
# exist, create it. Return the opened file.
def file_open(fullpath, mode):
    dirpath = path.dirname(fullpath)
    if mode.find('w') >= 0: # write mode
        if not path.exists(dirpath): # directory doesn't exist, create.
            makedirs(dirpath)

    return open(fullpath, mode)


# Helper function. Given a full 1D list or array and a subset of said list, return the complement in a list.
def complement(full, sub):
    # Convert to sets first.
    fullset = set(full)
    subset = set(sub)

    return list(fullset - subset)


# Helper function. Read corpus into a numpy array of strings from a user supplied filepath.
def read_corpus(fullpath):
    corpus = file_open(fullpath, 'r')
    line = corpus.readline()
    # Store into regular list first to take advantage of dynamic resizing.
    temp = []

    while line:
        temp.append(line)
        line = corpus.readline()

    # Convert to numpy array to allow list index access.
    result = numpy.array(temp)
    corpus.close()

    return result


# Helper function. Write a list (or numpy array) of strings into a corpus file from a user supplied filepath
def write_corpus(strings, fullpath):
    corpus = file_open(fullpath, 'w')
    for string in strings:
        corpus.write(string)

    corpus.close()


# Helper function. Given a full path to a toml config file, parse settings and store them in an OrderedDict.
def parse_config(fullpath):
    result = OrderedDict()
    config = file_open(fullpath, 'r')
    line = config.readline()
    while line:
        tokens = line.split('=')

        if len(tokens) > 1:
            result[tokens[0].strip()] = tokens[1].strip()
        else: # this is a config header line (no value)
            result[tokens[0].strip()] = ''

        line = config.readline()

    config.close()

    return result


# Helper function. Given an OrderedDict with keys = config keys, values = config settings, write to a toml config file
# as specified by the user.
def write_config(configs, fullpath):
    file = file_open(fullpath, 'w')

    for key in configs.keys():
        setting = configs[key]
        line = key

        if setting: # empty setting means it's a config section header
            line = line + ' = ' + setting + '\n'
        else:
            line = line + '\n'

        file.write(line)

    file.close()


# Helper function. Insert a key to a dict or OrderedDict and set value in a list. If key already exists, append value
# to the existing value.
def dict_insert(container, key, value):
    if key not in container:
        container[key] = [value]
    else:
        container[key] = container[key].append(value)


# Helper function. Given a full path to a query relevance file, a 1D array of sorted document indices and a
# path to a directory, save the resampled and reordered docIDs in a new query relevance file in the directory. Save the
# query docID mapping in the format of "original,new" (minus the quotes) in a mapping file in the directory.
def qrel_mapper(qrel_path, fold, targetdir):
    # parse for the original relevance labels and store them in temp storage.
    # temp storage to store original relevance labels.
    qrel = file_open(qrel_path, 'r')
    orig_rel = dict()
    qrel_line = qrel.readline()

    while not qrel_line:
        tokens = qrel_line.split()
        # Format: key = original docID, value = list [tuple (qID, relevance)]
        dict_insert(orig_rel,key=tokens[1], value=(tokens[0],tokens[2]))
        qrel_line = qrel.readline()

    qrel.close()

    # Write out docID mapping for the query relevance file
    # Format: original,new on each line for docIDs in the sampled fold
    # temp storage for resampled relevance
    qmap = file_open(targetdir + 'qmap.txt', 'w')
    temp = dict()

    for i in range(len(fold)):
        docID_orig = fold[i]
        qmap_line = str(docID_orig) + ',' + str(i) + '\n'
        qmap.write(qmap_line)
        # Also put relevance into temp storage for resampled docs. Format: key = qID,
        # value = list of [tuple (new docID, rel)]
        doc_rel = orig_rel[docID_orig]

        for tuple in doc_rel:
            dict_insert(temp,key=tuple[0],value=(str(i),tuple[1]))

    qmap.close()

    # Write the sampled qrel file with new docIDs.
    qrels_samp = file_open(targetdir + 'qrels-sampled.txt', 'w')
    sorted_qids = sorted(temp.keys())

    for qID in sorted_qids:
        rsamp_line = qID

        for tuple in temp[qID]:
            rsamp_line = rsamp_line + ' ' + tuple[0] + ' ' + tuple[1] + '\n'
        qrels_samp.write(rsamp_line)

    qrels_samp.close()