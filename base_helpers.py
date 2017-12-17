import pytoml, os.path


# Helper function. Given a data array and an array of index, return an array pointed to by the index
def get_by_index(data_array, ind_array):
    result = [data_array[i] for i in ind_array]
    return result


# Helper function. Given a fullpath and mode, open this file for read or write. If it's write mode and directory doesn't
# exist, create it. Return the opened file.
def file_open(fullpath, mode):
    dirpath = os.path.dirname(fullpath)
    if mode.find('w') >= 0: # write mode
        if not os.path.exists(dirpath): # directory doesn't exist, create.
            os.makedirs(dirpath)

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
    result = []

    while line:
        line = line.strip('\n')
        #if line: # skip any lines that are just the new line
        result.append(line)
        line = corpus.readline()

    # Convert to numpy array to allow list index access.
    # result = numpy.array(temp)
    corpus.close()
    return result

# Helper function. Write a list (or numpy array) of strings into a corpus file from a user supplied filepath
def write_corpus(strings, fullpath):
    corpus = file_open(fullpath, 'w')
    if (len(strings) > 0):
        for string in strings[:-1]:
            corpus.write(string + "\n")
        else:
            corpus.write(strings[-1])

    corpus.close()


# Helper function. Given a full path to a toml config file, parse settings and store them in a dict. Now using
# pytoml library.
def parse_config(fullpath):
    fid = file_open(fullpath, 'r')
    result = pytoml.load(fid)
    fid.close()

    return result


# Helper function.  Given a dict configs with keys = config keys, values = config settings, write to a toml config file
# as specified by the user. Now using pytoml library.
def write_config(configs, fullpath):
    fid = file_open(fullpath, 'w')
    string = pytoml.dump(configs, fid)
    fid.close()


# Helper function. Insert a key to a dict or OrderedDict and set value in a list. If key already exists, append value
# to the existing value.
def dict_insert(container, key, value):
    if key not in container:
        container[key] = [value]
    else:
        container[key].append(value)


# Helper function. Parse query relevance. Return a dictionary of the form {qID: {docID: gain}}
def qrel_parse(qrel_path):
    qrel_dict = dict()
    fid = file_open(qrel_path, 'r')
    line = fid.readline()
    while line:
        tokens = line.split()
        qID = int(tokens[0])
        docID = int(tokens[1])
        gain = int(tokens[2])
        if qID not in qrel_dict:
            qrel_dict[qID] = dict()
        qrel_dict[qID][docID] = gain
        line = fid.readline()

    fid.close()
    return qrel_dict


# Helper function: return value associated with given key from the OrderedDict config.
# Normally returns a string (with " stripped from front and end). If key is not found in config, return -1.
# def config_getval(config, key):
#     if key in config.keys():
#         return config[key].strip('"')
#     else:
#         return -1


# Helper function: set value associated with given key to the OrderedDict config.
# def config_setval(config, key, val):
#     value = '"' + val + '"'
#     config[key] = value


# Generate doc_ID map in the format "new_id old_id". old_index are the old doc IDs, new_index are the new ones. Both lists have to be the same size.
def doc_mapper(target_path, old_index, new_index):
    fid = file_open(target_path, 'w')
    ind = 0
    n = len(old_index)
    while ind != n:
        line = str(new_index[ind]) + ' ' + str(old_index[ind])
        if ind < n-1:
            fid.write(line + '\n')
        else:
            fid.write(line)
        ind += 1
    fid.close()


# Helper function. Given a full path to a query relevance file, a 1D array of sorted document indices and a
# path to a directory, save the resampled and reordered docIDs in a new query relevance file in the directory. Save the
# query docID mapping in the format of "original,new" (minus the quotes) in a mapping file in the directory.
def qrel_mapper(qrel_path, fold, targetdir):
    # parse for the original relevance labels and store them in temp storage.
    # temp storage to store original relevance labels.
    qrel = file_open(qrel_path, 'r')
    orig_rel = dict()
    qrel_line = qrel.readline()

    while qrel_line:
        tokens = qrel_line.split()
        # Format: key = original docID, value = list [tuple (qID, relevance)]
        dict_insert(orig_rel,key=tokens[1], value=(tokens[0],tokens[2]))
        qrel_line = qrel.readline()

    qrel.close()

    # Write out docID mapping for the query relevance file
    # Format:
    # new_docID old_docID
    # on each line for docIDs in the sampled fold
    # temp storage for resampled relevance
    dmap = file_open(targetdir + 'doc_map.txt', 'w')
    temp = dict()

    for i in range(len(fold)):
        docID_orig = str(fold[i])
        dmap_line = str(i) + ' ' + docID_orig + '\n'
        dmap.write(dmap_line)
        # Also put relevance into temp storage for resampled docs. Format: key = qID,
        # value = list of [tuple (new docID, rel)]

        if docID_orig in orig_rel.keys(): # current doc in the fold has associated relevance judgment(s)
            doc_rel = orig_rel[docID_orig]
            for tuple in doc_rel: # process all relevance judgments for it
                dict_insert(temp, key=tuple[0], value=(str(i), tuple[1]))
        else: # current doc in the fold doesn't have relevance judgment
            continue

    dmap.close()

    # Write the sampled qrel file with new docIDs.
    qrels_samp = file_open(targetdir + 'qrels-sampled.txt', 'w')
    sorted_qids = sorted(temp.keys())

    for qID in sorted_qids:
        for tuple in temp[qID]:
            rsamp_line = qID + ' ' + tuple[0] + ' ' + tuple[1] + '\n'
            qrels_samp.write(rsamp_line)

    qrels_samp.close()


# Parse a dmap file to generate a dmap dictionary in the form {new_docID: old_docID}.
def parse_dmap(dmap_path):
    dmap = dict()
    fid = file_open(dmap_path, 'r')
    line = fid.readline()
    while line:
        tokens = line.split()
        dmap[int(tokens[0])] = int(tokens[1])
        line = fid.readline()

    fid.close()

    return dmap