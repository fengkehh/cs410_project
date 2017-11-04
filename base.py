import numpy


# Return a 2D array composed of randomly sampled lists of index.
#
# @param index_data: A full list of index of the ordered, iterable object.
#
# @param n: a positive integer indicating the number of documents to be sampled from the full list. If n <= 0 sample the
# full index
#
# @param k: the number of times sampling should be done. Each sampled list of index is stored on a separate ROW of
# the resulting 2D array.
#
# @param replace: boolean indicating whether sample uses replacement. if True, ignore n (use n = index_data or
# len(index_data) instead)
#
# @return: A k by n 2D array composed of k randomly sampled lists of index, each of size n.
def sample(index_data, n, k = 1, replace = False):
    if n <= 0:
        n = len(index_data)

    result = numpy.zeros((k, n))
    for i in range(k):
        result[i, 0:n] = numpy.random.choice(index_data, n, replace)

    return result


# Helper function. Read corpus into a numpy array of strings from a user supplied filepath.
def read_corpus(fullpath):
    corpus = open(fullpath, 'r')
    line = corpus.readline()
    # Store into regular list first to take advantage of dynamic resizing.
    temp = []

    while line:
        temp.append(line)
        line = corpus.readline()

    # Convert to numpy array to allow list index access.
    result = numpy.array(temp)

    return result


# Helper function. Write a list (or numpy array) of strings into a corpus file from a user supplied filepath
def write_corpus(strings, fullpath):
    corpus = open(fullpath, 'w')
    for string in strings:
        corpus.write(string)


test = read_corpus('./cranfield/cranfield.dat')

print(test[1000])