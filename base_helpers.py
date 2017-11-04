import numpy
from collections import OrderedDict


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


# Helper function. Given a full 1D list or array and a subset of said list, return the complement.
def complement(full, sub):
    # Convert to sets first.
    fullset = set(full)
    subset = set(sub)

    return fullset - subset


# Helper function. Given a full path to a toml config file, parse settings and store them in an OrderedDict.
def parse_config(fullpath):
    result = OrderedDict()
    config = open(fullpath, 'r')
    line = config.readline()
    while line:
        tokens = line.split('=')

        if len(tokens) > 1:
            result[tokens[0].strip()] = tokens[1].strip()
        else: # this is a config header line (no value)
            result[tokens[0].strip()] = ''

        line = config.readline()

    return result


# Helper function. Given an OrderedDict with keys = config keys, values = config settings, write to a toml config file
# as specified by the user.
def write_config(configs, fullpath):
    file = open(fullpath, 'w')

    for key in configs.keys():
        setting = configs[key]
        line = key

        if setting: # empty setting means it's a config section header
            line = line + ' = ' + setting + '\n'
        else:
            line = line + '\n'

        file.write(line)