**Base Directory (BD):**

Contains the config and stopwords.

1. config.toml:

Data configuration toml

stop-words = "stopwords_file.txt"

dataset = "dataset_name"

query-judgements = ""

corpus = "corpus_format.toml"  // Note: corpus_format.toml is assumed to be in the dataset subfolder

2. stopwords_file.txt:

File containing all the stopwords.

**BD/dataset_name:**

Contains the corpus.dat, corpus_format.toml, query file and query relevance file.

1. dataset-queries.txt:

Query file. Each line is a query.

2. dataset-qrels.txt:

Query relevance file with lines in the format "qID docID relevance"
