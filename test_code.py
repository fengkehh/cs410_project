import metapy
from base import *

fidx = metapy.index.make_forward_index('./cranfield_config.toml')
index = fidx.docs()

folds = gen_cv_folds(index, k = 10)

gen_data_folds('cranfield_config.toml', folds)