import metapy
from base import *
from math import floor

fidx = metapy.index.make_forward_index('./test_lv1/test_lv2/config.toml')
index = fidx.docs()

folds = gen_cv_folds(index, k = 10)

gen_data_folds('cranfield_config.toml', folds)