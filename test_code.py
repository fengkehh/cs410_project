import metapy
from base import *

fidx = metapy.index.make_forward_index('./fake_data/fake_config.toml')
index = fidx.docs()

folds = gen_boot_folds(index, k = 2)

gen_data_folds('./fake_data/fake_config.toml', folds)