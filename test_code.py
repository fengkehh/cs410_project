import metapy
from base import *
from eval import *

fidx = metapy.index.make_forward_index('./cranfield_config.toml')
index = fidx.docs()

folds = gen_cv_folds(index, k = 10)

# Generate
gen_data_folds('./cranfield_config.toml', folds)

# Try out evaluate() on cranfield_set
evaluate('./cranfield_config.toml', './cranfield_config.toml','./cranfield/cranfield-queries.txt', {'k':1, 'b':0.5}, './cranfield/results.txt', cutoff = 10)
