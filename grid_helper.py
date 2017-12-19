import numpy
import itertools


# Helper function: generate a list of points for ONE parameter given the start, end and number of points.
def gen_grid_1d(start, end, num_pts, isInt = False):
    result = None
    if isInt:
        result = numpy.linspace(start, end, num_pts, dtype=int)
    else:
        result = numpy.linspace(start, end, num_pts)

    return result


def gen_grid_exp(startpow, endpow, num_pts):
    pow_grid = numpy.linspace(startpow, endpow, num_pts)
    result = numpy.power(2, pow_grid)
    return result

def gen_grid(param_spec):
    params_grid = dict()
    for key in param_spec:
        if param_spec[key][0] == 'lin':
            params_grid[key] = gen_grid_1d(*param_spec[key][1:4])
        else:
            params_grid[key] = gen_grid_exp(*param_spec[key][1:4])

    return params_grid