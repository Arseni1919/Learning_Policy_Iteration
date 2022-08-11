# import numpy as np
# s = np.random.poisson(5, 1)[0]
# print(s)

# from scipy.stats import poisson
#
# #calculate probability
# print(poisson.pmf(k=5, mu=3))

import itertools
combs = itertools.product(range(20), repeat=2)
print(list(combs))