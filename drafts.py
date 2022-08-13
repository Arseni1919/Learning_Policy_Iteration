# import numpy as np
# s = np.random.poisson(5, 1)[0]
# print(s)

# from scipy.stats import poisson, skellam
#
# #calculate probability
# # print(poisson.pmf(k=1, mu=0.001))
# print(skellam.pmf(k=-3, mu1=4, mu2=2))
# print(skellam.pmf(k=3, mu1=2, mu2=4))

# import itertools
# combs = itertools.product(range(20), repeat=2)
# print(list(combs))

d= {1: 10, 2: 4, 3: 1}

print(d[max(d, key=d.get)])