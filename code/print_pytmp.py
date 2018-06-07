import os
import pickle
from sys import argv


filename = argv[1]
path = open("../data/feature/feat_{}.pytmp".format(filename), 'r')
tmp = pickle.load(path)
path.close()

print (tmp)
