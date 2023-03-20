"""Print pickle files matching a glob string."""
import glob
import pickle
import sys

if len(sys.argv) <= 1:
    raise Exception("Missing arguments: print_pickle.py file1 [file2 ...]")

for i in range(len(sys.argv) - 1):
    infile = sys.argv[i + 1]
    concise_string = infile.split("/")[-1].split(".")[0]
    with open(infile, 'rb') as pickle_file:
        x = pickle.load(pickle_file)
        print(concise_string, x)
        print()
