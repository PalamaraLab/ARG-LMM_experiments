"""Combine results from runs with different seeds."""
from functools import reduce
import glob
import operator
import os
import pickle
import sys

if len(sys.argv) <= 2:
    raise Exception("Missing arguments: combine_seeds.py base_path num_seeds [start_seed]")
# base_path is the path excluding "_s{seed}-1.pkl"
base_path = sys.argv[1]
num_seeds = int(sys.argv[2])
if len(sys.argv) > 3:
    start_seed = int(sys.argv[3])
else:
    start_seed = 1

seed_results = []
for i in range(start_seed, start_seed + num_seeds):
    with open(base_path + "_s" + str(i) + "-1.pkl", 'rb') as infile:
        seed_results.append(pickle.load(infile))

def recursive_combine(thing):
    assert isinstance(thing, list)
    if isinstance(thing[0], dict):
        combined = {}
        for key in thing[0]:
            if key in ["args", "bin_lower", "bin_upper"]:
                combined[key] = thing[0][key]
            else:
                combined[key] = recursive_combine([list_entry[key] for list_entry in thing])
        return combined
    else:
        assert isinstance(thing[0], list)
        return reduce(operator.concat, thing) # concatenates the lists for all seeds

results = recursive_combine(seed_results)

out_path = base_path + ".pkl"
if os.path.isfile(out_path):
    print("Error: could not write results")
    print("Please delete file {} before continuing".format(out_path))
else:
    print("Writing results to {}".format(out_path))
    with open(out_path, "wb") as outfile:
        pickle.dump(results, outfile)
