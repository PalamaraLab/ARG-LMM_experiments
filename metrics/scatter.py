"""Calculate data for TMRCA scatter plots
"""

import os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


# Python imports
import argparse
import glob
import logging
import numpy as np
import os
import pickle
import tskit

# Our packages
import arg_needle_lib
from arg_needle import normalize_arg


logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S")

parser = argparse.ArgumentParser(description="Create data for scatter plots.")
# Paths
parser.add_argument("--pickle_path", help="Path to pickle output", action="store", default="foo_scatter.pkl")
# Other arguments
parser.add_argument("--mode", help="Path to first tree", action="store", default="none")
parser.add_argument("--glob", help="Glob to get entries (checks for num_replicas)", action="store", default="none")
parser.add_argument("--num_replicas", help="Number of replicas", action="store", default=25, type=int)
parser.add_argument("--num_comparisons", help="Number of comparisons", action="store", default=20000, type=int)
parser.add_argument("--random_seed", help="Random seed", action="store", default=1, type=int)
parser.add_argument("--use_saved_norm", help="Use saved .norm.trees", action="store", default=1, type=int)

args = parser.parse_args()

args.demofile = os.path.join(CURRENT_DIR, "../common/CEU2.demo")
use_saved_norm = (args.use_saved_norm != 0)

logging.info("Command-line args:")
args_to_print = vars(args)
for k in sorted(args_to_print):
    logging.info(k + ": " + str(args_to_print[k]))

base_strings = []
for whole_path in sorted(glob.glob(args.glob)):
    concise_string = whole_path.split("/")[-1].split("_s")[0]
    base_strings.append("/".join(whole_path.split("/")[:-1] + [concise_string]))

print("Creating scatter plots over {} replicas for the following base paths:".format(args.num_replicas))
for base_string in base_strings:
    print("  {}".format(base_string))

trees_strings = []
for string in base_strings:
    for i in range(args.num_replicas):
        trees_strings.append("{}_s{}-1".format(string, i+1))

for trees_string in trees_strings:
    suffixes = [".trees", ".true.trees"]
    if use_saved_norm:
        suffixes.append(".norm.trees")
    for suffix in suffixes:
        if not os.path.exists(trees_string + suffix):
            raise ValueError("File {} does not exist".format(trees_string + suffix))

output_pkl = {}
for trees_string in trees_strings:
    logging.info("Processing trees for string {}".format(trees_string))
    ts1 = tskit.load(trees_string + ".true.trees")
    ts2 = tskit.load(trees_string + ".trees")
    assert ts1.sequence_length == ts2.sequence_length
    assert ts1.num_samples == ts2.num_samples
    sample_ids = np.arange(ts1.num_samples)
    logging.info("  Done loading")

    if use_saved_norm:
        ts3 = tskit.load(trees_string + ".norm.trees")
        assert ts1.sequence_length == ts3.sequence_length
        assert ts1.num_samples == ts3.num_samples
    else:
        ts3 = normalize_arg(ts2, args.demofile, verbose=False)
        logging.info("  Done normalizing")

    all_ts = [ts1, ts2, ts3]
    concise_trees_string = trees_string.split("/")[-1]
    output_pkl[concise_trees_string] = {}
    for j in range(len(all_ts)):
        arg = arg_needle_lib.tskit_to_arg(all_ts[j])

        logging.info("  Done converting tree {}".format(j+1))
        times = []
        np.random.seed(args.random_seed)
        for i in range(args.num_comparisons):
            position = np.random.uniform(0., all_ts[j].sequence_length)
            id1, id2 = np.random.choice(sample_ids, size=2, replace=False)
            times.append(arg.mrca(id1, id2, position).height)

        output_pkl[concise_trees_string]["times" + str(j+1)] = times

    with open(args.pickle_path, "wb") as pkl_file:
        pickle.dump(output_pkl, pkl_file)
