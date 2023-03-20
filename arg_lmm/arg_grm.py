"""Heritability and polygenic prediction using ground-truth ARG-GRMs.
"""

import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CURRENT_DIR, '../')) # for utils, simulator


# Python imports
import argparse
import logging
import numpy as np
import os
import pickle
import psutil; process = psutil.Process(os.getpid())

# Our packages
import arg_needle_lib
from arg_needle_lib import gower_center

# Files from this repository
from common.simulator import Simulator
from common.utils import ukb_sample
from lmm_utils import sim_pheno_new, get_allele_frequencies
from lmm_utils import make_grm, gcta

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

parser = argparse.ArgumentParser(description='Simulate ARGs and GRM.')
# Paths
parser.add_argument("--base_tmp_dir", help="Base temporary directory for intermediate output", action="store",
    default=os.path.join(CURRENT_DIR, "../data/temp/"))
parser.add_argument("--gcta_path", help="Path to GCTA binary", action="store",
    default=os.path.join(CURRENT_DIR, "../gcta"))
parser.add_argument("--pkl_path", help="Pickle file path", action="store", default=None)
# Other arguments
parser.add_argument("-a", "--alpha", help="Value of Speed et al. alpha for frequency weight", action="store", default=0., type=float)
parser.add_argument("--arg_grm", help="Direct ARG GRMs", action="store", default=0, type=int)
parser.add_argument("--diploid", help="Whether to do diploid", action="store", default=0, type=int)
parser.add_argument("--h2", help="Value of h2 to be simulated using infinitesimal model", action="store", default=0.8, type=float)
parser.add_argument("--mkl_num_threads", help="Number of MKL threads (default of 0 means not set)", action="store", default=0, type=int)
parser.add_argument("--mut_factors", help="log10(factor)*2 for mutation as a list", action="store", default="2", type=str)
parser.add_argument("--num_samples", help="Number of samples", action="store", default=1000, type=int)
parser.add_argument("--num_seeds", help="Number of seeds", action="store", default=1, type=int)
parser.add_argument("--sim_length", help="Simulation length", action="store", default=3e5, type=float)
parser.add_argument("--start_seed", help="Start seed", action="store", default=1, type=int)
parser.add_argument("--test_alphas", help="Value of Speed et al. alpha for frequency weight", action="store", default="", type=str)
parser.add_argument("--verbose", help="Verbose printing", action="store", default=1, type=int)
args = parser.parse_args()

if args.mkl_num_threads > 0:
    import mkl
    mkl.set_num_threads(args.mkl_num_threads)

args.mapfile = None
args.demofile = os.path.join(CURRENT_DIR, "../common/CEU2.demo")
args.mu = 1.65e-8
args.rho = 1.2e-8

logging.info("Command-line args:")
args_to_print = vars(args)
for k in sorted(args_to_print):
    logging.info(k + ": " + str(args_to_print[k]))
verbose = (args.verbose != 0)
use_arg_grm = (args.arg_grm != 0)
diploid = (args.diploid != 0)
if args.test_alphas != "":
    test_alphas = [float(alpha_string) for alpha_string in args.test_alphas.split(',')]
else:
    test_alphas = [args.alpha]
if args.mut_factors != "":
    mut_factors = [int(mut_factor) for mut_factor in args.mut_factors.split(',')]
else:
    mut_factors = []

if diploid:
    simulator_num_samples = args.num_samples*2
else:
    simulator_num_samples = args.num_samples

simulator = Simulator(args.mapfile, args.demofile,
    sample_size=simulator_num_samples,
    mu=args.mu, rho=args.rho)

if args.pkl_path is not None and os.path.exists(args.pkl_path):
    with open(args.pkl_path, 'rb') as infile:
        results = pickle.load(infile)
else:
    results = {}


for key in ["sites", "trees", "arg_ts"]:
    if key not in results:
        results[key] = []
for key in ["snp_h2", "seq_h2", "mut_h2", "arg_h2"]:
    if key not in results:
        results[key] = {}
    for alpha in test_alphas:
        if alpha not in results[key]:
            results[key][alpha] = []
if "mut_h2_vary" not in results:
    results["mut_h2_vary"] = {}
for factor in mut_factors:
    if factor not in results["mut_h2_vary"]:
        results["mut_h2_vary"][factor] = {}
    for alpha in test_alphas:
        if alpha not in results["mut_h2_vary"][factor]:
            results["mut_h2_vary"][factor][alpha] = []

if "gcta_ms" not in results:
    results["gcta_ms"] = {}
    results["gcta_ms"][0] = []
    results["gcta_ms"][-1] = []

if "mut_ms" not in results:
    results["mut_ms"] = {}
    results["mut_ms"][0] = []
    results["mut_ms"][-1] = []

if "mut_ms_vary" not in results:
    results["mut_ms_vary"] = {}
for factor in mut_factors:
    if factor not in results["mut_ms_vary"]:
        results["mut_ms_vary"][factor] = {}
        results["mut_ms_vary"][factor][0] = []
        results["mut_ms_vary"][factor][-1] = []


logging.info("Starting data (possibly read from pickle file)")
logging.info(results)


for seed_offset in range(args.num_seeds):
    seed = seed_offset + args.start_seed
    logging.info("Starting simulation " + str(seed))
    simulation = simulator.simulation(args.sim_length, random_seed=seed)
    logging.info("{} nodes, {} edges, {} mutations".format(simulation.num_nodes, simulation.num_edges, simulation.num_mutations))
    logging.info("Memory: {}".format(process.memory_info().rss))

    pheno, num_sites = sim_pheno_new([simulation], args.h2, args.alpha, seed, diploid)

    if verbose:
        logging.info((simulation.num_mutations, simulation.num_samples))
        logging.info("Number of trees: " + str(simulation.num_trees))
    if len(results["sites"]) <= seed_offset:
        results["sites"].append(simulation.num_mutations)
    if len(results["trees"]) <= seed_offset:
        results["trees"].append(simulation.num_trees)
    snp_indices = ukb_sample(simulation, verbose)
    arg = arg_needle_lib.tskit_to_arg(simulation)
    arg.populate_children_and_roots()
    logging.info("Memory: {}".format(process.memory_info().rss))

    for alpha in test_alphas:
        if len(results["snp_h2"][alpha]) <= seed_offset:
            grm = make_grm(simulation, snp_indices, alpha, diploid)
            gower_center(grm)
            if verbose:
                logging.info("Gower-centered SNP-estimated kinship with alpha = " + str(alpha) + ":")
                logging.info(grm[:5, :5])
            logging.info("Memory: {}".format(process.memory_info().rss))
            results["snp_h2"][alpha].append(gcta(args.gcta_path, args.base_tmp_dir, [grm], pheno, [len(snp_indices)], args.pkl_path))
            if args.pkl_path is not None:
                with open(args.pkl_path, 'wb') as outfile:
                    pickle.dump(results, outfile)

    for alpha in test_alphas:
        if len(results["seq_h2"][alpha]) <= seed_offset:
            grm = make_grm(simulation, None, alpha, diploid)
            gower_center(grm)
            if verbose:
                logging.info("Gower-centered sequence-estimated kinship with alpha = " + str(alpha) + ":")
                logging.info(grm[:5, :5])
            logging.info("Memory: {}".format(process.memory_info().rss))
            results["seq_h2"][alpha].append(gcta(args.gcta_path, args.base_tmp_dir, [grm], pheno, [simulation.num_mutations], args.pkl_path))
            if args.pkl_path is not None:
                with open(args.pkl_path, 'wb') as outfile:
                    pickle.dump(results, outfile)

    # Partitioned heritability with bins. We use alpha = -1 and alpha = 0
    af = get_allele_frequencies(simulation, None)
    maf = np.minimum(af, 1 - af)
    bin_boundaries = [0, 0.0025, 0.01, 0.05, 0.5]
    for alpha in [-1, 0]:
        if len(results["gcta_ms"][alpha]) <= seed_offset:
            grms = []
            snp_counts = []
            for k in range(len(bin_boundaries) - 1):
                low = bin_boundaries[k]
                high = bin_boundaries[k+1]
                select = np.where((low < maf) & (maf <= high))[0]
                logging.info(str(select.shape[0]) + " SNPs used in bin " + str(k))
                if select.shape[0] == 0:
                    logging.info("Skipping GRM with range (" + str(low) + ", " + str(high) + "]")
                else:
                    grm = make_grm(simulation, select, alpha, diploid)
                    gower_center(grm)
                    grms.append(grm)
                    snp_counts.append(select.shape[0])
            logging.info("Memory: {}".format(process.memory_info().rss))
            results["gcta_ms"][alpha].append(gcta(args.gcta_path, args.base_tmp_dir, grms, pheno, snp_counts, args.pkl_path))
            if args.pkl_path is not None:
                with open(args.pkl_path, 'wb') as outfile:
                    pickle.dump(results, outfile)

    del simulation

    # Do everything again with mutated data
    if True:
        COUNT_PLACEHOLDER = 500 # needed as input for GCTA but unused
        for mut_factor in mut_factors:
            new_mu = args.mu * 10**(mut_factor / 2)
            logging.info("new_mu: " + str(new_mu))

            for alpha in test_alphas:
                if len(results["mut_h2_vary"][mut_factor][alpha]) <= seed_offset:
                    grm = arg_needle_lib.monte_carlo_arg_grm(arg, new_mu, seed, alpha, diploid, centering=False)
                    gower_center(grm)
                    if verbose:
                        logging.info("Gower-centered mutation-estimated kinship with alpha = " + str(alpha) + ":")
                        logging.info(grm[:5, :5])
                    logging.info("Memory: {}".format(process.memory_info().rss))
                    current_result = gcta(args.gcta_path, args.base_tmp_dir, [grm], pheno, [COUNT_PLACEHOLDER], args.pkl_path)
                    results["mut_h2_vary"][mut_factor][alpha].append(current_result)
                    if mut_factor == 0:
                        results["mut_h2"][alpha].append(current_result)
                    if args.pkl_path is not None:
                        with open(args.pkl_path, 'wb') as outfile:
                            pickle.dump(results, outfile)

            if True:
                # Partitioned heritability with bins. We use alpha = -1 and alpha = 0
                bin_boundaries = [0, 0.0025, 0.01, 0.05, 0.5]
                for alpha in [-1, 0]:
                    if len(results["mut_ms_vary"][mut_factor][alpha]) <= seed_offset:
                        grms = []
                        snp_counts = []
                        for k in range(len(bin_boundaries) - 1):
                            low = bin_boundaries[k]
                            high = bin_boundaries[k+1]
                            if simulator_num_samples * high < 1 - 1e-6:
                                logging.info("Skipping GRM with range (" + str(low) + ", " + str(high) + "]")
                            else:
                                grm = arg_needle_lib.monte_carlo_arg_grm(
                                    arg, new_mu, seed, alpha, diploid, centering=False,
                                    min_maf=low, max_maf=high)
                                gower_center(grm)
                                grms.append(grm)
                                snp_counts.append(COUNT_PLACEHOLDER)
                        logging.info("Memory: {}".format(process.memory_info().rss))
                        current_result = gcta(args.gcta_path, args.base_tmp_dir, grms, pheno, snp_counts, args.pkl_path)
                        results["mut_ms_vary"][mut_factor][alpha].append(current_result)
                        if mut_factor == 0:
                            results["mut_ms"][alpha].append(current_result)
                        if args.pkl_path is not None:
                            with open(args.pkl_path, 'wb') as outfile:
                                pickle.dump(results, outfile)

    if use_arg_grm:
        if diploid:
            print("Computing diploid ARG GRMs")
        for alpha in test_alphas:
            if len(results["arg_h2"][alpha]) <= seed_offset:
                logging.info("Computing ARG-based GRM")
                arg_grm = arg_needle_lib.exact_arg_grm(arg, alpha=alpha, diploid=diploid, centering=True)
                logging.info("Done computing ARG-based GRM")
                if verbose:
                    logging.info("Gower-centered and row-column centered ARG kinship:")
                    logging.info(arg_grm[:5, :5])
                results["arg_h2"][alpha].append(gcta(args.gcta_path, args.base_tmp_dir, [arg_grm], pheno, [simulation.num_mutations], args.pkl_path))
                if args.pkl_path is not None:
                    with open(args.pkl_path, 'wb') as outfile:
                        pickle.dump(results, outfile)

    if args.pkl_path is not None:
        with open(args.pkl_path, 'wb') as outfile:
            pickle.dump(results, outfile)


logging.info("True h2: " + str(args.h2))
logging.info("True alpha: " + str(args.alpha))
logging.info(results)
