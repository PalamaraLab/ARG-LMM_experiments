"""Heritability and polygenic prediction with inferred ARGs.

We build ARGs using SNP data only, and compare against imputation
using num_reference_samples.

In a few places the string "snarg" (shorthand for "SNP-inferred ARG")
is used for saving results.
"""

import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CURRENT_DIR, '../')) # for utils, simulator


# Python imports
import argparse
from datetime import datetime
import logging
import numpy as np
import os
import pickle
import psutil; process = psutil.Process(os.getpid())
import shutil
import subprocess
import time

# Our packages
from arg_needle import build_arg_simulation, add_default_arg_building_arguments, normalize_arg
import arg_needle_lib
from arg_needle_lib import gower_center

# Files from this repository
from common.simulator import Simulator
from common.utils import btime, create_list_if_not_exists, ukb_sample, run_impute4
from lmm_utils import sim_pheno_new, get_allele_frequencies, get_allele_frequencies_gen
from lmm_utils import make_grm, make_grm_gen, gcta

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
parser.add_argument("--impute4_path", help="Path to IMPUTE4 binary", action="store",
    default=os.path.join(CURRENT_DIR, "../impute4"))
parser.add_argument("--pkl_path", help="Pickle file path", action="store", default=None)
# Other arguments
parser.add_argument("--alphas", help="Values of Speed et al. alpha for frequency weight", action="store", default="-5e-1", type=str)
parser.add_argument("--arg_grm", help="Direct ARG GRMs", action="store", default=0, type=int)
parser.add_argument("--diploid", help="Whether to do diploid", action="store", default=0, type=int)
parser.add_argument("--h2s", help="Values of h2 to be simulated using infinitesimal model", action="store", default="080", type=str)
parser.add_argument("--inference", help="Whether to run inference", action="store", default=1, type=int)
parser.add_argument("--mkl_num_threads", help="Number of MKL threads (default of 0 means not set)", action="store", default=0, type=int)
parser.add_argument("--mut_factors", help="log10(factor)*2 for mutation as a list", action="store", default="0,2", type=str)
parser.add_argument("--normalize", help="Whether to normalize", action="store", default=1, type=int)
parser.add_argument("--num_seeds", help="Number of seeds", action="store", default=1, type=int)
parser.add_argument("--num_reference_samples", help="Number of reference samples", action="store", default="500,1000", type=str)
parser.add_argument("--sim_length", help="Simulation length", action="store", default=5e6, type=float)
parser.add_argument("--start_seed", help="Start seed", action="store", default=1, type=int)
parser.add_argument("--test_alphas", help="Values of Speed et al. alpha for frequency weight", action="store", default="", type=str)
parser.add_argument("--verbose", help="Verbose printing", action="store", default=1, type=int)

add_default_arg_building_arguments(parser)
args = parser.parse_args()

if args.mkl_num_threads > 0:
    import mkl
    mkl.set_num_threads(args.mkl_num_threads)

args.mapfile = None
args.demofile = os.path.join(CURRENT_DIR, "../common/CEU2.demo")
args.asmc_decoding_file = os.path.join(
    CURRENT_DIR,
    "../common/decoding_quantities/30-100-2000.decodingQuantities.gz")
args.mu = 1.65e-8
args.rho = 1.2e-8

logging.info("MALLOC_MMAP_THRESHOLD_={} MALLOC_MMAP_MAX_={} MALLOC_ARENA_MAX={}".format(
    os.environ.get('MALLOC_MMAP_THRESHOLD_'),
    os.environ.get('MALLOC_MMAP_MAX_'),
    os.environ.get('MALLOC_ARENA_MAX'),
))

logging.info("Command-line args:")
args_to_print = vars(args)
for k in sorted(args_to_print):
    logging.info(k + ": " + str(args_to_print[k]))
verbose = (args.verbose != 0)
use_arg_grm = (args.arg_grm != 0)
diploid = (args.diploid != 0)
inference = (args.inference != 0)
normalize = (args.normalize != 0)

h2s = [int(h2_string)*0.01 for h2_string in args.h2s.split(',')]
h2_strings = [h2_string for h2_string in args.h2s.split(',')]
alphas = [float(alpha_string) for alpha_string in args.alphas.split(',')]
alpha_strings = [alpha_string for alpha_string in args.alphas.split(',')]
logging.info("h2s:")
logging.info(h2s)
logging.info("alphas:")
logging.info(alphas)

if args.num_reference_samples == "":
    num_reference_samples = []
    max_reference_samples = 0
else:
    num_reference_samples = [int(references) for references in args.num_reference_samples.split(',')]
    max_reference_samples = max(num_reference_samples)
logging.info("reference samples:")
logging.info(num_reference_samples)
assert args.num_sequence_samples == 0
assert args.num_snp_samples != 0

if diploid:
    assert args.num_snp_samples % 2 == 0
    for num_references in num_reference_samples:
        assert num_references % 2 == 0

if args.test_alphas != "":
    test_alphas = [float(alpha_string) for alpha_string in args.test_alphas.split(',')]
else:
    test_alphas = alphas

if args.mut_factors != "":
    mut_factors = [int(mut_factor) for mut_factor in args.mut_factors.split(',')]
else:
    mut_factors = []

if use_arg_grm:
    raise ValueError("Not currently supported")

if diploid:
    raise ValueError("Not currently supported")
simulator_num_samples = max_reference_samples + args.num_snp_samples

simulator = Simulator(args.mapfile, args.demofile,
    sample_size=simulator_num_samples,
    mu=args.mu, rho=args.rho)

if args.pkl_path is not None and os.path.exists(args.pkl_path):
    with open(args.pkl_path, 'rb') as infile:
        results = pickle.load(infile)
else:
    results = {}

logging.info("Starting data (possibly read from pickle file)")
logging.info(results)

for seed_offset in range(args.num_seeds):
    seed = seed_offset + args.start_seed
    np.random.seed(seed)
    logging.info("Starting simulation " + str(seed))
    simulation = simulator.simulation(args.sim_length, random_seed=seed)
    logging.info("{} nodes, {} edges, {} mutations".format(simulation.num_nodes, simulation.num_edges, simulation.num_mutations))
    logging.info("Memory: {}".format(process.memory_info().rss))

    snp_indices = ukb_sample(simulation, verbose)
    all_positions = np.array([variant.site.position for variant in simulation.variants()])
    snp_positions = set(all_positions[snp_indices])

    # Run IMPUTE4 before subsampling the true ARG
    # TODO: do something special when there are 0 sequences?
    time_string = datetime.now().strftime("day-%Y-%m-%d-time-%H:%M:%S.%f") # ensure unique temp directories!
    impute_tmp_dir = os.path.join(args.base_tmp_dir, "impute4_" + time_string + "/")
    for references in num_reference_samples:
        run_impute4(args.impute4_path, os.path.join(impute_tmp_dir, str(references) + "/"),
                    simulation, snp_indices, references, args.num_snp_samples, diploid)

    if max_reference_samples > 0:
        logging.info("Subsampling true ARG")
        snp_sample_ids = [i + max_reference_samples for i in range(args.num_snp_samples)]
        old_simulation = simulation
        simulation = simulation.simplify(samples=snp_sample_ids)
        logging.info("Done subsampling")

        subset_snp_indices = []
        for i, variant in enumerate(simulation.variants()):
            if variant.site.position in snp_positions:
                subset_snp_indices.append(i)
        subset_snp_indices = np.array(subset_snp_indices)
        logging.info("Original {} SNPs down to {} SNPs after subsampling".format(
            len(snp_indices), len(subset_snp_indices)))
    else:
        subset_snp_indices = snp_indices

    if inference:
        foo = []
        with btime(lambda x: foo.append(x)):
            arg, max_memory = build_arg_simulation(
                args, simulation, args.base_tmp_dir,
                snp_indices=subset_snp_indices, mode="array")

            if args.num_sequence_samples > 0 and args.num_snp_samples > 0:
                logging.info("Subsampling to only the SNP samples")
                arg_ts = arg_needle_lib.arg_to_tskit(arg)
                snp_sample_ids = [i + args.num_sequence_samples for i in range(args.num_snp_samples)]
                arg_ts = arg_ts.simplify(samples=snp_sample_ids)
                arg = arg_needle_lib.tskit_to_arg(arg_ts)
                logging.info("Done subsampling")
            else:
                logging.info("Not doing anything for subsampling")

            if normalize:
                logging.info("Normalizing inferred ARG")
                arg = normalize_arg(arg, args.demofile)
                logging.info("Done normalizing inferred ARG")

            arg_ts = arg_needle_lib.arg_to_tskit(arg)

        create_list_if_not_exists(results, ["time", "inference"])
        if len(results["time"]["inference"]) <= seed_offset:
            results["time"]["inference"].append(foo[0])

        create_list_if_not_exists(results, ["memory", "ts_plus_decoder"])
        if len(results["memory"]["ts_plus_decoder"]) <= seed_offset:
            results["memory"]["ts_plus_decoder"].append(max_memory)

        current_memory = process.memory_info().rss
        logging.info("Memory: {}".format(current_memory))
        create_list_if_not_exists(results, ["memory", "subsampled"])
        if len(results["memory"]["subsampled"]) <= seed_offset:
            results["memory"]["subsampled"].append(current_memory)

        logging.info(str(arg_ts.num_nodes) + " nodes, " + str(arg_ts.num_edges) + " edges")

    # Record basic stats, everything except for bitsets which we do at the end
    # in case of memory errors
    for key in [
        "sites", "trees", "snarg_trees", "snarg_nodes", "snarg_edges"
    ]:
        create_list_if_not_exists(results, [key])

    if len(results["sites"]) <= seed_offset:
        results["sites"].append(simulation.num_mutations)
    if len(results["trees"]) <= seed_offset:
        results["trees"].append(simulation.num_trees)

    if inference:
        if len(results["snarg_trees"]) <= seed_offset:
            results["snarg_trees"].append(arg_ts.num_trees)
        if len(results["snarg_nodes"]) <= seed_offset:
            results["snarg_nodes"].append(arg_ts.num_nodes)
        if len(results["snarg_edges"]) <= seed_offset:
            results["snarg_edges"].append(arg_ts.num_edges)

    if args.pkl_path is not None:
        with open(args.pkl_path, 'wb') as outfile:
            pickle.dump(results, outfile)

    # iterate over h2s and alphas
    pheno_dict = {}
    for h2_tuple in zip(h2s, h2_strings):
        for alpha_tuple in zip(alphas, alpha_strings):
            pheno, num_sites = sim_pheno_new([simulation], h2_tuple[0], alpha_tuple[0], seed, diploid)
            logging.info("Done simulating phenotype for h2={}, alpha={}".format(h2_tuple[0], alpha_tuple[0]))
            pheno_dict_key = "h{}_a{}".format(h2_tuple[1], alpha_tuple[1])
            pheno_dict[pheno_dict_key] = pheno

    # SNP GRM
    for alpha in test_alphas:
        grm = None
        for pheno_dict_key, pheno in pheno_dict.items():
            create_list_if_not_exists(results, ["snp_h2", alpha, pheno_dict_key])
            if len(results["snp_h2"][alpha][pheno_dict_key]) <= seed_offset:
                if grm is None:
                    logging.info("Computing GRM for alpha = {}".format(alpha))
                    grm = make_grm(simulation, subset_snp_indices, alpha, diploid)
                    gower_center(grm)
                    if verbose:
                        logging.info("Gower-centered SNP-estimated kinship with alpha = " + str(alpha) + ":")
                        logging.info(grm[:5, :5])
                    logging.info("Memory: {}".format(process.memory_info().rss))

                current_result = gcta(args.gcta_path, args.base_tmp_dir, [grm], pheno, [simulation.num_mutations], args.pkl_path)
                results["snp_h2"][alpha][pheno_dict_key].append(current_result)
                logging.info("Done with snp_h2, {}, {}".format(alpha, pheno_dict_key))
            if args.pkl_path is not None:
                with open(args.pkl_path, 'wb') as outfile:
                    pickle.dump(results, outfile)

    # Imputed GRM
    for references in num_reference_samples:
        reference_tmp_dir = impute_tmp_dir + str(references) + "/"
        for alpha in test_alphas:
            grm = None
            for pheno_dict_key, pheno in pheno_dict.items():
                create_list_if_not_exists(results, ["impute_h2", references, alpha, pheno_dict_key])
                if len(results["impute_h2"][references][alpha][pheno_dict_key]) <= seed_offset:
                    if grm is None:
                        logging.info("Computing GRM for alpha = {}".format(alpha))
                        grm = make_grm_gen(reference_tmp_dir + "out.gen.gz", None, alpha, diploid, gen_gz=True)
                        gower_center(grm)
                        if verbose:
                            logging.info("Gower-centered imputation-estimated kinship with alpha = " + str(alpha) + ":")
                            logging.info(grm[:5, :5])
                        logging.info("Memory: {}".format(process.memory_info().rss))

                    current_result = gcta(args.gcta_path, args.base_tmp_dir, [grm], pheno, [simulation.num_mutations], args.pkl_path)
                    results["impute_h2"][references][alpha][pheno_dict_key].append(current_result)
                    logging.info("Done with impute_h2, {}, {}".format(alpha, pheno_dict_key))
                if args.pkl_path is not None:
                    with open(args.pkl_path, 'wb') as outfile:
                        pickle.dump(results, outfile)
    # IMPUTE MAF-stratified, also covers the SNP case if we let references = 0
    for references in num_reference_samples:
        reference_tmp_dir = impute_tmp_dir + str(references) + "/"
        bin_boundaries = [0, 0.0025, 0.01, 0.05, 0.5]
        af = get_allele_frequencies_gen(reference_tmp_dir + "out.gen.gz", gen_gz=True)
        maf = np.minimum(af, 1 - af)
        for alpha in [-1, 0]:
            grms = None
            snp_counts = None
            for pheno_dict_key, pheno in pheno_dict.items():
                create_list_if_not_exists(
                    results,
                    ["impute_h2_ms", references, alpha, pheno_dict_key])

                if len(results["impute_h2_ms"][references][alpha][pheno_dict_key]) <= seed_offset:
                    if grms is None:
                        logging.info("Computing MAF-stratified GRMs for alpha = {}".format(alpha))
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
                                grm = make_grm_gen(reference_tmp_dir + "out.gen.gz", select, alpha, diploid, gen_gz=True)
                                gower_center(grm)
                                grms.append(grm)
                                snp_counts.append(select.shape[0])
                        logging.info("Memory: {}".format(process.memory_info().rss))

                    current_result = gcta(args.gcta_path, args.base_tmp_dir, grms, pheno, snp_counts, args.pkl_path)
                    results["impute_h2_ms"][references][alpha][pheno_dict_key].append(current_result)
                    logging.info("Done with impute_h2_ms, {}, {}, {}".format(references, alpha, pheno_dict_key))

                if args.pkl_path is not None:
                    with open(args.pkl_path, 'wb') as outfile:
                        pickle.dump(results, outfile)

    if len(num_reference_samples) > 0:
        shutil.rmtree(impute_tmp_dir)

    # Sequence GRM
    if False:
        for alpha in test_alphas:
            grm = None
            for pheno_dict_key, pheno in pheno_dict.items():
                create_list_if_not_exists(results, ["seq_h2", alpha, pheno_dict_key])
                if len(results["seq_h2"][alpha][pheno_dict_key]) <= seed_offset:
                    if grm is None:
                        logging.info("Computing GRM for alpha = {}".format(alpha))
                        grm = make_grm(simulation, None, alpha, diploid)
                        gower_center(grm)
                        if verbose:
                            logging.info("Gower-centered sequence-estimated kinship with alpha = " + str(alpha) + ":")
                            logging.info(grm[:5, :5])
                        logging.info("Memory: {}".format(process.memory_info().rss))

                    current_result = gcta(args.gcta_path, args.base_tmp_dir, [grm], pheno, [simulation.num_mutations], args.pkl_path)
                    results["seq_h2"][alpha][pheno_dict_key].append(current_result)
                    logging.info("Done with seq_h2, {}, {}".format(alpha, pheno_dict_key))
                if args.pkl_path is not None:
                    with open(args.pkl_path, 'wb') as outfile:
                        pickle.dump(results, outfile)

    if inference:
        # Do everything again with mutated data
        arg = arg_needle_lib.tskit_to_arg(arg_ts)
        arg.populate_children_and_roots()
        COUNT_PLACEHOLDER = 500 # needed as input for GCTA but unused
        for mut_factor in mut_factors:
            new_mu = args.mu * 10**(mut_factor / 2)
            logging.info("new_mu: " + str(new_mu))
            for alpha in test_alphas:
                grm = None
                for pheno_dict_key, pheno in pheno_dict.items():
                    create_list_if_not_exists(
                        results,
                        ["snarg_mut_h2_vary", mut_factor, alpha, pheno_dict_key])
                    if len(results["snarg_mut_h2_vary"][mut_factor][alpha][pheno_dict_key]) <= seed_offset:
                        if grm is None:
                            logging.info("Computing GRM for alpha = {}".format(alpha))
                            grm = arg_needle_lib.monte_carlo_arg_grm(
                                arg, new_mu, seed, alpha, diploid=False, centering=False)
                            gower_center(grm)
                            if verbose:
                                logging.info("Gower-centered mutation-estimated kinship with alpha = " + str(alpha) + ":")
                                logging.info(grm[:5, :5])
                            logging.info("Memory: {}".format(process.memory_info().rss))

                        current_result = gcta(args.gcta_path, args.base_tmp_dir, [grm], pheno, [COUNT_PLACEHOLDER], args.pkl_path)
                        results["snarg_mut_h2_vary"][mut_factor][alpha][pheno_dict_key].append(current_result)
                        logging.info("Done with snarg_mut_h2_vary, {}, {}, {}".format(mut_factor, alpha, pheno_dict_key))
                    if args.pkl_path is not None:
                        with open(args.pkl_path, 'wb') as outfile:
                            pickle.dump(results, outfile)

            # Partitioned heritability with bins. We use alpha = -1 and alpha = 0
            bin_boundaries = [0, 0.0025, 0.01, 0.05, 0.5]
            for alpha in [-1, 0]:
                grms = None
                snp_counts = None
                for pheno_dict_key, pheno in pheno_dict.items():
                    create_list_if_not_exists(
                        results,
                        ["snarg_mut_ms_vary", mut_factor, alpha, pheno_dict_key])
                    if len(results["snarg_mut_ms_vary"][mut_factor][alpha][pheno_dict_key]) <= seed_offset:
                        if grms is None:
                            logging.info("Computing MAF-stratified GRMs for alpha = {}".format(alpha))
                            grms = []
                            snp_counts = []
                            for k in range(len(bin_boundaries) - 1):
                                low = bin_boundaries[k]
                                high = bin_boundaries[k+1]
                                if args.num_snp_samples * high < 1 - 1e-6:
                                    logging.info("Skipping GRM with range (" + str(low) + ", " + str(high) + "]")
                                else:
                                    grm = arg_needle_lib.monte_carlo_arg_grm(
                                        arg, new_mu, seed, alpha, diploid=False, centering=False,
                                        min_maf=low, max_maf=high)
                                    gower_center(grm)
                                    grms.append(grm)
                                    snp_counts.append(COUNT_PLACEHOLDER)
                            logging.info("Memory: {}".format(process.memory_info().rss))

                        current_result = gcta(args.gcta_path, args.base_tmp_dir, grms, pheno, snp_counts, args.pkl_path)
                        results["snarg_mut_ms_vary"][mut_factor][alpha][pheno_dict_key].append(current_result)
                        logging.info("Done with snarg_mut_ms_vary, {}, {}, {}".format(mut_factor, alpha, pheno_dict_key))

                    if args.pkl_path is not None:
                        with open(args.pkl_path, 'wb') as outfile:
                            pickle.dump(results, outfile)

    create_list_if_not_exists(results, ["bitsets"])
    if len(results["bitsets"]) <= seed_offset:
        logging.info("Converting from ts to arg")
        arg = arg_needle_lib.tskit_to_arg(arg_ts)
        arg.populate_children_and_roots()
        num_bitsets = arg_needle_lib.write_bitsets_detailed(arg, count_only=True) # counts haploid bitsets
        results["bitsets"].append(num_bitsets)

    if args.pkl_path is not None:
        with open(args.pkl_path, 'wb') as outfile:
            pickle.dump(results, outfile)

logging.info("True h2 range: " + str(args.h2s))
logging.info("True alpha range: " + str(args.alphas))
logging.info(results)
