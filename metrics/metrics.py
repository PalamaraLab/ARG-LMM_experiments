"""Different ARG-building methods with evaluation
"""

import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CURRENT_DIR, '../')) # for utils, simulator


# Python imports
import argparse
from datetime import datetime
import logging
import msprime
import numpy as np
import os
import pickle
import tszip

# Our packages
import arg_needle_lib
from arg_needle_lib import rf_total_variation_stab, rf_resolve_stab, mutation_match_binned
from arg_needle_lib import kc2_tmrca_mse_stab, kc2_special_stab, kc2_length_stab
from arg_needle import build_arg_simulation, add_default_arg_building_arguments, normalize_arg

# Files from this repository
from common.simulator import Simulator
from common.utils import wrapped_tsinfer, wrapped_relate, ukb_sample, modify_data_return_new_ts
from common.utils import btime, collect_garbage, create_list_if_not_exists

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

parser = argparse.ArgumentParser(description='Infer ARGs and evaluate.')
# Paths
parser.add_argument("--base_tmp_dir", help="Base temporary directory for intermediate output", action="store",
    default=os.path.join(CURRENT_DIR, "../data/temp/"))
parser.add_argument("--pkl_path", help="Pickle file path", action="store", default=None)
parser.add_argument("--relate_bin_dir", help="Directory containing Relate / RelateFileFormats binaries", action="store",
    default=os.path.join(CURRENT_DIR, ".."))
# Other arguments
parser.add_argument("--algorithm", help="thread (ARG-Needle), upgma (ASMC-clust), relate, tsinfer, or tsinfer_sparse",
    action="store", default=None)
parser.add_argument("--condition", help="Simulation condition", action="store", default=0, type=int)
parser.add_argument("--genotyping_error", help="Genotyping error rate", action="store", default=0, type=float)
parser.add_argument("--kc_lambda", help="KC lambda", action="store", default=0.02, type=float)
parser.add_argument("--kc_merging", help="KC merging", action="store", default=0, type=int)
parser.add_argument("--normalize", help="Normalize inferred ARGs", action="store", default=1, type=int)
parser.add_argument("--num_seeds", help="Number of seeds", action="store", default=1, type=int)
parser.add_argument("--permutation_seed", help="Seed for random permutation", action="store", default=0, type=int)
parser.add_argument("--save_trees", help="Save .trees file", action="store", default=0, type=int)
parser.add_argument("--sim_length", help="Simulation length", action="store", default=3e5, type=float)
parser.add_argument("--stab_eval_only", help="Only use stab evaluation", action="store", default=0, type=int)
parser.add_argument("--start_seed", help="Start seed", action="store", default=1, type=int)
parser.add_argument("--tsinfer_resolve", help="Randomly resolve for tsinfer", action="store", default=0, type=int)


add_default_arg_building_arguments(parser)
args = parser.parse_args()

args.mapfile = None
# make sure to pass in --asmc_decoding_file corresponding to condition
# see metrics.sh for how this is done
if args.condition == 0:
    args.demo = "CEU2"
    args.rho = 1.2e-8
elif args.condition == 1:
    args.demo = "const15k"
    args.rho = 1.2e-8
elif args.condition == 2:
    args.demo = "const10k"
    args.rho = 1.2e-8
# elif args.condition == 3:
#     args.demo = "CEU2"
#     args.rho = 1.2e-7
# elif args.condition == 4:
#     args.demo = "CEU2"
#     args.rho = 1.2e-9
elif args.condition == 5:
    args.demo = "CEU2"
    args.rho = 2.4e-8
else:
    assert args.condition == 6
    args.demo = "CEU2"
    args.rho = 6.0e-9

args.demofile = os.path.join(CURRENT_DIR, "../common", args.demo + ".demo")
args.mu = 1.65e-8

logging.info("Command-line args:")
args_to_print = vars(args)
for k in sorted(args_to_print):
    logging.info(k + ": " + str(args_to_print[k]))

kc_merging = (args.kc_merging != 0)
save_trees = (args.save_trees != 0)
stab_eval_only = (args.stab_eval_only != 0)
tsinfer_resolve = (args.tsinfer_resolve != 0)
use_norm = (args.normalize != 0)

# set up simulator with sequence + SNP samples
simulator = Simulator(args.mapfile, args.demofile,
    sample_size=args.num_sequence_samples + args.num_snp_samples,
    mu=args.mu, rho=args.rho)

summary_sites = {}
summary_overlap = {}
summary_direct = {}
arg_sizes = {}
timings = {"simulation": [], "inference": [], "normalize": [], "convert": [],
           "kc_direct": [], "tmrca_direct": [], "site_errors": [],
           "metrics": [], "stabs": [], "mutation_overlap": []}

if args.permutation_seed > 0 and not args.algorithm.startswith("thread"):
    raise ValueError("Very strange to ask for a permutation when not threading. Comment out if intentional.")

for seed_offset in range(args.num_seeds):
    seed = seed_offset + args.start_seed
    np.random.seed(seed)

    with btime(lambda x: timings["simulation"].append(x)):
        logging.info("Starting simulation " + str(seed))
        simulation = simulator.simulation(args.sim_length, random_seed=seed)
        if args.num_snp_samples > 0:
            snp_ids = ukb_sample(simulation, verbose=True)
            mode = "array"
        else:
            snp_ids = None
            mode = "sequence"

    if args.genotyping_error > 0 or args.permutation_seed > 0:
        # save the true underlying simulation
        true_simulation = simulation
        if args.genotyping_error > 0:
            logging.info("Will use genotyping error rate of {}".format(args.genotyping_error))
        # generate the permutation if required
        if args.permutation_seed > 0:
            # permute samples using a separately constructed RNG
            if args.num_sequence_samples > 0 and args.num_snp_samples > 0:
                raise ValueError("Not expecting to permute with both sequencing and SNP samples")
            rng = np.random.default_rng(args.permutation_seed)
            permutation = rng.permutation(args.num_sequence_samples + args.num_snp_samples)
            logging.info("Will use sample permutation beginning with: {}".format(permutation[:10]))
        else:
            permutation = None
        logging.info("Creating new data according to genotyping error and/or permutation")
        # Note: we use the same random seed for genotyping error as for globally, but the seeding
        # happens in a scoped default_rng
        simulation = modify_data_return_new_ts(
            simulation, snp_ids, error_rate=args.genotyping_error, random_seed=seed, permutation=permutation)
        # save the true snp_ids while changing snp_ids to None so we don't subsample twice
        true_snp_ids = snp_ids
        snp_ids = None
        logging.info("Done creating new data")

    # run inference algorithm on simulation to get an ARG
    with btime(lambda x: timings["inference"].append(x)):
        max_memory = None
        tree_path_inferred = args.pkl_path[:-4] + ".trees.tsz"
        if save_trees and os.path.exists(tree_path_inferred):
            logging.info("Skipping inference and reading inferred ARG from file: {}".format(
                tree_path_inferred))
            arg_ts = tszip.decompress(tree_path_inferred)
        else:
            if args.algorithm in ["tsinfer", "tsinfer_sparse"]:
                keep_unary = False
                if args.algorithm.startswith("tsinfer_sparse"):
                    array_strategy = True
                else:
                    array_strategy = False
                if args.num_snp_samples == 0:
                    arg_ts = wrapped_tsinfer(simulation, keep_unary=keep_unary, array_strategy=array_strategy)
                else:
                    # Build a single ARG for sequence and SNP samples at SNP sites
                    arg_ts = wrapped_tsinfer(simulation, snp_ids=snp_ids, keep_unary=keep_unary, array_strategy=array_strategy)
            elif args.algorithm == "relate":
                time_string = datetime.now().strftime("day-%Y-%m-%d-time-%H:%M:%S.%f") # ensure unique temp directories!
                relate_tmp_dir = os.path.join(args.base_tmp_dir, "relate_" + time_string + "/")
                if args.num_snp_samples == 0:
                    arg_ts = wrapped_relate(args.relate_bin_dir, relate_tmp_dir, simulation,
                                            condition=args.condition)
                else:
                    # Build a single ARG for sequence and SNP samples at SNP sites
                    arg_ts = wrapped_relate(args.relate_bin_dir, relate_tmp_dir, simulation,
                                            condition=args.condition, snp_ids=snp_ids)
            else:
                assert args.algorithm in ["thread", "upgma"]
                time_dict = {"hash": 0, "asmc": 0, "smooth": 0, "thread": 0}
                arg, max_memory = build_arg_simulation(
                    args, simulation, args.base_tmp_dir,
                    snp_indices=snp_ids, mode=mode, time_dict=time_dict)
                arg_ts = arg_needle_lib.arg_to_tskit(arg)
                for key in time_dict:
                    create_list_if_not_exists(timings, [key]).append(time_dict[key])

            if args.genotyping_error > 0 or args.permutation_seed > 0:
                # set the simulation to be the true underlying simulation, not the one with errors
                simulation = true_simulation
                snp_ids = true_snp_ids  # used downstream for SNP error
            if args.permutation_seed > 0:
                # reverse the permutation used for order of threading
                logging.info("Undoing the permutation before proceeding")
                inverse_perm = np.argsort(permutation)
                arg_ts = arg_needle_lib.arg_to_tskit(arg_needle_lib.tskit_to_arg(arg_ts), sample_permutation=inverse_perm)
                logging.info("Done undoing the permutation")

    assert arg_ts.num_samples == simulation.num_samples

    with btime(lambda x: timings["convert"].append(x)):
        logging.info(str(arg_ts.num_nodes) + " nodes, " + str(arg_ts.num_edges) + " edges, converting to arg_needle_lib")
        create_list_if_not_exists(arg_sizes, ["num_nodes"]).append(arg_ts.num_nodes)
        create_list_if_not_exists(arg_sizes, ["num_edges"]).append(arg_ts.num_edges)
        create_list_if_not_exists(arg_sizes, ["num_trees"]).append(arg_ts.num_trees)

        memory_in_bytes = collect_garbage()
        create_list_if_not_exists(arg_sizes, ["memory"]).append(memory_in_bytes)
        create_list_if_not_exists(arg_sizes, ["max_memory"])
        if max_memory is not None:
            arg_sizes["max_memory"].append(max(max_memory, memory_in_bytes))
        else:
            arg_sizes["max_memory"].append(memory_in_bytes)

        arg = arg_needle_lib.tskit_to_arg(arg_ts)
        logging.info("Done converting inferred tree sequence to ARG")

        arg_true = arg_needle_lib.tskit_to_arg(simulation)
        logging.info("Done converting true tree sequence to ARG")

        tree_path_inferred = args.pkl_path[:-4] + ".trees.tsz"
        tszip.compress(arg_ts, tree_path_inferred)
        byte_size = os.path.getsize(tree_path_inferred)
        create_list_if_not_exists(arg_sizes, ["disk"]).append(byte_size)
        create_list_if_not_exists(arg_sizes, ["upgma_chunk_sites"]).append(args.asmc_clust_chunk_sites)
        if save_trees:
            # TODO: note that this assumes num_seeds = 1
            tree_path_true = args.pkl_path[:-4] + ".true.trees.tsz"
            tszip.compress(simulation, tree_path_true)
            if os.path.exists(args.pkl_path[:-4] + ".perf.pkl"):
                # Rewrite timings["inference"] and arg_sizes["memory"]
                perf_dict = pickle.load(open(args.pkl_path[:-4] + ".perf.pkl", "rb"))
                arg_sizes["max_memory"][seed_offset] = perf_dict["max_memory"][seed_offset]
                arg_sizes["memory"][seed_offset] = perf_dict["memory"][seed_offset]
                arg_sizes["upgma_chunk_sites"][seed_offset] = perf_dict["upgma_chunk_sites"][seed_offset]
                timings["inference"][seed_offset] = perf_dict["time_infer"][seed_offset]
                logging.info("Rewrote existing time and memory stats")
            else:
                # save performance stats
                perf_dict = {
                    "max_memory": arg_sizes["max_memory"],
                    "memory": arg_sizes["memory"],
                    "upgma_chunk_sites": arg_sizes["upgma_chunk_sites"],
                    "time_infer": timings["inference"]
                }
                with open(args.pkl_path[:-4] + ".perf.pkl", "wb") as perf_pkl_file:
                    pickle.dump(perf_dict, perf_pkl_file)
        else:
            os.remove(tree_path_inferred)

    # compare simulation and arg
    with btime(lambda x: timings["metrics"].append(x)):
        arg.populate_children_and_roots()
        arg_true.populate_children_and_roots()
        if not stab_eval_only:
            with btime(lambda x: timings["tmrca_direct"].append(x)):
                create_list_if_not_exists(
                    summary_direct, ["tmrca_mse"]).append(
                    arg_needle_lib.tmrca_mse(arg, arg_true))
                logging.info("Done with MSE metric")
            with btime(lambda x: timings["kc_direct"].append(x)):
                create_list_if_not_exists(
                    summary_direct, ["kc_topology"]).append(
                    arg_needle_lib.kc_topology(arg, arg_true))
                logging.info("Done with KC metric")
        with btime(lambda x: timings["stabs"].append(x)):
            arg.populate_children_and_roots()
            arg_true.populate_children_and_roots()
            if args.num_sequence_samples + args.num_snp_samples > 5000:
                stab_sample_list = []
            else:
                stab_sample_list = [int(5e3)]
            for stab_sample in stab_sample_list:
                stab_results = kc2_tmrca_mse_stab(arg, arg_true, stab_sample)
                create_list_if_not_exists(
                    summary_direct, ["kc_topology_stab", stab_sample]).append(stab_results["kc2"])
                create_list_if_not_exists(
                    summary_direct, ["tmrca_mse_stab", stab_sample]).append(stab_results["tmrca_mse"])

                results = kc2_length_stab(arg, arg_true, stab_sample, lambdas=[1, args.kc_lambda])
                create_list_if_not_exists(
                    summary_direct, ["kc_length_stab", stab_sample]).append(results[0]) # KC squared
                create_list_if_not_exists(
                    summary_direct, ["kc_mix_stab", stab_sample, args.kc_lambda]).append(
                    results[1]) # KC squared

                if kc_merging:
                    merge_fractions = ["0.05", "0.1", "0.2", "0.3", "0.4", "0.5", "0.75"]
                    for fraction in merge_fractions:
                        kc2 = kc2_special_stab(
                            arg, arg_true, stab_sample,
                            special_behavior="random_merge",
                            merge_fraction=float(fraction),
                            random_kc_seed=seed)
                        create_list_if_not_exists(
                            summary_direct, ["kc_topology_stab", "merge_random", fraction]).append(kc2)

                        kc2 = kc2_special_stab(
                            arg, arg_true, stab_sample,
                            special_behavior="heuristic_merge",
                            merge_fraction=float(fraction))
                        create_list_if_not_exists(
                            summary_direct, ["kc_topology_stab", "merge_heuristic", fraction]).append(kc2)

                    if tsinfer_resolve and args.algorithm.startswith("tsinfer"):
                        random_resolve_values = []
                        for random_id in range(10):
                            resolve_seed = seed + (random_id + 1) * 10  # for kicks
                            kc2 = kc2_special_stab(
                                arg, arg_true, stab_sample,
                                special_behavior="random_resolve",
                                random_kc_seed=resolve_seed)
                            random_resolve_values.append(kc2)
                        stab_string_resolve = str(stab_sample) + "_resolve"
                        create_list_if_not_exists(
                            summary_direct, ["kc_topology_stab", stab_string_resolve]).append(
                            np.mean(random_resolve_values))

            logging.info("Done with stab metrics")

        dump_dict = {
            "args": args, # command-line args, not ARGs
            "arg_sizes": arg_sizes,
            "summary_sites": summary_sites,
            "summary_direct": summary_direct,
            "summary_overlap": summary_overlap,
            "timings": timings
        }
        with open(args.pkl_path, 'wb') as outfile:
            pickle.dump(dump_dict, outfile)
        logging.info("Wrote pickle results in case of crash")

        with btime(lambda x: timings["mutation_overlap"].append(x)):
            for stab_sample in [int(5e3)]:
                stab_results = rf_total_variation_stab(
                    arg_true, arg, stab_sample,
                    may_contain_polytomies=args.algorithm.startswith("tsinfer"))

                for key in ["total_variation"]:
                    create_list_if_not_exists(
                        summary_overlap, [key, stab_sample]).append(stab_results[key])
                create_list_if_not_exists(
                    summary_overlap, ["robinson_foulds", stab_sample]).append(
                    stab_results["scaled_robinson_foulds"])

                if tsinfer_resolve and args.algorithm.startswith("tsinfer"):
                    resolve_seeds = []
                    for random_id in range(1):
                        resolve_seeds.append(seed + (random_id + 1) * 10)  # for kicks

                    stab_results = rf_resolve_stab(arg, arg_true, stab_sample, resolve_seeds)
                    stab_string_resolve = str(stab_sample) + "_resolve"
                    create_list_if_not_exists(
                        summary_overlap, ["robinson_foulds", stab_string_resolve]).append(
                        stab_results["scaled_robinson_foulds"])

                sim_num_samples = simulation.num_samples
                thresholds = [2, 3, 5, 10, 20, 50, sim_num_samples]
                for i in range(len(thresholds) - 1):
                    logging.info("Computing MAC-stratified TV with {} <= MAC < {}".format(
                        thresholds[i], thresholds[i+1]))
                    stab_results = rf_total_variation_stab(
                        arg_true, arg, stab_sample, min_mac=thresholds[i], max_mac=thresholds[i+1],
                        may_contain_polytomies=args.algorithm.startswith("tsinfer"))
                    range_key = "{}-{}".format(thresholds[i], thresholds[i+1])
                    create_list_if_not_exists(
                        summary_overlap, ["total_variation_stratify", stab_sample, range_key]).append(
                        stab_results["total_variation"])

                    if tsinfer_resolve and args.algorithm.startswith("tsinfer"):
                        resolve_seeds = []
                        for random_id in range(1):
                            resolve_seeds.append(seed + (random_id + 1) * 10)  # for kicks

                        stab_results = rf_resolve_stab(
                            arg, arg_true, stab_sample, resolve_seeds, min_mac=thresholds[i], max_mac=thresholds[i+1])

            logging.info("Done with mutation overlap metrics")

            dump_dict = {
                "args": args, # command-line args, not ARGs
                "arg_sizes": arg_sizes,
                "summary_sites": summary_sites,
                "summary_direct": summary_direct,
                "summary_overlap": summary_overlap,
                "timings": timings
            }
            with open(args.pkl_path, 'wb') as outfile:
                pickle.dump(dump_dict, outfile)
            logging.info("Wrote pickle results in case of crash")

        dump_dict = {
            "args": args, # command-line args, not ARGs
            "arg_sizes": arg_sizes,
            "summary_sites": summary_sites,
            "summary_direct": summary_direct,
            "summary_overlap": summary_overlap,
            "timings": timings
        }
        with open(args.pkl_path, 'wb') as outfile:
            pickle.dump(dump_dict, outfile)
        logging.info("Wrote pickle results in case of crash")

        # Normalized ARG metrics, first set
        if use_norm:
            with btime(lambda x: timings["normalize"].append(x)):
                norm_arg = normalize_arg(
                    arg_needle_lib.tskit_to_arg(arg_ts), args.demofile)

            norm_arg.populate_children_and_roots()

            norm_ts = arg_needle_lib.arg_to_tskit(norm_arg)
            logging.info("Done converting normalized ARG to tree sequence")

            if save_trees:
                logging.info("Saving normalized trees")
                tree_path_inferred_norm = args.pkl_path[:-4] + ".norm.trees.tsz"
                tszip.compress(norm_ts, tree_path_inferred_norm)

            if args.num_sequence_samples + args.num_snp_samples > 5000:
                stab_sample_list = []
            else:
                stab_sample_list = [int(5e3)]
            for stab_sample in stab_sample_list:
                stab_results = kc2_tmrca_mse_stab(norm_arg, arg_true, stab_sample)
                create_list_if_not_exists(
                    summary_direct, ["norm_tmrca_mse_stab", stab_sample]).append(
                    stab_results["tmrca_mse"])

                results = kc2_length_stab(norm_arg, arg_true, stab_sample, lambdas=[1, args.kc_lambda])
                create_list_if_not_exists(
                    summary_direct, ["norm_kc_length_stab", stab_sample]).append(
                    results[0]) # KC squared
                create_list_if_not_exists(
                    summary_direct, ["norm_kc_mix_stab", stab_sample, args.kc_lambda]).append(
                    results[1]) # KC squared

            for stab_sample in [int(5e3)]:
                stab_results = rf_total_variation_stab(
                    arg_true, norm_arg, stab_sample,
                    may_contain_polytomies=args.algorithm.startswith("tsinfer"))

                for key in ["norm_total_variation"]:
                    # remove the "norm_" prefix
                    create_list_if_not_exists(
                        summary_overlap, [key, stab_sample]).append(stab_results[key[5:]])

                sim_num_samples = simulation.num_samples
                thresholds = [2, 3, 5, 10, 20, 50, sim_num_samples]
                for i in range(len(thresholds) - 1):
                    logging.info("Computing MAC-stratified TV with {} <= MAC < {}".format(
                        thresholds[i], thresholds[i+1]))
                    stab_results = rf_total_variation_stab(
                        arg_true, norm_arg, stab_sample, min_mac=thresholds[i], max_mac=thresholds[i+1],
                        may_contain_polytomies=args.algorithm.startswith("tsinfer"))
                    range_key = "{}-{}".format(thresholds[i], thresholds[i+1])
                    create_list_if_not_exists(
                        summary_overlap, ["norm_total_variation_stratify", stab_sample, range_key]).append(
                        stab_results["total_variation"])

            logging.info("Done with mutation overlap metrics")

            dump_dict = {
                "args": args, # command-line args, not ARGs
                "arg_sizes": arg_sizes,
                "summary_sites": summary_sites,
                "summary_direct": summary_direct,
                "summary_overlap": summary_overlap,
                "timings": timings
            }
            with open(args.pkl_path, 'wb') as outfile:
                pickle.dump(dump_dict, outfile)
            logging.info("Wrote pickle results in case of crash")
        else:
            timings["normalize"].append(0)

        # Errors / mutational consistency
        with btime(lambda x: timings["site_errors"].append(x)):
            sim_num_samples = simulation.num_samples
            assert sim_num_samples > 200
            thresholds = [2, 3, 5, 10, 20, 50]

            binned_errors, binned_diffs, binned_sites, errors, diffs = mutation_match_binned(
                arg, simulation, thresholds)
            create_list_if_not_exists(summary_sites, ["errors"]).append(binned_errors)
            create_list_if_not_exists(summary_sites, ["diffs"]).append(binned_diffs)
            create_list_if_not_exists(summary_sites, ["sites"]).append(binned_sites)
            create_list_if_not_exists(summary_sites, ["error_rate"]).append(
                sum(binned_errors) / sum(binned_sites))
            logging.info("Seed " + str(seed) + ", " + str(sum(binned_errors)) + \
                " errors out of " + str(sum(binned_sites)) + " sites")

            # Work on just the SNPs that have been sampled
            if args.num_snp_samples > 0:
                snp_sites = len(snp_ids)
                snp_errors = np.sum(np.array(errors)[snp_ids])
                snp_diffs = np.sum(np.array(diffs)[snp_ids])
                create_list_if_not_exists(summary_sites, ["snp_errors"]).append(snp_errors)
                create_list_if_not_exists(summary_sites, ["snp_diffs"]).append(snp_diffs)
                create_list_if_not_exists(summary_sites, ["snp_sites"]).append(snp_sites)
                create_list_if_not_exists(summary_sites, ["snp_error_rate"]).append(
                    snp_errors / snp_sites)

            if tsinfer_resolve and args.algorithm.startswith("tsinfer"):
                binned_errors, binned_diffs, binned_sites, _, _ = mutation_match_binned(
                    arg, simulation, thresholds, resolve_reps=1)
                create_list_if_not_exists(summary_sites, ["errors_resolve"]).append(binned_errors)
                create_list_if_not_exists(summary_sites, ["diffs_resolve"]).append(binned_diffs)
                create_list_if_not_exists(summary_sites, ["sites_resolve"]).append(binned_sites)
                create_list_if_not_exists(summary_sites, ["error_rate_resolve"]).append(
                    sum(binned_errors) / sum(binned_sites))
                logging.info("Seed " + str(seed) + ", " + str(sum(binned_errors)) + \
                    " resolve errors out of " + str(sum(binned_sites)) + " sites")

            # Fresh mutations metric
            mut_simulation = msprime.mutate(simulation, rate=args.mu, random_seed=seed)
            old_simulation = simulation
            simulation = mut_simulation

            binned_errors, binned_diffs, binned_sites, _, _ = mutation_match_binned(
                arg, simulation, thresholds)
            create_list_if_not_exists(summary_sites, ["fresh_errors"]).append(binned_errors)
            create_list_if_not_exists(summary_sites, ["fresh_diffs"]).append(binned_diffs)
            create_list_if_not_exists(summary_sites, ["fresh_sites"]).append(binned_sites)
            create_list_if_not_exists(summary_sites, ["fresh_error_rate"]).append(
                sum(binned_errors) / sum(binned_sites))
            logging.info("Seed " + str(seed) + ", " + str(sum(binned_errors)) + \
                " fresh errors out of " + str(sum(binned_sites)) + " sites")

            if tsinfer_resolve and args.algorithm.startswith("tsinfer"):
                binned_errors, binned_diffs, binned_sites, _, _ = mutation_match_binned(
                    arg, simulation, thresholds, resolve_reps=1)
                create_list_if_not_exists(
                    summary_sites, ["fresh_errors_resolve"]).append(binned_errors)
                create_list_if_not_exists(
                    summary_sites, ["fresh_diffs_resolve"]).append(binned_diffs)
                create_list_if_not_exists(
                    summary_sites, ["fresh_sites_resolve"]).append(binned_sites)
                create_list_if_not_exists(
                    summary_sites, ["fresh_error_rate_resolve"]).append(
                    sum(binned_errors) / sum(binned_sites))
                logging.info("Seed " + str(seed) + ", " + str(sum(binned_errors)) + \
                    " fresh resolve errors out of " + str(sum(binned_sites)) + " sites")

    dump_dict = {
        "args": args, # command-line args, not ARGs
        "arg_sizes": arg_sizes,
        "summary_sites": summary_sites,
        "summary_direct": summary_direct,
        "summary_overlap": summary_overlap,
        "timings": timings
    }
    with open(args.pkl_path, 'wb') as outfile:
        pickle.dump(dump_dict, outfile)
    logging.info("Wrote final pickle results")

print(dump_dict)
