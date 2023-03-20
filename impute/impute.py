"""ARG-building followed by imputation / evaluation
"""

import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CURRENT_DIR, '../')) # for utils, simulator


# Python imports
import argparse
from datetime import datetime
import gzip
import logging
import numpy as np
import os
import pickle
import msprime
import shutil

# Our packages
import arg_needle_lib
from arg_needle import build_arg_simulation, add_default_arg_building_arguments

# Files from this repository
from common.simulator import Simulator
from common.utils import btime, ukb_sample, run_impute4, run_beagle5


logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

parser = argparse.ArgumentParser(description='Compare imputation frameworks.')
# Paths
parser.add_argument("--base_tmp_dir", help="Base temporary directory for intermediate output", action="store", default="../data/temp/")
parser.add_argument("--beagle5_path", help="Path to BEAGLE5 JAR", action="store",
    default=os.path.join(CURRENT_DIR, "../beagle5.jar"))
parser.add_argument("--impute4_path", help="Path to IMPUTE4 binary", action="store",
    default=os.path.join(CURRENT_DIR, "../impute4"))
parser.add_argument("--pkl_path", help="Pickle file path", action="store", default=None)
# Other arguments
parser.add_argument("--num_seeds", help="Number of seeds", action="store", default=10, type=int)
parser.add_argument("--sim_length", help="Simulation length", action="store", default=3e5, type=float)
parser.add_argument("--start_seed", help="Start seed", action="store", default=1, type=int)

add_default_arg_building_arguments(parser)
args = parser.parse_args()

args.mapfile = None
args.demofile = os.path.join(CURRENT_DIR, "../common/CEU2.demo")
args.asmc_decoding_file = os.path.join(
    CURRENT_DIR,
    "../common/decoding_quantities/30-100-2000.decodingQuantities.gz")
args.mu = 1.65e-8
args.rho = 1.2e-8

logging.info("Command-line args:")
args_to_print = vars(args)
for k in sorted(args_to_print):
    logging.info(k + ": " + str(args_to_print[k]))

# set up simulator with sequence + SNP samples
simulator = Simulator(args.mapfile, args.demofile,
    sample_size=args.num_sequence_samples + args.num_snp_samples,
    mu=None, rho=args.rho)

bin_thresholds = [0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
bin_upper = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

if args.pkl_path is not None and os.path.exists(args.pkl_path):
    with open(args.pkl_path, 'rb') as infile:
        dump_dict = pickle.load(infile)
        arg_sizes = dump_dict["arg_sizes"]
        timings = dump_dict["timings"]
        summary_impute = dump_dict["summary_impute"]
else:
    summary_impute = {"true_arg": [], "infer_arg_float": [], "infer_arg_zero_one": [],
                      "beagle5": [], "impute4": [],
                      "bin_lower": bin_thresholds, "bin_upper": bin_upper}

    arg_sizes = {"num_nodes": [], "num_edges": [], "num_trees": []}
    timings = {"simulation": [], "inference": [], "convert": [],
               "metrics": [], "metrics_part_two": []}

for seed_offset in range(args.num_seeds):
    seed = seed_offset + args.start_seed
    np.random.seed(seed)

    with btime(lambda x: timings["simulation"].append(x)):
        logging.info("Starting simulation " + str(seed))
        simulation = simulator.simulation(args.sim_length, random_seed=seed)
        simulation = msprime.mutate(simulation, rate=args.mu, random_seed=seed,
            model=msprime.InfiniteSites(alphabet=msprime.NUCLEOTIDES))
        if args.num_snp_samples > 0:
            snp_ids = ukb_sample(simulation, verbose=True)

    with btime(lambda x: timings["metrics"].append(x)):
        # run IMPUTE4
        if len(summary_impute["impute4"]) <= seed_offset:
            time_string = datetime.now().strftime("day-%Y-%m-%d-time-%H:%M:%S.%f") # ensure unique temp directories!
            impute_tmp_dir = os.path.join(args.base_tmp_dir, "impute4_" + time_string + "/")
            run_impute4(
                args.impute4_path, impute_tmp_dir,
                simulation, snp_ids, args.num_sequence_samples, args.num_snp_samples)

            # output is in out.gen.gz
            s_true = [[] for i in range(len(bin_thresholds))]
            s_infer = [[] for i in range(len(bin_thresholds))]
            snp_ids_index = 0
            with gzip.open(impute_tmp_dir + "out.gen.gz", 'rt') as in_file:
                for i, (line, variant) in enumerate(zip(in_file, simulation.variants())):
                    if snp_ids_index < len(snp_ids) and i == snp_ids[snp_ids_index]:
                        snp_ids_index += 1
                        maf = None
                    else:
                        row = variant.genotypes
                        af = row[args.num_sequence_samples:].mean()
                        maf = min(af, 1 - af)
                        maf_bin = np.searchsorted(bin_thresholds, maf, side="right") - 1

                        impute_result = [float(x) for x in line.strip('\n').split()[5:]]
                        result_array = np.array(impute_result)
                        haploid_dosages = 0.5*result_array[1::3] + result_array[2::3]

                        s_true[maf_bin].extend(row[args.num_sequence_samples:])
                        s_infer[maf_bin].extend(haploid_dosages)
                    if i % 1000 == 0:
                        logging.info("{} {}".format(i, maf))
            # Report MAF-stratified aggregate r^2
            aggregate_r2 = []
            for i in range(len(bin_thresholds)):
                r = np.corrcoef(s_true[i], s_infer[i])[0, 1]
                aggregate_r2.append(r*r)
                print("Aggregate r2 for [{:.3f}, {:.3f}): {:.4f}".format(
                    bin_thresholds[i], bin_upper[i], r*r))
            summary_impute["impute4"].append(aggregate_r2)
            shutil.rmtree(impute_tmp_dir)

        if len(summary_impute["beagle5"]) <= seed_offset:
            time_string = datetime.now().strftime("day-%Y-%m-%d-time-%H:%M:%S.%f") # ensure unique temp directories!
            beagle_tmp_dir = os.path.join(args.base_tmp_dir, "beagle5_" + time_string + "/")
            run_beagle5(
                args.beagle5_path, beagle_tmp_dir,
                simulation, snp_ids, args.num_sequence_samples, args.num_snp_samples)

            # Read in .csv results
            s_true = [[] for i in range(len(bin_thresholds))]
            s_infer = [[] for i in range(len(bin_thresholds))]
            snp_ids_index = 0
            with open(beagle_tmp_dir + "imputed.csv", 'r') as in_file:
                for i, (line, variant) in enumerate(zip(in_file, simulation.variants())):
                    if snp_ids_index < len(snp_ids) and i == snp_ids[snp_ids_index]:
                        snp_ids_index += 1
                        maf = None
                    else:
                        row = variant.genotypes
                        af = row[args.num_sequence_samples:].mean()
                        maf = min(af, 1 - af)
                        maf_bin = np.searchsorted(bin_thresholds, maf, side="right") - 1

                        impute_result = [float(x) for x in line.strip('\n').split()[3:]]

                        s_true[maf_bin].extend(row[args.num_sequence_samples:])
                        s_infer[maf_bin].extend(impute_result)
                    if i % 1000 == 0:
                        logging.info("{} {}".format(i, maf))
            # Report MAF-stratified aggregate r^2
            aggregate_r2 = []
            for i in range(len(bin_thresholds)):
                r = np.corrcoef(s_true[i], s_infer[i])[0, 1]
                aggregate_r2.append(r*r)
                print("Aggregate r2 for [{:.3f}, {:.3f}): {:.4f}".format(
                    bin_thresholds[i], bin_upper[i], r*r))
            summary_impute["beagle5"].append(aggregate_r2)
            shutil.rmtree(beagle_tmp_dir)

        if len(summary_impute["true_arg"]) <= seed_offset:
            arg_true = arg_needle_lib.tskit_to_arg(simulation)
            logging.info("Done converting true tree sequence to ARG")
            arg_true.populate_children_and_roots()
            logging.info("Done populating children and roots")

            s_true = [[] for i in range(len(bin_thresholds))]
            s_infer = [[] for i in range(len(bin_thresholds))]
            snp_ids_index = 0
            for i, variant in enumerate(simulation.variants()):
                if snp_ids_index < len(snp_ids) and i == snp_ids[snp_ids_index]:
                    snp_ids_index += 1
                    maf = None
                else:
                    row = variant.genotypes
                    af = row[args.num_sequence_samples:].mean()
                    maf = min(af, 1 - af)
                    maf_bin = np.searchsorted(bin_thresholds, maf, side="right") - 1
                    new_row = row.copy()
                    new_row[args.num_sequence_samples:] = -1
                    impute_input = new_row.tolist()
                    result = arg_needle_lib.impute(arg_true, variant.position, impute_input)

                    s_true[maf_bin].extend(row[args.num_sequence_samples:])
                    s_infer[maf_bin].extend(result[args.num_sequence_samples:])
                if i % 1000 == 0:
                    logging.info("{} {}".format(i, maf))
            aggregate_r2 = []
            for i in range(len(bin_thresholds)):
                r = np.corrcoef(s_true[i], s_infer[i])[0, 1]
                aggregate_r2.append(r*r)
                print("Aggregate r2 for [{:.3f}, {:.3f}): {:.4f}".format(
                    bin_thresholds[i], bin_upper[i], r*r))
            summary_impute["true_arg"].append(aggregate_r2)

    if len(summary_impute["infer_arg_float"]) <= seed_offset and len(summary_impute["infer_arg_zero_one"]) <= seed_offset:
        # run inference algorithm on simulation to get an ARG
        with btime(lambda x: timings["inference"].append(x)):
            arg, max_memory = build_arg_simulation(
                args, simulation, args.base_tmp_dir, snp_indices=snp_ids, mode="both")

        with btime(lambda x: timings["convert"].append(x)):
            arg_true = arg_needle_lib.tskit_to_arg(simulation)
            logging.info("Done converting true tree sequence to ARG")
            arg_true.populate_children_and_roots()
            logging.info("Done populating children and roots")

            arg.populate_children_and_roots()

        with btime(lambda x: timings["metrics_part_two"].append(x)):
            # Get sites, mask to -1, feed in, bin results, compute aggregate r^2 and report
            # Perform imputation for both inferred ARG and true ARG
            # We use sequence MAF for binning
            if False:
                s_true = [[] for i in range(len(bin_thresholds))]
                s_infer = [[] for i in range(len(bin_thresholds))]
                snp_ids_index = 0
                for i, variant in enumerate(simulation.variants()):
                    if snp_ids_index < len(snp_ids) and i == snp_ids[snp_ids_index]:
                        snp_ids_index += 1
                        maf = None
                    else:
                        row = variant.genotypes
                        af = row[args.num_sequence_samples:].mean()
                        maf = min(af, 1 - af)
                        maf_bin = np.searchsorted(bin_thresholds, maf, side="right") - 1
                        new_row = row.copy()
                        new_row[args.num_sequence_samples:] = -1
                        impute_input = new_row.tolist()
                        result = arg_needle_lib.impute(arg, variant.position, impute_input, old=True)

                        s_true[maf_bin].extend(row[args.num_sequence_samples:])
                        s_infer[maf_bin].extend(result[args.num_sequence_samples:])
                    if i % 1000 == 0:
                        logging.info("{} {}".format(i, maf))
                aggregate_r2 = []
                for i in range(len(bin_thresholds)):
                    r = np.corrcoef(s_true[i], s_infer[i])[0, 1]
                    aggregate_r2.append(r*r)
                    print("Aggregate r2 for [{:.3f}, {:.3f}): {:.4f}".format(
                        bin_thresholds[i], bin_upper[i], r*r))
                summary_impute["infer_arg_zero_one"].append(aggregate_r2)

            if True:
                s_true = [[] for i in range(len(bin_thresholds))]
                s_infer = [[] for i in range(len(bin_thresholds))]
                snp_ids_index = 0
                for i, variant in enumerate(simulation.variants()):
                    if snp_ids_index < len(snp_ids) and i == snp_ids[snp_ids_index]:
                        snp_ids_index += 1
                        maf = None
                    else:
                        row = variant.genotypes
                        af = row[args.num_sequence_samples:].mean()
                        maf = min(af, 1 - af)
                        maf_bin = np.searchsorted(bin_thresholds, maf, side="right") - 1
                        new_row = row.copy()
                        new_row[args.num_sequence_samples:] = -1
                        impute_input = new_row.tolist()
                        result = arg_needle_lib.impute(arg, variant.position, impute_input)

                        s_true[maf_bin].extend(row[args.num_sequence_samples:])
                        s_infer[maf_bin].extend(result[args.num_sequence_samples:])
                    if i % 1000 == 0:
                        logging.info("{} {}".format(i, maf))
                aggregate_r2 = []
                for i in range(len(bin_thresholds)):
                    r = np.corrcoef(s_true[i], s_infer[i])[0, 1]
                    aggregate_r2.append(r*r)
                    print("Aggregate r2 for [{:.3f}, {:.3f}): {:.4f}".format(
                        bin_thresholds[i], bin_upper[i], r*r))
                summary_impute["infer_arg_float"].append(aggregate_r2)

    dump_dict = {
        "args": args, # command-line args, not ARGs
        "arg_sizes": arg_sizes,
        "timings": timings,
        "summary_impute": summary_impute
    }
    if args.pkl_path is not None:
        with open(args.pkl_path, 'wb') as outfile:
            pickle.dump(dump_dict, outfile)


for seed_offset in range(args.num_seeds):
    seed = seed_offset + args.start_seed
    if True:
        logging.info("Seed " + str(seed) + ", aggregate r2 for ours / IMPUTE4 / BEAGLE5")
    else:
        logging.info("Seed " + str(seed) + ", aggregate r2 for true / inferred-zero-one / inferred-float / IMPUTE4 / BEAGLE5")
    for i in range(len(bin_thresholds)):
        if True:
            logging.info("Bin [{:.3f}, {:.3f}): {:.4f} / {:.4f} / {:.4f}".format(
                bin_thresholds[i], bin_upper[i],
                summary_impute["infer_arg_float"][seed_offset][i],
                summary_impute["impute4"][seed_offset][i],
                summary_impute["beagle5"][seed_offset][i]))
        else:
            logging.info("Bin [{:.3f}, {:.3f}): {:.4f} / {:.4f} / {:.4f} / {:.4f} / {:.4f}".format(
                bin_thresholds[i], bin_upper[i],
                summary_impute["true_arg"][seed_offset][i],
                summary_impute["infer_arg_zero_one"][seed_offset][i],
                summary_impute["infer_arg_float"][seed_offset][i],
                summary_impute["impute4"][seed_offset][i],
                summary_impute["beagle5"][seed_offset][i]))
    logging.info("")
