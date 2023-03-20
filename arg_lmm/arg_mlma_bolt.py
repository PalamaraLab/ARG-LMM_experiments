"""Polygenic phenotype plus a rare variant, with GWAS / LOCO.

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
import glob
import logging
import numpy as np
import os
import pandas as pd
import pickle
import subprocess
import shutil

# Our packages
import arg_needle_lib
from arg_needle import build_arg_simulation, add_default_arg_building_arguments

# Files from this repository
from common.simulator import Simulator
from common.utils import ukb_sample, collect_garbage, create_list_if_not_exists, run_impute4, phase_data_return_new_ts
from lmm_utils import sim_pheno_new, write_plink_new, write_pheno, arg_write_bgen, get_allele_counts


logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

parser = argparse.ArgumentParser(description='Simulate ARGs and GRM.')
# Paths
parser.add_argument("--base_tmp_dir", help="Base temporary directory for intermediate output", action="store",
    default=os.path.join(CURRENT_DIR, "../data/temp/"))
parser.add_argument("--beagle5_path", help="Path to BEAGLE5 JAR", action="store",
    default=os.path.join(CURRENT_DIR, "../beagle5.jar"))
parser.add_argument("--bolt_path", help="Path to BOLT binary", action="store",
    default=os.path.join(CURRENT_DIR, "../bolt"))
parser.add_argument("--gcta_path", help="Path to GCTA binary", action="store",
    default=os.path.join(CURRENT_DIR, "../gcta"))
parser.add_argument("--impute4_path", help="Path to IMPUTE4 binary", action="store",
    default=os.path.join(CURRENT_DIR, "../impute4"))
parser.add_argument("--pkl_path", help="Pickle file path", action="store", default=None)
parser.add_argument("--plink2_path", help="Path to PLINK2 binary", action="store",
    default=os.path.join(CURRENT_DIR, "../plink2"))
# Other arguments
parser.add_argument("-a", "--alpha", help="Value of Speed et al. alpha for frequency weight", action="store", default=-0.5, type=float)
parser.add_argument("--diploid", help="Whether to do diploid", action="store", default=0, type=int)
parser.add_argument("--h2", help="Value of h2 to be simulated using infinitesimal model", action="store", default=0.5, type=float)
parser.add_argument("--mkl_num_threads", help="Number of MKL threads (default of 0 means not set)", action="store", default=0, type=int)
parser.add_argument("--num_chromosomes", help="Number of chromosomes", action="store", default=22, type=int)
parser.add_argument("--num_reference_samples", help="Number of reference samples", action="store", default=1000, type=int)
parser.add_argument("--num_seeds", help="Number of seeds", action="store", default=1, type=int)
parser.add_argument("--phasing_error", help="Whether to unphase / rephase", action="store", default=0, type=int)
parser.add_argument("--rare_beta", help="Effect size of rare variant", action="store", default=1e-2, type=float)
parser.add_argument("--rare_maf", help="MAF of rare variant", action="store", default=0.005, type=float)
parser.add_argument("--rare_sim_length", help="Length of special rare variant chromosome", action="store", default=1e6, type=float)
parser.add_argument("--sim_length", help="Length per chromosome", action="store", default=1e6, type=float)
parser.add_argument("--start_seed", help="Start seed", action="store", default=1, type=int)
parser.add_argument("--test_alphas", help="Value of Speed et al. alpha for frequency weight", action="store", default="0,-0.5,-1", type=str)
parser.add_argument("--verbose", help="Verbose printing", action="store", default=1, type=int)

add_default_arg_building_arguments(parser)
args = parser.parse_args()
assert args.num_reference_samples >= args.num_sequence_samples

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

logging.info("Command-line args:")
args_to_print = vars(args)
for k in sorted(args_to_print):
    logging.info(k + ": " + str(args_to_print[k]))
verbose = (args.verbose != 0)
test_alphas = [float(alpha_string) for alpha_string in args.test_alphas.split(',')]

diploid = (args.diploid != 0)
if diploid:
    assert args.num_snp_samples % 2 == 0
    assert args.num_sequence_samples % 2 == 0
    assert args.num_reference_samples % 2 == 0
phasing_error = (args.phasing_error != 0)
if phasing_error and not diploid:
    raise ValueError("Phasing error is only possible in diploid simulations")

rare_allele_count = max(1, np.round(args.rare_maf * args.num_snp_samples))
logging.info("Rare MAC: {}".format(rare_allele_count))
if True:
    rare_maf_exact = rare_allele_count / args.num_snp_samples
    rare_h2 = rare_maf_exact * (1 - rare_maf_exact) * (args.rare_beta) ** 2
    logging.info("h2 from rare variant: {}".format(rare_h2))
    if (args.h2 + rare_h2 >= 1):
        raise ValueError("h2 from background + rare effect is too large, exiting")

simulator = Simulator(args.mapfile, args.demofile,
    sample_size=args.num_reference_samples + args.num_snp_samples,
    mu=args.mu, rho=args.rho)

if diploid:
    num_snp_individuals = args.num_snp_samples // 2
else:
    num_snp_individuals = args.num_snp_samples

if args.pkl_path is not None and os.path.exists(args.pkl_path):
    with open(args.pkl_path, 'rb') as infile:
        results = pickle.load(infile)
else:
    results = {}

if True:
    if "bitsets" in results and "impute4" in results["bitsets"]:
        del results["bitsets"]["impute4"]
    if "power" in results and "impute4" in results["power"]:
        del results["power"]["impute4"]

logging.info("Starting data (possibly read from pickle file)")
logging.info(results)


def bolt_bonferroni(temp_base_path, bgen_base_path, min_maf=0.01, max_maf=0.5):
    """Run BOLT using SNP data, testing various types of variants

    Arguments:
        temp_base_path: should contain .phen phenotype and .bed/.bim/.fam
            genotypes (assuming bed_suffix is "")
        bgen_base_path: bgen_base_path + ".bgen" / ".sample" will contain BGEN data
        min_maf: minimum MAF (inclusive) for thresholded tests
        max_maf: maximum MAF (inclusive) for thresholded tests

    Returns:
        Does testing all variants reach Bonferroni significance?
        Does testing MAF thresholded reach Bonferroni significance?
        What is the minimum p from testing all variants?
        What is the minimum p from testing MAF thresholded variants?
    """
    logging.info("Running BOLT")

    bolt_parameters = [
        args.bolt_path,
        "--lmmInfOnly",
        # "--lmm",
        "--LDscoresUseChip",
        "--verboseStats",
        "--bfile",
        temp_base_path,
        "--phenoFile",
        temp_base_path + ".phen",
        "--phenoCol",
        "pheno",
        "--statsFile",
        temp_base_path + ".stats"
    ]

    if bgen_base_path is None:
        subprocess.run(bolt_parameters)
        bolt_stats = pd.read_csv(temp_base_path + '.stats', sep='\t')
        bolt_stats = bolt_stats[bolt_stats['CHR'] == 1]
    else:
        bolt_parameters += [
            "--bgenFile",
            bgen_base_path + ".bgen",
            "--sampleFile",
            bgen_base_path + ".sample",
            "--statsFileBgenSnps",
            temp_base_path + ".bgen.stats"
        ]
        subprocess.run(bolt_parameters)
        bolt_stats = pd.read_csv(temp_base_path + '.bgen.stats', sep='\t')
        bolt_stats = bolt_stats[bolt_stats['CHR'] == 1]

    results = {}
    for bolt_output_key, results_key in zip(['P_LINREG', 'P_BOLT_LMM_INF'], ['linreg', 'lmm_inf']):
        all_p = np.array(bolt_stats[[bolt_output_key]])
        all_af = np.array(bolt_stats[['A1FREQ']])
        num_tests = np.count_nonzero(~np.isnan(all_p))
        min_p = np.nanmin(all_p)
        bonferroni_p = 5e-2 / num_tests
        logging.info("Number of tests = {}".format(num_tests))
        logging.info("Smallest non-nan p = {}".format(min_p))
        logging.info("Bonferroni p = {}".format(bonferroni_p))
        first_result = int(min_p < bonferroni_p)
        first_min_p = min_p

        logging.info("Computing for {} <= MAF <= {}".format(min_maf, max_maf))
        all_maf = np.minimum(all_af, 1 - all_af)
        # We added the 1e-9 because of precision issues
        subset_p = all_p[(all_maf >= (min_maf - 1e-9)) & (all_maf <= (max_maf + 1e-9))]
        num_tests = np.count_nonzero(~np.isnan(subset_p))
        min_p = np.nanmin(subset_p)
        bonferroni_p = 5e-2 / num_tests
        logging.info("Number of tests = {}".format(num_tests))
        logging.info("Smallest non-nan p = {}".format(min_p))
        logging.info("Bonferroni p = {}".format(bonferroni_p))
        second_result = int(min_p < bonferroni_p)
        second_min_p = min_p
        print("Searchable string", bolt_output_key, results_key,
            first_result, second_result, first_min_p, second_min_p)
        results[results_key] = (first_result, second_result, first_min_p, second_min_p)

    return results


for seed_offset in range(args.num_seeds):
    simulations = []
    snp_indices_list = []
    seed = seed_offset + args.start_seed
    logging.info("Starting simulation with seed " + str(seed))
    for chromosome in range(args.num_chromosomes):
        msprime_seed = seed * args.num_chromosomes + chromosome
        sequence_sample_ids = [i for i in range(args.num_sequence_samples)]
        snp_sample_ids = [args.num_reference_samples + i for i in range(args.num_snp_samples)]
        if chromosome == 0:
            simulation_full = simulator.simulation(args.rare_sim_length, random_seed=msprime_seed)
            chr1_with_sequence = simulation_full.simplify(samples=sequence_sample_ids + snp_sample_ids)
        else:
            simulation_full = simulator.simulation(args.sim_length, random_seed=msprime_seed)
        simulation = simulation_full.simplify(samples=snp_sample_ids)
        simulations.append(simulation)
        del simulation_full

    logging.info("Building phenotype in a memory-efficient manner")
    if len(simulations) > 1:
        # this line seeds np.random
        # heritability 1, no effects on chromosome 1!
        pheno, num_sites = sim_pheno_new(simulations[1:], 1, args.alpha, seed, diploid)
    else:
        # this line seeds np.random
        # heritability 1!
        logging.info("Only one chromosome, putting effects on chromosome 1")
        pheno, num_sites = sim_pheno_new(simulations, 1, args.alpha, seed, diploid)
    if verbose:
        logging.info(str(num_sites) + " sites used to build phenotype")
    create_list_if_not_exists(results, ["sites"])
    if len(results["sites"]) <= seed_offset:
        results["sites"].append(num_sites)

    # plant a single large effect on chromosome 1, rare is usually 0.001 - 0.005
    chr1_positions = np.array([variant.site.position for variant in simulations[0].variants()])
    allele_count = get_allele_counts(simulations[0], None)
    minor_allele_count = np.minimum(allele_count, args.num_snp_samples - allele_count)
    correct_alleles = np.nonzero(minor_allele_count == rare_allele_count)[0]
    choice_id = np.random.choice(correct_alleles, 1, replace=False)[0]
    logging.info("Chose 1 out of {} SNPs with minor allele count={}".format(len(correct_alleles), rare_allele_count))
    for i, variant in enumerate(simulations[0].variants()):
        if i == choice_id:
            choice_geno = variant.genotypes
            if diploid:
                choice_geno = choice_geno.reshape((num_snp_individuals, 2)).sum(axis=-1)
    choice_pos = chr1_positions[choice_id]
    if False:
        # pheno should have variance 1, effect_size ends up being positive in this case
        effect_size = args.rare_h2_fraction / (1 - args.rare_h2_fraction) / np.std(choice_geno, ddof=1)
        pheno += choice_geno * effect_size

        # normalize pheno to have mean 0 and var = h2
        pheno -= np.mean(pheno)
        pheno /= np.std(pheno, ddof=1)
        pheno *= np.sqrt(args.h2)

        # sample environmental component with variance 1-h2 and add it to phenotype
        pheno += np.random.randn(num_snp_individuals) * np.sqrt(1-args.h2)

        # normalize it all
        pheno -= np.mean(pheno)
        pheno /= np.std(pheno, ddof=1)
    elif False:
        pheno *= np.sqrt(args.h2)
        pheno += np.random.randn(num_snp_individuals) * np.sqrt(1-args.h2)
        pheno += choice_geno * args.rare_beta

        # normalize it all
        pheno -= np.mean(pheno)
        pheno /= np.std(pheno, ddof=1)
    else:
        pheno *= np.sqrt(args.h2)
        pheno += choice_geno * args.rare_beta
        pheno += np.random.randn(num_snp_individuals) * np.sqrt(1-args.h2-rare_h2)

        # normalize it all
        pheno -= np.mean(pheno)
        logging.info("Variance before standardizing: {}".format(np.var(pheno, ddof=1)))
        pheno /= np.std(pheno, ddof=1)

    logging.info("Done building phenotype")

    for chromosome in range(args.num_chromosomes):
        msprime_seed = seed * args.num_chromosomes + chromosome
        simulation = simulations[chromosome]
        # seed np.random explicitly rather than rely on previous setting
        np.random.seed(msprime_seed)
        snp_indices = ukb_sample(simulation, verbose)
        snp_indices_list.append(snp_indices)

    # Map the SNP indices for chromosome 1 to the sequence + SNP simulation
    chr1_snp_positions = set(chr1_positions[snp_indices_list[0]])
    chr1_with_sequence_snp_indices = []
    for i, variant in enumerate(chr1_with_sequence.variants()):
        if variant.site.position in chr1_snp_positions:
            chr1_with_sequence_snp_indices.append(i)
    chr1_with_sequence_snp_indices = np.array(chr1_with_sequence_snp_indices)
    logging.info("{} vs. {} indices, should match".format(
        len(snp_indices_list[0]), len(chr1_with_sequence_snp_indices)))

    # Write all SNPs for BOLT-LMM
    if True:
        temp_base_path = write_plink_new(
            args.plink2_path, args.base_tmp_dir, simulations, snp_indices_list, args.pkl_path, diploid)
    else:
        temp_base_path = write_plink_new(
            args.plink2_path, args.base_tmp_dir, simulations[:1], snp_indices_list[:1], args.pkl_path, diploid)

    write_pheno(pheno, temp_base_path + '.phen', header=True)
    logging.info("Done writing phenotype")

    # SNP
    if args.num_sequence_samples == 0 and not phasing_error:
        create_list_if_not_exists(results, ["bitsets", "snp"])
        create_list_if_not_exists(results, ["power", "snp", "direct"])
        create_list_if_not_exists(results, ["power", "snp", "snp_lmm"])
        if (len(results["bitsets"]["snp"]) <= seed_offset) or (
            len(results["power"]["snp"]["direct"]) <= seed_offset) or (
            len(results["power"]["snp"]["snp_lmm"]) <= seed_offset):

            if len(results["bitsets"]["snp"]) <= seed_offset:
                results["bitsets"]["snp"].append(len(snp_indices_list[0]))

            bolt_results = None
            if len(results["power"]["snp"]["direct"]) <= seed_offset:
                if bolt_results is None:
                    bolt_results = bolt_bonferroni(temp_base_path, None)
                results["power"]["snp"]["direct"].append(bolt_results["linreg"])

            if len(results["power"]["snp"]["snp_lmm"]) <= seed_offset:
                if bolt_results is None:
                    bolt_results = bolt_bonferroni(temp_base_path, None)
                results["power"]["snp"]["snp_lmm"].append(bolt_results["lmm_inf"])

    # IMPUTE4
    if args.num_sequence_samples > 0 and not phasing_error:
        create_list_if_not_exists(results, ["bitsets", "impute4"])
        create_list_if_not_exists(results, ["power", "impute4", "direct"])
        create_list_if_not_exists(results, ["power", "impute4", "snp_lmm"])
        if (len(results["bitsets"]["impute4"]) <= seed_offset) or (
            len(results["power"]["impute4"]["direct"]) <= seed_offset) or (
            len(results["power"]["impute4"]["snp_lmm"]) <= seed_offset):

            time_string = datetime.now().strftime("day-%Y-%m-%d-time-%H:%M:%S.%f") # ensure unique temp directories!
            impute_tmp_dir = os.path.join(args.base_tmp_dir, "impute4_" + time_string + "/")
            # In diploid case, assumes perfect phasing is available
            run_impute4(args.impute4_path, impute_tmp_dir,
                        chr1_with_sequence, chr1_with_sequence_snp_indices, args.num_sequence_samples,
                        args.num_snp_samples, diploid)
            logging.info("Converting from .gen.gz to .bgen")
            subprocess.run([
                args.plink2_path,
                "--memory",
                "2048",
                "--oxford-single-chr",
                "1",
                "--sample",
                impute_tmp_dir + "out.sample",
                "--gen",
                impute_tmp_dir + "out.gen.gz",
                "ref-last", # new for PLINK2
                "--export",
                "bgen-1.2",
                "bits=8",
                "--out",
                temp_base_path + ".test"
            ])
            logging.info("Done converting")
            shutil.rmtree(impute_tmp_dir)

            if len(results["bitsets"]["impute4"]) <= seed_offset:
                results["bitsets"]["impute4"].append(chr1_with_sequence.num_mutations)

            bolt_results = None
            if len(results["power"]["impute4"]["direct"]) <= seed_offset:
                if bolt_results is None:
                    bolt_results = bolt_bonferroni(temp_base_path, temp_base_path + ".test")
                results["power"]["impute4"]["direct"].append(bolt_results["linreg"])

            if len(results["power"]["impute4"]["snp_lmm"]) <= seed_offset:
                if bolt_results is None:
                    bolt_results = bolt_bonferroni(temp_base_path, temp_base_path + ".test")
                results["power"]["impute4"]["snp_lmm"].append(bolt_results["lmm_inf"])

    if args.pkl_path is not None:
        with open(args.pkl_path, 'wb') as outfile:
            pickle.dump(results, outfile)

    if args.num_sequence_samples > 0 and not phasing_error:
        # can skip the true and inferred ARG in this case?
        for temp_file in glob.glob(temp_base_path + '*'):
            os.remove(temp_file)
        continue

    # SNARG
    create_list_if_not_exists(results, ["bitsets", "snarg"])
    create_list_if_not_exists(results, ["power", "snarg", "direct"])
    create_list_if_not_exists(results, ["power", "snarg", "snp_lmm"])
    if (len(results["bitsets"]["snarg"]) <= seed_offset) or (
        len(results["power"]["snarg"]["direct"]) <= seed_offset) or (
        len(results["power"]["snarg"]["snp_lmm"]) <= seed_offset):

        # In diploid case, we have the opportunity to phase
        if diploid and phasing_error:
            time_string = datetime.now().strftime("day-%Y-%m-%d-time-%H:%M:%S.%f") # ensure unique temp directories!
            beagle_tmp_dir = os.path.join(args.base_tmp_dir, "beagle5_" + time_string + "/")
            new_ts = phase_data_return_new_ts(
                args.beagle5_path, beagle_tmp_dir, chr1_with_sequence, chr1_with_sequence_snp_indices,
                args.num_sequence_samples // 2, args.num_snp_samples // 2)
            # previous command removes beagle_tmp_dir already
            num_sequence_samples_save = args.num_sequence_samples
            # Hack
            args.num_sequence_samples = 0
            arg, max_memory = build_arg_simulation(
                args, new_ts, args.base_tmp_dir, snp_indices=None, mode="array")

            if args.num_sequence_samples > 0 and args.num_snp_samples > 0:
                logging.info("Subsampling to only the SNP samples")
                arg_ts = arg_needle_lib.arg_to_tskit(arg)
                snp_sample_ids = [i + args.num_sequence_samples for i in range(args.num_snp_samples)]
                arg_ts = arg_ts.simplify(samples=snp_sample_ids)
                arg = arg_needle_lib.tskit_to_arg(arg_ts)
                logging.info("Done subsampling")
            else:
                logging.info("Not doing anything for subsampling")

            args.num_sequence_samples = num_sequence_samples_save
        else:
            arg, max_memory = build_arg_simulation(
                args, chr1_with_sequence, args.base_tmp_dir,
                snp_indices=chr1_with_sequence_snp_indices, mode="array")

            if args.num_sequence_samples > 0 and args.num_snp_samples > 0:
                logging.info("Subsampling to only the SNP samples")
                arg_ts = arg_needle_lib.arg_to_tskit(arg)
                snp_sample_ids = [i + args.num_sequence_samples for i in range(args.num_snp_samples)]
                arg_ts = arg_ts.simplify(samples=snp_sample_ids)
                arg = arg_needle_lib.tskit_to_arg(arg_ts)
                logging.info("Done subsampling")
            else:
                logging.info("Not doing anything for subsampling")

        arg.populate_children_and_roots()
        _, num_bitsets = arg_write_bgen(args.plink2_path, args.base_tmp_dir, arg, args.pkl_path, diploid)
        os.remove(temp_base_path + '.arg.bitinfo')

        if len(results["bitsets"]["snarg"]) <= seed_offset:
            results["bitsets"]["snarg"].append(num_bitsets)

        bolt_results = None
        if len(results["power"]["snarg"]["direct"]) <= seed_offset:
            if bolt_results is None:
                bolt_results = bolt_bonferroni(temp_base_path, temp_base_path + ".test")
            results["power"]["snarg"]["direct"].append(bolt_results["linreg"])

        if len(results["power"]["snarg"]["snp_lmm"]) <= seed_offset:
            if bolt_results is None:
                bolt_results = bolt_bonferroni(temp_base_path, temp_base_path + ".test")
            results["power"]["snarg"]["snp_lmm"].append(bolt_results["lmm_inf"])

        del arg

    if args.pkl_path is not None:
        with open(args.pkl_path, 'wb') as outfile:
            pickle.dump(results, outfile)

    # True ARG
    if not phasing_error:
        create_list_if_not_exists(results, ["bitsets", "arg"])
        create_list_if_not_exists(results, ["power", "arg", "direct"])
        create_list_if_not_exists(results, ["power", "arg", "snp_lmm"])
        if (len(results["bitsets"]["arg"]) <= seed_offset) or (
            len(results["power"]["arg"]["direct"]) <= seed_offset) or (
            len(results["power"]["arg"]["snp_lmm"]) <= seed_offset):

            arg = arg_needle_lib.tskit_to_arg(simulations[0])
            arg.populate_children_and_roots()
            _, num_bitsets = arg_write_bgen(args.plink2_path, args.base_tmp_dir, arg, args.pkl_path, diploid)
            os.remove(temp_base_path + '.arg.bitinfo')

            if len(results["bitsets"]["arg"]) <= seed_offset:
                results["bitsets"]["arg"].append(num_bitsets)

            bolt_results = None
            if len(results["power"]["arg"]["direct"]) <= seed_offset:
                if bolt_results is None:
                    bolt_results = bolt_bonferroni(temp_base_path, temp_base_path + ".test")
                results["power"]["arg"]["direct"].append(bolt_results["linreg"])

            if len(results["power"]["arg"]["snp_lmm"]) <= seed_offset:
                if bolt_results is None:
                    bolt_results = bolt_bonferroni(temp_base_path, temp_base_path + ".test")
                results["power"]["arg"]["snp_lmm"].append(bolt_results["lmm_inf"])

            del arg

    for temp_file in glob.glob(temp_base_path + '*'):
        os.remove(temp_file)
    if args.pkl_path is not None:
        with open(args.pkl_path, 'wb') as outfile:
            pickle.dump(results, outfile)

logging.info("True h2: " + str(args.h2))
logging.info("True alpha: " + str(args.alpha))
logging.info("Rare beta: " + str(args.rare_beta))
logging.info("Rare MAC: " + str(rare_allele_count))
logging.info(results)
