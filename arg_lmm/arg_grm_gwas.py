"""Simulate a polygenic phenotype from multiple chromosomes and perform GWAS.

Compare linear regression to LMMs with LOCO, using various types of GRMs.
- Sequence GRM
- SNP GRM
- Monte Carlo ARG-GRM (ground-truth ARG-GRM)
"""

import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CURRENT_DIR, '../')) # for utils, simulator


# Python imports
import argparse
from functools import partial
import glob
import logging
import msprime
import numpy as np
import os
import pandas as pd
import pickle
import psutil; process = psutil.Process(os.getpid())
from scipy.stats import chi2
import subprocess

# Our packages
from arg_needle_lib import gower_center, write_grm

# Files from this repository
from common.simulator import Simulator
from common.utils import ukb_sample
from lmm_utils import sim_pheno_new, write_plink_new, write_pheno
from lmm_utils import make_grm


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
parser.add_argument("--plink2_path", help="Path to PLINK2 binary", action="store",
    default=os.path.join(CURRENT_DIR, "../plink2"))
# Other arguments
parser.add_argument("-a", "--alpha", help="Value of Speed et al. alpha for frequency weight", action="store", default=-0.5, type=float)
parser.add_argument("--diploid", help="Whether to do diploid", action="store", default=0, type=int)
parser.add_argument("--h2", help="Value of h2 to be simulated using infinitesimal model", action="store", default=0.8, type=float)
parser.add_argument("--mkl_num_threads", help="Number of MKL threads (default of 0 means not set)", action="store", default=0, type=int)
parser.add_argument("--mut_factors", help="log10(factor)*2 for mutation as a list", action="store", default="", type=str)
parser.add_argument("--num_chromosomes", help="Number of chromosomes", action="store", default=22, type=int)
parser.add_argument("--num_samples", help="Number of samples", action="store", default=1000, type=int)
parser.add_argument("--num_seeds", help="Number of seeds", action="store", default=1, type=int)
parser.add_argument("--sim_length", help="Length per chromosome", action="store", default=1e6, type=float)
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

if "sites" not in results:
    results["sites"] = []
if "logp" not in results:
    results["logp"] = {}
    results["logp"]["linreg"] = []
    results["improvement"] = {}
if "chi2" not in results:
    results["chi2"] = {}
    results["chi2"]["linreg"] = []

for results_key in ["seq", "snp", "mut"] + ["mut" + str(mut_factor) for mut_factor in mut_factors]:
    if results_key not in results["logp"]:
        results["logp"][results_key] = {}
        results["improvement"][results_key] = {}
    for alpha in test_alphas:
        if alpha not in results["logp"][results_key]:
            results["logp"][results_key][alpha] = []
            results["improvement"][results_key][alpha] = []
    if results_key not in results["chi2"]:
        results["chi2"][results_key] = {}
    for alpha in test_alphas:
        if alpha not in results["chi2"][results_key]:
            results["chi2"][results_key][alpha] = []

logging.info("Starting data (possibly read from pickle file)")
logging.info(results)

def mean_log10(df, key):
    return np.mean(-np.log10(df[[key]].to_numpy()))

def median_log10(df, key):
    return -np.log10(np.median(df[[key]]))

def read_and_record(field_getter, name, results, linreg_logp, temp_base_path):
    data = pd.concat([pd.read_csv(temp_base_path + '.' + str(k+1) + '.mlma', sep='\t') for k in range(args.num_chromosomes)])
    value = mean_log10(data, 'p')
    improvement = value / linreg_logp - 1
    logging.info(("Improvement for " + name, improvement))
    if len(field_getter(results["logp"])) <= seed_offset:
        field_getter(results["logp"]).append(value)
    if len(field_getter(results["improvement"])) <= seed_offset:
        field_getter(results["improvement"]).append(improvement)

    p_values = data[['p']].to_numpy()
    chi2s = chi2.isf(p_values, df=1)
    if len(field_getter(results["chi2"])) <= seed_offset:
        field_getter(results["chi2"]).append(np.mean(chi2s))

for seed_offset in range(args.num_seeds):
    simulations = []
    mut_simulations = []
    snp_indices_list = []
    seed = seed_offset + args.start_seed
    logging.info("Starting simulation with seed " + str(seed))
    for chromosome in range(args.num_chromosomes):
        msprime_seed = seed * args.num_chromosomes + chromosome
        simulation = simulator.simulation(args.sim_length, random_seed=msprime_seed)
        simulations.append(simulation)

    logging.info("Building phenotype in a memory-efficient manner")
    # this line seeds np.random
    pheno, num_sites = sim_pheno_new(simulations, args.h2, args.alpha, seed, diploid)
    if verbose:
        logging.info(str(num_sites) + " sites used to build phenotype")
    if len(results["sites"]) <= seed_offset:
        results["sites"].append(num_sites)
    logging.info("Done building phenotype")

    for chromosome in range(args.num_chromosomes):
        msprime_seed = seed * args.num_chromosomes + chromosome
        simulation = simulations[chromosome]
        snp_indices = ukb_sample(simulation, verbose)
        snp_indices_list.append(snp_indices)
        mut_simulation = msprime.mutate(simulation, rate=args.mu, random_seed=msprime_seed)
        mut_simulations.append(mut_simulation)

    # Write out PLINK files for SNP data
    temp_base_path = write_plink_new(args.plink2_path, args.base_tmp_dir, simulations, snp_indices_list, args.pkl_path, diploid)
    write_pheno(pheno, temp_base_path + '.phen')
    logging.info("Done with writing")

    # Use PLINK to do linear regression
    subprocess.run([
        args.plink2_path,
        "--memory",
        "2048",
        "--bfile",
        temp_base_path,
        "--pheno",
        temp_base_path + ".phen",
        "--glm",
        "allow-no-covars",
        "--out",
        temp_base_path + ".plink"
    ], check=True)
    linreg = pd.read_csv(temp_base_path + '.plink.PHENO1.glm.linear', sep='\t')
    linreg_logp = mean_log10(linreg, 'P')
    logging.info(("Linreg", linreg_logp))
    if "linreg" in results["logp"] and len(results["logp"]["linreg"]) <= seed_offset:
        results["logp"]["linreg"].append(linreg_logp)
    # new
    if "linreg" in results["chi2"] and len(results["chi2"]["linreg"]) <= seed_offset:
        p_values = linreg[['P']].to_numpy()
        chi2s = chi2.isf(p_values, df=1)
        results["chi2"]["linreg"].append(np.mean(chi2s))

    def foo(c, factor):
        new_rate = args.mu * 10**(factor*0.5)
        logging.info("New rate: " + str(new_rate))
        msprime_seed = seed * args.num_chromosomes + c
        bar = msprime.mutate(simulations[c], rate=new_rate, random_seed=msprime_seed)
        return (bar, None)

    memory_efficient = True
    # Run GCTA using sequence / SNP / mutated sequence kernels
    for s in [
        ("seq", lambda c: (simulations[c], None)),
        ("snp", lambda c: (simulations[c], snp_indices_list[c])),
        ("mut", lambda c: (mut_simulations[c], None))
    ] + [
        ("mut" + str(mut_factor), partial(foo, factor=mut_factor)) for mut_factor in mut_factors
    ]:
        results_key = s[0]
        if results_key == "mut" and len(mut_factors) == 1:
            # in this case, we're trying to hack the mut_factor so don't run twice
            continue
        for alpha in test_alphas:
            if len(results["improvement"][results_key][alpha]) <= seed_offset or len(results["chi2"][results_key][alpha]) <= seed_offset:
                grms = []
                num_sites_list = []
                grm_sum = np.zeros((args.num_samples, args.num_samples), dtype=np.float32)
                num_sites_sum = 0
                logging.info("Computing GRMs")
                for chromosome in range(args.num_chromosomes):
                    logging.info(chromosome)
                    simulation, snp_indices = s[1](chromosome)
                    grm = make_grm(simulation, snp_indices, alpha, diploid)
                    grm_sum += grm
                    if not memory_efficient:
                        grms.append(grm)
                    if snp_indices is None:
                        num_sites = simulation.num_mutations
                    else:
                        num_sites = len(snp_indices)
                    num_sites_list.append(num_sites)
                    num_sites_sum += num_sites

                del grm

                if memory_efficient:
                    logging.info("Computing LOCO GRMs through second pass")
                else:
                    logging.info("Gower centering and writing LOCO GRMs")
                for chromosome in range(args.num_chromosomes):
                    if memory_efficient:
                        logging.info(chromosome)
                        simulation, snp_indices = s[1](chromosome)
                        # grm = make_grm(simulation, snp_indices, alpha, diploid)
                        # grm_loco = gower_center(grm_sum - grm)
                        # this should be equivalent but take less memory
                        # equivalence due to Gower centering
                        grm_loco = make_grm(simulation, snp_indices, alpha, diploid)
                        grm_loco -= grm_sum
                        gower_center(grm_loco)
                    else:
                        grm_loco = gower_center(grm_sum - grms[chromosome])
                    num_sites_loco = num_sites_sum - num_sites_list[chromosome]
                    path_prefix = temp_base_path + '.' + str(chromosome + 1)
                    write_grm(grm_loco, num_sites_loco, path_prefix, write_binary=True)

                    del grm_loco

                    logging.info("Running GCTA")

                    chrom_string = str(chromosome + 1)
                    subprocess.run([
                        args.gcta_path,
                        "--mlma",
                        "--grm",
                        temp_base_path + "." + chrom_string,
                        "--chr",
                        chrom_string,
                        "--bfile",
                        temp_base_path,
                        "--pheno",
                        temp_base_path + ".phen",
                        "--out",
                        temp_base_path + "." + chrom_string,
                        # --thread-num?
                    ], check=True)

                    # Remove the GRMs to save on disk space
                    for temp_file in glob.glob(path_prefix + '.grm.*'):
                        os.remove(temp_file)

                read_and_record(lambda x: x[results_key][alpha], results_key + " " + str(alpha), results, linreg_logp, temp_base_path)

                if args.pkl_path is not None:
                    with open(args.pkl_path, 'wb') as outfile:
                        pickle.dump(results, outfile)

    for temp_file in glob.glob(temp_base_path + '*'):
        os.remove(temp_file)

logging.info("True h2: " + str(args.h2))
logging.info("True alpha: " + str(args.alpha))
logging.info(results)
