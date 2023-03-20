# This file is part of the ARG-Needle genealogical inference and
# analysis software suite.
# Copyright (C) 2023 ARG-Needle Developers.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import argparse
import gzip
import logging
import numpy as np
import os
import pandas as pd

import msprime
assert msprime.__version__ >= '1.0.0'

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


# REQUIRED FILES
import arg_needle
CEU_DEMO_PATH = os.path.join(
    os.path.dirname(arg_needle.__file__),
    "resources/CEU.demo")
UKB_CHR2_SPECTRUM = os.path.join(
    os.path.dirname(arg_needle.__file__),
    "resources/ukb_chr2_spectrum.tsv")

# OUTPUT PATHS
hap_path = 'example.hap.gz'
map_path = 'example.map'
sample_path = 'example.sample'

# SIMULATION PARAMETERS
parser = argparse.ArgumentParser(description='Prepare ARG inference example.')
parser.add_argument("--num_individuals", default=200, type=int,
    help="number of diploid individuals (note: need at least 150 diploids for ASMC, default=200)")
parser.add_argument("--length", default=2e6, type=float,
    help="sequence length (default=2e6 for 2 Mb)")
parser.add_argument("--mode", default="array",
    help="either array or sequence (default=array)")
parser.add_argument("--start_position", default=1e7, type=float,
    help="start position of variants (default=1e7 for 10 Mb)")
parser.add_argument("--rho", default=1.2e-8, type=float,
    help="recombination rate (default=1.2e-8)")
parser.add_argument("--mu", default=1.65e-8, type=float,
    help="mutation rate (default=1.65e-8)")
parser.add_argument("--seed", default=42, type=int,
    help="random seed for simulation (default=42)")
parser.add_argument("--chromosome", default=1, type=int,
    help="chromosome number used for writing (default=1)")


def ukb_sample(simulation, random_seed=1, verbose=True):
    """Use UKB spectrum to sample variants and obtain realistic SNP data.

    Arguments:
        simulation: tskit.TreeSequence with mutations
        random_seed: random seed to use
        verbose: boolean for whether to log updates to stdout

    Returns:
        numpy array containing SNP indices
    """
    # Seed the RNG
    rng = np.random.default_rng(seed=random_seed)

    # We use the UKB chr2 array data MAF spectrum, along with chr2 length
    spectrum_path = UKB_CHR2_SPECTRUM
    spectrum_segment_size = 243*(10**6)  # 243 Mb
    sim_length = simulation.sequence_length
    factor = sim_length / spectrum_segment_size
    bin_starts = np.arange(51) / 100

    # Read in the spectrum file
    scaled_bin_counts = []
    with open(spectrum_path, 'r') as infile:
        for line in infile:
            values = line.strip('\n').split()
            scaled_bin_counts.append(int(round(int(values[2]) * factor)))
    scaled_bin_counts = np.array(scaled_bin_counts)
    if scaled_bin_counts.shape != bin_starts.shape:
        raise ValueError("Uh-oh, unexpected spectrum file")

    binned_ids = [[] for i in range(len(bin_starts))]

    num_variants = sum(1 for _ in simulation.variants())
    # Note: simulation.num_mutations may be different because of finite-sites simulation
    af = np.zeros(num_variants)
    for i, variant in enumerate(simulation.variants()):
        geno = variant.genotypes
        af[i] = float(np.sum(geno)) / float(len(geno))

    if len(geno) <= 100:
        raise ValueError("Number of samples for computing allele frequency is too small")

    maf = np.minimum(af, 1 - af)
    binned_maf = np.floor(maf * 100).astype(int)
    num_skipped = 0
    for i in range(len(binned_maf)):
        if maf[i] == 0:
            num_skipped += 1
        else:
            binned_ids[binned_maf[i]].append(i)
    if verbose and num_skipped > 0:
        logging.info("Skipping {} monomorphic SNPs".format(num_skipped))

    chosen_ones = []
    for i in range(len(bin_starts)):
        if len(binned_ids[i]) < scaled_bin_counts[i]:
            if verbose:
                warning_string = "Can only choose {} instead of {} IDs from [{:.2f}, {:.2f})".format(
                    len(binned_ids[i]), scaled_bin_counts[i], bin_starts[i], bin_starts[i] + 0.01)
                logging.info(warning_string)
            chosen_ones.extend(binned_ids[i])
        else:
            chosen_ones.extend(rng.choice(binned_ids[i], scaled_bin_counts[i], replace=False))
    chosen_ones = np.sort(chosen_ones)
    if verbose:
        logging.info(f"Chose {len(chosen_ones)} out of {maf.shape[0]} SNPs")
    return chosen_ones


def main():
    args = parser.parse_args()
    logging.info("Command-line args:")
    args_to_print = vars(args)
    for k in sorted(args_to_print):
        logging.info("  {}: {}".format(k, args_to_print[k]))

    if args.mode not in ["array", "sequence"]:
        raise ValueError("Mode must be either array or sequence")

    if args.num_individuals < 150:
        raise ValueError("ASMC requires at least 150 diploid individuals")

    # Simulate ARG
    df = pd.read_table(CEU_DEMO_PATH, header=None)
    df.columns  = ['GEN', 'NE']
    demography = msprime.Demography()
    # (these initial sizes get overwritten anyways...)
    demography.add_population(name="A", initial_size=1e4)
    for t, ne in zip(df.GEN.values, df.NE.values):
        # Divide by 2 because msprime wants diploids but demography is haploid
        demography.add_population_parameters_change(t, initial_size=ne / 2, population="A")

    ts = msprime.sim_ancestry(
        samples={"A": args.num_individuals},
        demography=demography,
        sequence_length=args.length,
        recombination_rate=args.rho,
        random_seed=args.seed)
    # Finite-sites model
    mts = msprime.sim_mutations(
        ts, rate=args.mu, random_seed=args.seed, model=msprime.BinaryMutationModel())
    # Uncomment here to save simulated arg or ts to disk
    # arg_path = 'example_true.argn'
    # ts_path = 'example_true.trees'
    # arg = arg_needle_lib.tskit_to_arg(ts)
    # arg_needle_lib.serialize_arg(arg, arg_path)
    # mts.dump(ts_path)

    # Write hap/map/sample files
    with open(sample_path, 'w+') as samplefile:
        samplefile.write('ID_1 ID_2 missing\n0 0 0\n')
        for i in range(1, args.num_individuals + 1):
            samplefile.write(f"s{i} s{i} 0\n")

    if args.mode == "array":
        # Sample SNPs according to UKB chromosome 2 MAF spectrum and density
        snp_indices = ukb_sample(mts, random_seed=args.seed, verbose=True)
    index = 0
    with gzip.open(hap_path, 'wb+') as hapfile, open(map_path, 'w+') as mapfile:
        for i, v in enumerate(mts.variants()):
            if args.mode == "sequence" or (
                index != len(snp_indices) and i == snp_indices[index]
            ):
                snp_id = f"snp{i}"
                # we add the start position argument here
                bp = v.position + args.start_position
                cm = bp * args.rho * 100
                chrom = args.chromosome
                genotypes = v.genotypes.tolist()
                geno_str = ' '.join([str(g) for g in genotypes])
                mapfile.write(f"{chrom}\t{snp_id}\t{cm}\t{bp:.0f}\n")
                hapfile.write(f"{chrom} {snp_id} {bp:.0f} G A {geno_str}\n".encode())
                index += 1

if __name__ == "__main__":
    main()
