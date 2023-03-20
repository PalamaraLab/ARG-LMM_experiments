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


"""Advanced ARG inference in multiple steps

In the case of large, real data analysis, we may want to infer ARGs in several steps
for time and memory constraints:
    1. First thread a few samples to the ARG, serializing progress.
    2. Second, thread the remaining samples to the ARG and serialize the result.
    3. Perform ARG normalization as well as removal of unneeded "padding" on the
        boundaries. It can be useful to perform this separately as it tends to increase
        the memory of the process.

This workflow can be run over three steps by calling this script with `--step=1`,
then `--step=2`, then `--step=3`.

To run ASMC-clust, set command line flag `--asmc-clust 1` (cannot be used for step 2).
"""
import argparse
import gzip
import logging
import numpy as np
import os
import subprocess

from arg_needle import build_arg, extend_arg, add_default_arg_building_arguments
from arg_needle import normalize_arg, trim_arg
import arg_needle_lib

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

import arg_needle
CEU_DEMO_PATH = os.path.join(
    os.path.dirname(arg_needle.__file__),
    "resources/CEU.demo")


parser = argparse.ArgumentParser(description='Infer ARGs and evaluate.')
parser.add_argument("--hap_gz", default='example.hap.gz',
    help="Path to .hap[s][.gz] file (default=example.hap.gz)")
parser.add_argument("--map", default='example.map',
    help="Path to genetic map (default=example.map)")
parser.add_argument("--out", default='example_advanced',
    help="Path to output without .argn extension (default=example_advanced)")
parser.add_argument("--mode", default='array',
    help='Data mode ("array" or "sequence", default="array")')
# Step number: 1, 2, or 3
parser.add_argument("--step", default=1, type=int,
    help="Which of steps 1, 2, 3 to run (default=1)")
# ARG normalization option
parser.add_argument("--normalize", default=1, type=int,
    help="Whether to apply ARG normalization (default=1, nonzero means true)")
parser.add_argument("--normalize_demography", default=CEU_DEMO_PATH,
    help=f"Demography with haploid population sizes for normalization (default={CEU_DEMO_PATH})")
# Other options
parser.add_argument("--chromosome", default=1, type=int,
    help="Chromosome number to store in the ARG (default=1)")
parser.add_argument("--trim_num_snps", default="50",
    help="Number of padding SNPs to trim off each side (default=50)." \
    " Can either be a single integer or a string of two integers separated by a single comma without space, e.g. 0,50." \
    " In the latter case the two values represent the trimming from the beginning and the end.")
parser.add_argument("--verbose", default=0, type=int,
    help="Whether to log verbose output (default=0, nonzero means true)")

# Add other ARG-building arguments defined in src/inference.py
add_default_arg_building_arguments(parser)


def log_arg_info(arg):
    logging.info("ARG samples={}, chromosome={}, start={}, end={}".format(
        arg.num_samples(), arg.chromosome, arg.offset + arg.start, arg.offset + arg.end))


def main():
    args = parser.parse_args()

    # Display the arguments used
    logging.info("Command-line arguments:")
    args_to_print = vars(args)
    for k in sorted(args_to_print):
        logging.info("  {}: {}".format(k, args_to_print[k]))

    verbose = (args.verbose != 0)
    if args.mode not in ["array", "sequence"]:
        raise ValueError("Mode must be either array or sequence")

    # Find the hapsfile basename
    if args.hap_gz.endswith('.hap'):
        haps_root = args.hap_gz[:-4]
    elif args.hap_gz.endswith('.haps'):
        haps_root = args.hap_gz[:-5]
    elif args.hap_gz.endswith('.hap.gz'):
        haps_root = args.hap_gz[:-7]
    elif args.hap_gz.endswith('.haps.gz'):
        haps_root = args.hap_gz[:-8]
    else:
        raise ValueError(f"Expected '.hap[.gz]' or '.haps[.gz]', given {args.hap_gz}")

    samplefile = f"{haps_root}.sample"
    if not os.path.exists(args.hap_gz):
        raise ValueError(f"File {args.hap_gz} is missing")
    if not os.path.exists(samplefile):
        raise ValueError(f"File {samplefile} is missing")
    if not os.path.exists(args.map):
        raise ValueError(f"File {args.map} is missing")

    out_directory = os.path.dirname(args.out)
    if out_directory != "" and not os.path.exists(out_directory):
        logging.info(f"Creating directory {out_directory} for output")
        os.makedirs(out_directory)

    # Find the number of samples
    num_samples = 2 * (int(subprocess.check_output(['wc', '-l', samplefile]).split()[0]) - 2)

    assert args.step in [1, 2, 3]
    if args.step == 1:
        # for step 1, we let user set the number of samples
        if args.mode == "array":
            if args.num_snp_samples == 0 or args.num_snp_samples > num_samples:
                raise ValueError(f"Set num_snp_samples between 1 and {num_samples}")
        else:
            if args.num_sequence_samples == 0 or args.num_sequence_samples >= num_samples:
                raise ValueError(f"Set num_sequence_samples between 1 and {num_samples - 1}")
    else:
        # after step 2, we want to have all the samples
        if args.mode == "array":
            if args.num_snp_samples != 0:
                logging.info(f"Note: num_snp_samples will be overwritten with {num_samples}")
            args.num_snp_samples = num_samples
        else:
            if args.num_sequence_samples != 0:
                logging.info(f"Note: num_sequence_samples will be overwritten with {num_samples}")
            args.num_sequence_samples = num_samples

    if args.step == 1:
        # Run ARG-building routine on the first set of samples
        arg_step1, max_memory = build_arg(args, haps_root, args.map, args.mode, verbose=verbose)
        arg_step1.set_chromosome(args.chromosome)
        log_arg_info(arg_step1)
        arg_needle_lib.serialize_arg(arg_step1, f'{args.out}.step1.argn')
    elif args.step == 2:
        # Run ARG-building routine on the remainder of samples
        # Important to reserve space for additional samples
        arg_step1 = arg_needle_lib.deserialize_arg(
            f'{args.out}.step1.argn', reserved_samples=num_samples)
        arg_step2, max_memory = extend_arg(
            args, arg_step1, haps_root, args.map, args.mode, verbose=verbose)

        arg_step2.set_chromosome(args.chromosome)
        log_arg_info(arg_step1)
        arg_needle_lib.serialize_arg(arg_step2, f'{args.out}.step2.argn')
    else:
        # Trim unused regions used for SNP padding, apply ARG normalization, and write out
        arg_step2 = arg_needle_lib.deserialize_arg(f'{args.out}.step2.argn')

        bps = []
        with open(args.map) as infile:
            for line in infile:
                bps.append(int(line.strip().split(" ")[-1].split("\t")[-1]))

        if args.trim_num_snps.isdigit():
            trim_num_snps_start = int(args.trim_num_snps)
            trim_num_snps_end = int(args.trim_num_snps)
        else:
            split_list = args.trim_num_snps.split(",")
            if len(split_list) != 2:
                raise ValueError("trim_num_snps should be an integer or contain two integers separated by a comma")
            if not split_list[0].isdigit() or not split_list[1].isdigit():
                raise ValueError("trim_num_snps should be an integer or contain two integers separated by a comma")
            trim_num_snps_start = int(split_list[0])
            trim_num_snps_end = int(split_list[1])

        start_snp_count = trim_num_snps_start
        end_snp_count = len(bps) - 1 - trim_num_snps_end

        logging.info("Start SNP ID / end SNP ID: {} {}".format(start_snp_count, end_snp_count))
        if not start_snp_count < end_snp_count:
            raise ValueError("Asking to trim too much, please decrease the value(s) in trim_num_snps")
        logging.info("Start SNP position / end SNP position: {} {}".format(bps[start_snp_count], bps[end_snp_count]))

        start_trim = -1
        end_trim = -1
        if start_snp_count > 0:
            start_trim = (bps[start_snp_count] + bps[start_snp_count - 1])*0.5
        if end_snp_count < len(bps) - 1:
            end_trim = (bps[end_snp_count] + bps[end_snp_count + 1])*0.5
        if start_trim != -1:
            start_trim_diff = start_trim - int(start_trim)
            start_trim = int(start_trim)
            if end_trim != -1:
                end_trim = end_trim - start_trim_diff

        logging.info("Trim positions after finding midpoints (-1 means no trim): {} {}".format(start_trim, end_trim))
        logging.info("About to trim")
        arg = trim_arg(arg_step2, start_trim, end_trim, verbose=verbose)

        if args.normalize != 0:
            # Normalize using given demography
            arg = normalize_arg(arg, args.normalize_demography, assume_haploid=True, verbose=verbose)

        arg.set_chromosome(args.chromosome)
        log_arg_info(arg)
        arg_needle_lib.serialize_arg(arg, f'{args.out}.argn')

        if False:
            # Remove temporary files
            # Copy and paste into your own workflow to run
            os.remove(f'{args.out}.step1.argn')
            os.remove(f'{args.out}.step2.argn')
            logging.info('Deleted intermediate .argn files')

if __name__ == "__main__":
    main()
