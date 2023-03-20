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


"""ARG inference using ARG-Needle and ASMC-clust

To run ASMC-clust, set command line flag `--asmc-clust 1`
"""
import argparse
import logging
import numpy as np
import os
import subprocess

from arg_needle import build_arg, add_default_arg_building_arguments, normalize_arg
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
parser.add_argument("--out", default='example',
    help="Path to output without .argn extension (default=example)")
parser.add_argument("--mode", default='array',
    help='Data mode ("array" or "sequence", default="array")')
# ARG normalization option
parser.add_argument("--normalize", default=1, type=int,
    help="Whether to apply ARG normalization (default=1, nonzero means true)")
parser.add_argument("--normalize_demography", default=CEU_DEMO_PATH,
    help=f"Demography with haploid population sizes for normalization (default={CEU_DEMO_PATH})")
# Other options
parser.add_argument("--chromosome", default=1, type=int,
    help="Chromosome number to store in the ARG (default=1)")
parser.add_argument("--verbose", default=0, type=int,
    help="Whether to log verbose output (default=0, nonzero means true)")


# Add other ARG-building arguments defined in src/inference.py
add_default_arg_building_arguments(parser)

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

    if args.mode == "array":
        args.num_snp_samples = num_samples
    else:
        args.num_sequence_samples = num_samples

    # Run ARG-building routine
    arg, max_memory = build_arg(args, haps_root, args.map, args.mode, verbose=verbose)

    if args.normalize != 0:
        # Normalize using given demography
        arg = normalize_arg(arg, args.normalize_demography, assume_haploid=True, verbose=verbose)

    arg.set_chromosome(args.chromosome)
    logging.info("ARG samples={}, chromosome={}, start={}, end={}".format(
        arg.num_samples(), arg.chromosome, arg.offset + arg.start, arg.offset + arg.end))
    arg_needle_lib.serialize_arg(arg, f'{args.out}.argn')

if __name__ == "__main__":
    main()
