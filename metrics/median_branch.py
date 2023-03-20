"""Get the median branch length for mixed KC metric."""

import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CURRENT_DIR, '../')) # for utils, simulator


# Python imports
import argparse
import logging
import numpy as np
import pickle

# Files from this repository
from common.simulator import Simulator

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

parser = argparse.ArgumentParser(description='Simulate coalescent trees and get median branch length.')
# Other arguments
parser.add_argument("--sim_length", help="Simulation length", action="store", default=1e3, type=float)
parser.add_argument("--num_samples", help="Number of samples", action="store", default=4000, type=int)
parser.add_argument("--num_seeds", help="Number of seeds", action="store", default=100, type=int)
parser.add_argument("--start_seed", help="Start seed", action="store", default=1, type=int)
args = parser.parse_args()

args.mapfile = None
args.demo = "CEU2"
args.demofile = os.path.join(CURRENT_DIR, "../common", args.demo + ".demo")
args.rho = 0
args.mu = 0

logging.info("Command-line args:")
args_to_print = vars(args)
for k in sorted(args_to_print):
    logging.info(k + ": " + str(args_to_print[k]))

# set up simulator with sequence + SNP samples
simulator = Simulator(args.mapfile, args.demofile,
    sample_size=args.num_samples,
    mu=args.mu, rho=args.rho)

heights = []
for seed_offset in range(args.num_seeds):
    seed = seed_offset + args.start_seed
    logging.info("Starting simulation " + str(seed))
    simulation = simulator.simulation(args.sim_length, random_seed=seed)
    print(simulation.num_nodes, simulation.num_edges)
    for edge in simulation.edges():
        parent = simulation.node(edge.parent)
        child = simulation.node(edge.child)
        height = parent.time - child.time
        heights.append(height)

heights = np.array(heights)

heights2 = heights.reshape(100, len(heights) // 100)
medians_per_run = np.median(heights2, axis=1)
print(medians_per_run.tolist())
print(np.mean(medians_per_run), np.median(medians_per_run))

heights = np.sort(heights)
middle = len(heights) // 2
print(len(heights), middle)
print(heights[middle-5:middle+5])

"""
For N = 4000, get
49.17391741727159 49.17967021930483
799800 399900
[49.19227201 49.19297811 49.19297811 49.19300729 49.19300729 49.19335735
 49.19345796 49.19370344 49.19377792 49.19385551]

For N = 2000, get
76.65109502306001 76.73947457732508
[76.64359059 76.64511887 76.64535322 76.64543552 76.64543552 76.64548737
 76.64548737 76.64689401 76.64751853 76.64751853]

For N = 8000, get
30.12041101490281 30.07579898821171
1599800 799900
[30.12093316 30.1210764  30.1210764  30.12109525 30.12118911 30.12121409
 30.12150434 30.12152445 30.12174242 30.12186838]

For N = 4000, can use lambda = 0.02
(1 - lambda) / lambda = 49.19
1 / lambda = 50.19
lambda = 0.019924 ~= 0.02
"""
