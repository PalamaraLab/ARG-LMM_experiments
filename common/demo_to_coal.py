"""Translate a demography file to a .coal file for Relate.

For documentation, see https://myersgroup.github.io/relate/modules.html#.coal.
"""
IN_PATH = "CEU2.demo"
OUT_PATH = "ceu2.coal"

generations = []
coal_rates = []
with open(IN_PATH, "r") as infile:
    for line in infile:
        tokens = line.strip('\n').split()
        generations.append(float(tokens[0]))
        coal_rates.append(0.5 / float(tokens[1])) # Treating the demography as diploid

with open(OUT_PATH, "w") as outfile:
    outfile.write("group1\n")
    outfile.write(" ".join(['{:.6g}'.format(item) for item in generations]) + "\n")
    outfile.write(" ".join(['{:.6g}'.format(item) for item in [0, 0] + coal_rates]) + "\n")
