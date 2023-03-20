import msprime
import numpy as np
import pandas
import math

class Simulator:
    def __init__(self, mapfile, demofile, sample_size, mu, rho=1.2e-8, check=False):
        self.mu = mu
        self.sample_size = sample_size
        if mapfile is not None:
            print("Reading " + mapfile, "...")
            self.recomb_map = msprime.RecombinationMap.read_hapmap(mapfile)
        else:
            self.recomb_map = None
        self.demography = self.read_demo(demofile)
        self.rho = rho
        self.configure_demography()
        if check:
            self.check_demographics()

    def read_demo(self, demofile):
        df = pandas.read_csv(demofile, sep="\t", header=None)
        df.columns = ['generation', 'size']
        return df

    def configure_demography(self):
        self.demographic_events = []
        self.pc = [msprime.PopulationConfiguration(self.sample_size)]
        for index in self.demography.index:
            if index == self.demography.shape[0] - 1: break
            forward_time = self.demography['generation'][index + 1]
            forward_size = self.demography['size'][index + 1]
            now_time = self.demography['generation'][index]
            now_size = self.demography['size'][index]
            g = (math.log(now_size) - math.log(forward_size)) / (forward_time - now_time)
            self.demographic_events.append(
            msprime.PopulationParametersChange(now_time, now_size, growth_rate=0))

    def check_demographics(self):
        dp = msprime.DemographyDebugger(
            population_configurations=self.pc, 
            demographic_events=self.demographic_events)
        dp.print_history()

    def simulation(self, length, random_seed=10, output=None):
        if self.recomb_map is None:
            tree_seq = msprime.simulate(
                population_configurations = self.pc, 
                demographic_events = self.demographic_events,
                mutation_rate = self.mu, 
                length=length,
                recombination_rate=self.rho,
                random_seed=random_seed
            )
        else:
            print("Ignoring length and rho, using recombination map instead")
            tree_seq = msprime.simulate(
                population_configurations = self.pc, 
                demographic_events = self.demographic_events,
                mutation_rate = self.mu, 
                recombination_map=self.recomb_map,
                random_seed=random_seed
            )
        if output != None:
            with open(output + ".vcf", "w") as vcf_file:
                tree_seq.write_vcf(vcf_file, 2)
            tree_seq.dump(output + ".tree")
        return tree_seq
