import paramless_cython as pm

from scipy.integrate import simps
import numpy as np
import csv
import json
import sys

class MetabolicFitness(pm.ModelFitnessFunction):
    def __init__(self, c, domain):
        self.c = c
        self.domain = domain
        
    def get(self, definitions):
        result = dict()
        for key, value in definitions.items():
            result[key] = self.total_intake(value, self.domain) - self.c * self.energy(value, self.domain)
        return result
        
    def energy(self, surface, domain):
        return simps(surface, domain)
        
    def resource_density(self, a):
        return 4.0 * (1.0 - a)

    def total_intake(self, surface, domain):
        #create a list of pairs x(a) = a for a in domain
        tuples_surface_domain = zip(surface, domain)
        #r(a)*(x(a)/x(a)+a) for all a in domain
        integrand = [self.resource_density(surface_domain[1])*(surface_domain[0]/(surface_domain[0]+surface_domain[1])) for surface_domain in tuples_surface_domain]
        #integrate over domain
        return simps(integrand, domain)

def getDominant(pop):
    result = 0
    max = 0
    for key, value in pop.items():
        if value > max:
            max = value
            result = key
    return result
    
def main(argv):
    # Load json file for parameters
    with open(argv[0]) as config_file:
        config = json.load(config_file)
    
    if "seed" in config:    
        np.random.seed(config["seed"])
        pm.setup(config["seed"])
    
    if ("domain_lower" in config) and ("domain_upper" in config):
        if "domain_step" in config:
            domain = np.arange(config["domain_lower"], config["domain_upper"], config["domain_step"])
        else:
            domain = np.arange(config["domain_lower"], config["domain_upper"], 0.01)
    else:
        domain = np.arange(0.001, 1.0, 0.01)
        
    initial_surface = np.zeros_like(domain)
    population = {1:20}
    definitions = {1:initial_surface}
    if "lower_bound" in config:
        if "upper_bound" in config:
            mutator = pm.GaussianMutator(config["mutation_epsilon"], domain, config["width"], config["lower_bound"], config["upper_bound"])
        else:
            mutator = pm.GaussianMutator(config["mutation_epsilon"], domain, config["width"], config["lower_bound"])
    else:
        mutator = pm.GaussianMutator(config["mutation_epsilon"], domain, config["width"])
    fitness_function = MetabolicFitness(config["c"], domain)
    
    if config["evolver"] == "moran":
        evolver_class = pm.MoranEvolver
    else:
        evolver_class = pm.WrightFisherEvolver
    
    if "ctol" in config:
        if "atol" in config:
            if "mut_chance" in config:
                evolver = evolver_class(fitness_function, mutator, config["ctol"], config["atol"], config["mut_chance"])
            else:
                evolver = evolver_class(fitness_function, mutator, config["ctol"], config["atol"])
        else:
            evolver = evolver_class(fitness_function, mutator, config["ctol"])
    else:
        evolver = evolver_class(fitness_function, mutator)
    
    with open(config["output_file"], "wb") as output_file:
        writer = csv.writer(output_file)
        last_entry_time = 0
        seq = 0
        dominant = getDominant(population)
        dominant_def = definitions[dominant]
        for i in xrange(1, config["iterations"]):
            [population, definitions] = evolver.do_step(population, definitions)
            temp = getDominant(population)
            if (dominant != temp):
                dominant = temp
                dominant_def = definitions[dominant]
                last_entry_time = i
                seq += 1
            if i % 10000 == 0:
                writer.writerow([i - last_entry_time, dominant_def])
                    
    print definitions[getDominant(population)]


if __name__ == "__main__":
    main(sys.argv[1:])    
