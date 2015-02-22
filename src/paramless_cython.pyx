# cython: boundscheck=False
# cython: wraparound=False
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport fabs
from numpy.math cimport INFINITY
from libc.stdlib cimport rand, srand, RAND_MAX, malloc, free
from libc.math cimport exp

# The maximum number of iterations to be used in attempting to get a new mutant
cdef int MAX_ITERATIONS = 1000000


cdef class FitnessFunction(object):
    """ An abstract class to be subclassed for use by an Evolver """

    def get(self, vector):
        """ Abstract function to be implemented by a subclass.
        Evaluates the fitness function of a vector according to this object

        Args:
            vector: The resident to be evaluated

        Returns:
            The fitness of the vector
        """
        raise NotImplementedError

cdef class ModelFitnessFunction(object):
    """ An abstract class to be subclassed for use by a ModelEvolver """

    def get(self, definitions):
        """ Abstract function to be implemented by a subclass.
        Evaluates the fitness function of a dictionary of vectors according to this object

        Args:
            definitions: The vectors to be evaluated

        Returns:
            A dictionary containing the fitnesses of the vectors
        """
        raise NotImplementedError

cdef class Mutator(object):
    """ An abstract class to be subclassed for use by an Evolver """

    def doMutation(self, vector):
        """ Abstract function to be implemented by a subclass.
        Mutates the given vector according to this object

        Args:
            vector: The function array to be mutated

        Returns:
            The mutated function        
        """
        raise NotImplementedError

cdef class Evolver(object):
    """ An abstract class to be subclassed for use by the main evolve function """

    def do_step(self, resident_surface):
        """ Abstract function to be implemented by a subclass.
        Performs the main body of the evolution

        Args:
            resident_surface: The resident surface array to be used for this step

        Returns:
            The resident at the end of the step which may be the original one and whether
                an invasion occurred.
        """
        raise NotImplementedError


cdef class ModelEvolver(object):
    """ An abstract class to be subclassed for use by the model evolve function """

    def do_step(self, definitions, population):
        """ Abstract function to be implemented by a subclass.
        Performs the main body of the evolution

        Args:
            definitions: The definition dictionary of the residents to be used for this step
            population: The dictionary containing the number of each type of resident present in the model

        Returns:
            The resident at the end of the step which may be the original one and whether
                an invasion occurred.
        """
        raise NotImplementedError


cdef class DistanceFitnessFunction(FitnessFunction):
    """ A FitnessFunction implementation for use for fitting to a given target function """
    # The target function
    cdef double[:] target

    def __init__(self, np.ndarray[double, ndim=1] target):
        self.target = target

    @cython.cdivision(True)
    cpdef get(self, double[:] vector):
        """ Evaluates the fitness function of a vector according to:
            fitness = 1/distance

        Args:
            vector: The resident to be evaluated

        Returns:
            The fitness of vector
        """
        return 1.0 / one_norm_distance(vector, self.target)

cdef class ModelDistanceFitnessFunction(ModelFitnessFunction):
    """ A ModelFitnessFunction implementation for use for fitting to a given target function """
    # The target function
    cdef double[:] target

    def __init__(self, np.ndarray[double, ndim=1] target):
        self.target = target

    @cython.cdivision(True)
    cpdef get(self, definitions):
        """ Evaluates the fitness function of a dictionary of vectors according to:
            fitness = 1/distance

        Args:
            definitions: The vectors to be evaluated

        Returns:
            A dictionary of the fitnesses of the vectors
        """
        result = dict()
        cdef int key
        cdef double[:] value
        for key, value in definitions.items():
            result[key] = 1.0 / one_norm_distance(value, self.target)
        return result

cdef one_norm_distance(double[:] u, double[:] v):
    """ Given two arrays returns the sum of position by position distances.
    Used as a distance function.

    Args:
        u,v: The vectors to be used to determine the distance between them

    Returns:
        The distance between the given vectors
    """
    cdef double sum = 0
    cdef size = u.shape[0]
    cdef unsigned int i
    for i in range(size):
        sum += fabs(u[i] - v[i])
    return sum


cdef class PointMutator(Mutator):
    """ A Mutator implementation for use in mutating single points in a vector
    Will result in a vector having one point changed.
    """
    # The epsilon to be used in mutations and absolute bounds for resulting
    # vectors
    cdef double mutation_epsilon, lower_bound, upper_bound

    def __init__(self, double mutation_epsilon, double lower_bound=INFINITY, double upper_bound=-INFINITY):
        self.mutation_epsilon = mutation_epsilon
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    cdef double[:] cyDoMutation(self, double[:] vector):
        """ The Cython code used to mutate the given vector.

        Args:
            vector: the vector to be mutated

        Returns:
            A point mutated vector
        """
        cdef int is_inside = 0
        cdef int attempt = 0
        cdef double[:] mutant
        while (is_inside != 1):
            mutant = self.attempt_point_mutation(vector)
            is_inside = cy_within_bounds(
                mutant, self.lower_bound, self.upper_bound)
            attempt += 1
            if attempt > MAX_ITERATIONS:
                raise RuntimeError(
                    "Attempted too many mutations without producing anything within bounds")
        return mutant

    def doMutation(self, np.ndarray[double, ndim=1] vector):
        """ A wrapper function to allow Python code to access the optimised Cython code. """
        return np.asarray(self.cyDoMutation(vector))

    cdef double[:] attempt_point_mutation(self, double[:] vector):
        """ The body of the mutation code. Mutation is completed through randomly choosing a point
            in the vector and increasing or decreasing its value by the mutation epsilon.

        Args:
            vector: The vector to be mutated

        Returns:
            The mutated vector        
        """
        cdef double[:] mutant = np.copy(vector)
        cdef int position = int(rand() / float(RAND_MAX) * vector.shape[0])
        if (rand() % 2):
            mutant[position] = mutant[position] + self.mutation_epsilon
        else:
            mutant[position] = mutant[position] - self.mutation_epsilon
        return mutant

cdef class PointDistributionMutator(Mutator):
    """ A Mutator implementation for use in mutating two points in a vector while keeping the total area
        under the curve constant
        Will result in a vector having two or no points changed.
    """
    # The epsilon to be used in the mutations
    cdef double mutation_epsilon

    def __init__(self, double mutation_epsilon):
        self.mutation_epsilon = mutation_epsilon

    cdef double[:] cyDoMutation(self, double[:] vector):
        """ The Cython code used to mutate the given vector.

        Args:
            vector: the vector to be mutated

        Returns:
            A point mutated vector
        """
        cdef double[:] mutant = np.copy(vector)
        cdef int pos_up = int(rand() / float(RAND_MAX) * vector.shape[0]), pos_down = int(rand() / float(RAND_MAX) * vector.shape[0])
        cdef adjusted_epsilon = self.mutation_epsilon
        if (mutant[pos_down] - adjusted_epsilon < 0):
            adjusted_epsilon = mutant[pos_down]
        mutant[pos_up] += adjusted_epsilon
        mutant[pos_down] -= adjusted_epsilon
        return mutant

    def doMutation(self, np.ndarray[double, ndim=1] vector):
        """ A wrapper function to allow Python code to access the optimised Cython code. """
        return np.asarray(self.cyDoMutation(vector))

cdef class GaussianMutator(Mutator):
    """ A Mutator implementation for use in mutating the entire vector according to a Gaussian distribution """
    # The epsilon to be used in mutations, the width of the distribution and
    # absolute bounds for resulting vectors
    cdef double mutation_epsilon, width, lower_bound, upper_bound
    # The domain of the vectors to be used
    cdef double[:] domain

    def __init__(self, double mutation_epsilon, np.ndarray[double, ndim=1] domain, double width, double lower_bound=INFINITY, double upper_bound=-INFINITY):
        self.mutation_epsilon = mutation_epsilon
        self.domain = domain
        self.width = width
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    cdef double[:] cyDoMutation(self, double[:] vector):
        """ The Cython code used to mutate the given vector.

        Args:
            vector: the vector to be mutated

        Returns:
            A point mutated vector
        """
        cdef int is_inside = 0
        cdef int attempt = 0
        cdef double[:] mutant
        while (is_inside != 1):
            mutant = self.attempt_gaussian_mutation(vector)
            is_inside = cy_within_bounds(
                mutant, self.lower_bound, self.upper_bound)
            attempt += 1
            if attempt > MAX_ITERATIONS:
                raise RuntimeError(
                    "Attempted too many mutations without producing anything within bounds")
        return mutant

    def doMutation(self, np.ndarray[double, ndim=1] vector):
        """ A wrapper function to allow Python code to access the optimised Cython code. """
        return np.asarray(self.cyDoMutation(vector))

    cdef double[:] attempt_gaussian_mutation(self, double[:] vector):
        """ The body of the mutation code. Mutation is completed through randomly choosing a point
            in the vector and a width and increasing or decreasing its value and the surrounding values according
            to a Gaussian distribution multiplied by the mutation epsilon.

        Args:
            vector: The vector to be mutated

        Returns:
            The mutated vector        
        """
        cdef int index = int(rand() / float(RAND_MAX) * vector.shape[0])
        cdef double value = self.domain[index]
        cdef double[:] mutant = np.copy(vector)
        cdef double[:] perturbation = self.helper(value)
        cdef int i, size = mutant.shape[0]
        # upwards
        if (rand() % 2):
            for i in range(size):
                mutant[i] += perturbation[i]
        # downwards
        else:
            for i in range(size):
                mutant[i] -= perturbation[i]
        return mutant

    cdef double[:] helper(self, double value):
        """ The helper function tasked with creating an array representing a Gaussian distribution centred
            on the given value.

        Args:
            value: The centre of the distribution.

        Returns:
            A memoryview representing a Gaussian distribution.
        """
        cdef double[:] result = np.zeros_like(self.domain)
        cdef double adj_width = self.width * rand() / float(RAND_MAX)
        cdef int i, size = self.domain.shape[0]
        for i in range(size):
            result[i] = self.mutation_epsilon * \
                exp(-((self.domain[i] - value)
                      * (self.domain[i] - value)) / adj_width)
        return result

cdef class GaussianDistributionMutator(Mutator):
    """ A Mutator implementation for use in mutating the entire vector according to a Gaussian distribution while
        ensuring that the area under the curve remains constant
    """
    # The epsilon to be used in mutations and the width of the distribution
    cdef double mutation_epsilon, width
    # The domain of the vectors to be used
    cdef double[:] domain

    def __init__(self, double mutation_epsilon, np.ndarray[double, ndim=1] domain, double width):
        self.mutation_epsilon = mutation_epsilon
        self.domain = domain
        self.width = width

    cdef double[:] cyDoMutation(self, double[:] vector):
        """ The Cython code used to mutate the given vector.

        Args:
            vector: the vector to be mutated

        Returns:
            A point mutated vector
        """
        cdef double[:] mutant = np.copy(vector)
        cdef int pos_up = int(rand() / float(RAND_MAX) * vector.shape[0]), pos_down = int(rand() / float(RAND_MAX) * vector.shape[0])
        cdef adjusted_epsilon = self.mutation_epsilon
        cdef int i, size = mutant.shape[0]
        cdef double min = mutant[0]
        for i in range(size):
            if (mutant[i] < min):
                min = mutant[i]
        if (min - adjusted_epsilon < 0):
            adjusted_epsilon = min
        cdef double width = self.width * rand() / float(RAND_MAX)
        while width < 0.001:
            width = self.width * rand() / float(RAND_MAX)
        cdef double[:] perturbation_up = self.helper(adjusted_epsilon, self.domain[pos_up], width), perturbation_down = self.helper(adjusted_epsilon, self.domain[pos_down], width)
        for i in range(mutant.shape[0]):
            mutant[i] += perturbation_up[i] - perturbation_down[i]
        return mutant

    def doMutation(self, np.ndarray[double, ndim=1] vector):
        """ A wrapper function to allow Python code to access the optimised Cython code. """
        return np.asarray(self.cyDoMutation(vector))

    cdef double[:] helper(self, double eps, double value, double width):
        """ The helper function tasked with creating an array representing a Gaussian distribution centred
            on the given value.

        Args:
            eps: The multiplier to be used for the distribution
            value: The centre of the distribution.
            width: The width of the distribution to be used

        Returns:
            A memoryview representing a Gaussian distribution.
        """
        cdef double[:] result = np.zeros_like(self.domain)
        cdef int i, size = self.domain.shape[0]
        for i in range(size):
            result[i] = eps * exp(-(self.domain[i] - value)**2 / width)
        return result

cdef int cy_within_bounds(double[:] vector, double lower, double upper):
    """ Checks if a vector is within the given bounds.
        Cython funciton.

    Args:
        vector: The vector to be checked
        lower: The lower bound
        upper: The upper bound

    Returns:
        1 if vector is within the bounds, 0 otherwise
    """
    cdef double mn = vector[0], mx = vector[0]
    cdef int i, size = vector.shape[0]
    for i in range(size):
        if vector[i] < mn:
            mn = vector[i]
        if vector[i] > mx:
            mx = vector[i]
    if lower != INFINITY and mn < lower:
        return 0
    if upper != -INFINITY and mx > upper:
        return 0
    return 1

cdef class StandardEvolver(Evolver):
    """ An Evolver implementation using the standard evolution algorithm """
    # The floating point tolerance to be used
    cdef double atol
    # The fitness function object to be used
    cdef fitness_function
    # The mutator to be used
    cdef mutator

    def __init__(self, fitness_function, mutator, double atol=1e-8):
        self.fitness_function = fitness_function
        self.mutator = mutator
        self.atol = atol

    cpdef do_step(self, np.ndarray[double, ndim=1] resident_surface):
        """ Performs the main body of the evolution.
            This implementation simply creates a mutant, compares its fitness against the resident and returns
            the most fit and whether this was the mutant

        Args:
            resident_surface: The resident surface array to be used for this step

        Returns:
            The resident at the end of the step which may be the original one and whether
                an invasion occurred.
        """
        cdef double[:] mutant = self.mutator.doMutation(resident_surface)
        cdef double fitness_resident = self.fitness_function.get(resident_surface), fitness_mutant = self.fitness_function.get(mutant)
        if fitness_resident < fitness_mutant and abs(fitness_resident - fitness_mutant) > self.atol:
            return np.asarray(mutant), (1 == 1)
        else:
            return resident_surface, (1 == 0)

cdef class MoranEvolver(Evolver):
    """ An Evolver implementation for the Moran process """
    # The comparison and floating point tolerances and mutation chance to be
    # used
    cdef double atol, ctol, mut_chance
    # The fitness function to be used
    cdef fitness_function
    # The mutator to be used
    cdef mutator

    def __init__(self, fitness_function, mutator, double ctol=1e-8, double atol=1e-8, double mut_chance=1e-2):
        self.fitness_function = fitness_function
        self.mutator = mutator
        self.ctol = ctol
        self.atol = atol
        self.mut_chance = mut_chance

    cpdef do_step(self, population, definitions):
        fitness = self.fitness_function.get(definitions)
        fitness = normalise(fitness)
        cdef double fit_sum = 0
        cdef int pop_sum = 0
        cdef int key
        cdef double fit
        for fit in fitness.values():
            fit_sum += fit
        cdef int count
        for count in population.values():
            pop_sum += count

        cdef double to_replicate = rand() / float(RAND_MAX) * fit_sum
        cdef int to_replace = int(rand() / float(RAND_MAX) * pop_sum)
        cdef int to_replicate_id = -1, to_replace_id = -1
        
        cdef double t = 0
        for key, fit in fitness.items():
            t += fit
            if t >= to_replicate:
                to_replicate_id = key
                break

        cdef int t2 = 0
        for key, count in population.items():
            t2 += count
            if t2 >= to_replace:
                to_replace_id = key
                break

        population[to_replace_id] -= 1

        cdef int max = 0
        cdef double[:] mutant
        cdef double[:] definition
        if rand() / float(RAND_MAX) < self.mut_chance:
            mutant = self.mutator.doMutation(
                np.asarray(definitions[to_replicate_id]))
            for key, definition in definitions.items():
                if key > max:
                    max = key
                if one_norm_distance(mutant, definition) < self.ctol:
                    population[key] += 1
                    break
            else:
                population[max + 1] = 1
                definitions[max + 1] = np.asarray(mutant)
        else:
            population[to_replicate_id] += 1

        if population[to_replace_id] == 0:
            population.pop(to_replace_id, None)
            definitions.pop(to_replace_id, None)

        return population, definitions

cdef class WrightFisherEvolver(Evolver):
    """ An Evolver implementation for the Moran process """
    # The comparison and floating point tolerances and mutation chance to be
    # used
    cdef double atol, ctol, mut_chance
    # The fitness function to be used
    cdef fitness_function
    # The mutator to be used
    cdef mutator

    def __init__(self, fitness_function, mutator, double ctol=1e-8, double atol=1e-8, double mut_chance=1e-2):
        self.fitness_function = fitness_function
        self.mutator = mutator
        self.ctol = ctol
        self.atol = atol
        self.mut_chance = mut_chance

    cpdef do_step(self, population, definitions):
        fitness = self.fitness_function.get(definitions)
        fitness = normalise(fitness)
        cdef double fit_sum = 0
        cdef int pop_sum = 0
        cdef int key
        cdef double fit
        for fit in fitness.values():
            fit_sum += fit
        cdef int count
        for count in population.values():
            pop_sum += count

        cdef int * replacements = <int * > malloc(pop_sum * cython.sizeof(int))
        if replacements is NULL:
            raise MemoryError()

        cdef double to_replicate
        cdef int i = 0, max = 0
        cdef double t
        while i < pop_sum:
            to_replicate = rand() / float(RAND_MAX) * fit_sum
            t = 0
            for key, fit in fitness.items():
                t += fit
                if t > to_replicate:
                    break
            replacements[i] = key
            if key > max:
                max = key
            i += 1

        cdef new_pop = dict(), new_def = dict()
        cdef double[:] mutant
        cdef double[:] definition
        for i in range(pop_sum):
            if rand() / float(RAND_MAX) < self.mut_chance:
                mutant = self.mutator.doMutation(
                    np.asarray(definitions[replacements[i]]))
                for key, definition in new_def.items():
                    if key > max:
                        max = key
                    if one_norm_distance(mutant, definition) < self.ctol:
                        new_pop[key] += 1
                        break
                else:
                    new_pop[max + 1] = 1
                    new_def[max + 1] = np.asarray(mutant)
                    max += 1
            elif replacements[i] in new_pop:
                new_pop[replacements[i]] += 1
            else:
                new_pop[replacements[i]] = 1
                new_def[replacements[i]] = definitions[replacements[i]]

        free(replacements)

        return new_pop, new_def

cpdef normalise(values):
    cdef double minval = min(values.itervalues())
    cdef int key
    for key in values.iterkeys():
        values[key] -= minval
    return values
        
def setup(int seed):
    srand(seed)
