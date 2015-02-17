#cython: boundscheck=False
#cython: wraparound=False
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport fabs
from numpy.math cimport INFINITY
from libc.stdlib cimport rand, srand, RAND_MAX
from libc.math cimport exp

#The maximum number of iterations to be used in attempting to get a new mutant
cdef int MAX_ITERATIONS = 1000000


cdef class FitnessFunction(object):
    """ An abstract class to be subclassed for use by an Evolver """
    def get(self, resident, mutant):
        """ Abstract function to be implemented by a subclass.
        Evaluates the fitness function of a resident and mutant according to this object
        
        Args:
            resident: The resident to be evaluated
            mutant: The mutant to be evaluated
        
        Returns:
            The fitness of both the resident and mutant
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


cdef class DistanceFitnessFunction(FitnessFunction):
    """ A FitnessFunction implementation for use for fitting to a given target function """
    #The target function
    cdef double [:] target
    def __init__(self, np.ndarray[double, ndim=1] target):
        self.target = target

    @cython.cdivision(True)
    cpdef get(self, double [:] resident, double [:] mutant):
        """ Evaluates the fitness function of a resident and mutant according to:
            fitness = 1/distance
        
        Args:
            resident: The resident to be evaluated
            mutant: The mutant to be evaluated
        
        Returns:
            The fitness of both the resident and mutant
        """
        fitness_resident = 1.0 / one_norm_distance(resident, self.target)
        fitness_mutant = 1.0 / one_norm_distance(mutant, self.target)
        return fitness_resident, fitness_mutant

cdef one_norm_distance(double [:] u, double[:] v):
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
    #The epsilon to be used in mutations and absolute bounds for resulting vectors
    cdef double mutation_epsilon, lower_bound, upper_bound
    def __init__(self, double mutation_epsilon, double lower_bound=INFINITY, double upper_bound=-INFINITY):
        self.mutation_epsilon = mutation_epsilon
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    cdef double [:] cyDoMutation(self, double [:] vector):
        """ The Cython code used to mutate the given vector.
        
        Args:
            vector: the vector to be mutated
            
        Returns:
            A point mutated vector
        """
        cdef int is_inside = 0
        cdef int attempt = 0
        cdef double [:] mutant
        while (is_inside != 1):
            mutant = self.attempt_point_mutation(vector)
            is_inside=cy_within_bounds(mutant, self.lower_bound, self.upper_bound)
            attempt+=1
            if attempt > MAX_ITERATIONS:
                raise RuntimeError("Attempted too many mutations without producing anything within bounds")
        return mutant
        
    def doMutation(self, np.ndarray[double, ndim=1] vector):
        """ A wrapper function to allow Python code to access the optimised Cython code. """
        return np.asarray(self.cyDoMutation(vector))

    cdef double [:] attempt_point_mutation(self, double [:] vector):
        """ The body of the mutation code. Mutation is completed through randomly choosing a point
            in the vector and increasing or decreasing its value by the mutation epsilon.
        
        Args:
            vector: The vector to be mutated
            
        Returns:
            The mutated vector        
        """
        cdef double [:] mutant = np.copy(vector)
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
    #The epsilon to be used in the mutations
    cdef double mutation_epsilon
    def __init__(self, double mutation_epsilon):
        self.mutation_epsilon = mutation_epsilon
    
    cdef double [:] cyDoMutation(self, double [:] vector):
        """ The Cython code used to mutate the given vector.
        
        Args:
            vector: the vector to be mutated
            
        Returns:
            A point mutated vector
        """
        cdef double [:] mutant = np.copy(vector)
        cdef int pos_up = int(rand() / float(RAND_MAX) * vector.shape[0]), pos_down = int(rand() / float(RAND_MAX) * vector.shape[0])
        cdef adjusted_epsilon = self.mutation_epsilon
        if (mutant[pos_down] - adjusted_epsilon < 0 ):
            adjusted_epsilon = mutant[pos_down]
        mutant[pos_up] += adjusted_epsilon
        mutant[pos_down] -= adjusted_epsilon
        return mutant
        
    def doMutation(self, np.ndarray[double, ndim=1] vector):
        """ A wrapper function to allow Python code to access the optimised Cython code. """
        return np.asarray(self.cyDoMutation(vector))
        
cdef class GaussianMutator(Mutator):
    """ A Mutator implementation for use in mutating the entire vector according to a Gaussian distribution """
    #The epsilon to be used in mutations, the width of the distribution and absolute bounds for resulting vectors
    cdef double mutation_epsilon, width, lower_bound, upper_bound
    #The domain of the vectors to be used
    cdef double [:] domain
    def __init__(self, double mutation_epsilon, np.ndarray[double, ndim=1] domain, double width, double lower_bound=INFINITY, double upper_bound=-INFINITY):
        self.mutation_epsilon = mutation_epsilon
        self.domain = domain
        self.width = width
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    cdef double [:] cyDoMutation(self, double [:] vector):
        """ The Cython code used to mutate the given vector.
        
        Args:
            vector: the vector to be mutated
            
        Returns:
            A point mutated vector
        """
        cdef int is_inside = 0
        cdef int attempt = 0
        cdef double [:] mutant
        while (is_inside != 1):
            mutant = self.attempt_gaussian_mutation(vector)
            is_inside=cy_within_bounds(mutant, self.lower_bound, self.upper_bound)
            attempt+=1
            if attempt > MAX_ITERATIONS:
                raise RuntimeError("Attempted too many mutations without producing anything within bounds")
        return mutant
        
    def doMutation(self, np.ndarray[double, ndim=1] vector):
        """ A wrapper function to allow Python code to access the optimised Cython code. """
        return np.asarray(self.cyDoMutation(vector))

    cdef double [:] attempt_gaussian_mutation(self, double [:] vector):
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
        cdef double [:] mutant = np.copy(vector)
        cdef double [:] perturbation = self.helper(value)
        cdef int i, size = mutant.shape[0]
        #upwards
        if (rand() % 2):
            for i in range(size):
                mutant[i] += perturbation[i]
        #downwards
        else:
            for i in range(size):
                mutant[i] -= perturbation[i]
        return mutant
        
    cdef double [:] helper(self, double value):
        """ The helper function tasked with creating an array representing a Gaussian distribution centred
            on the given value.
            
        Args:
            value: The centre of the distribution.
            
        Returns:
            A memoryview representing a Gaussian distribution.
        """
        cdef double [:] result = np.zeros_like(self.domain)
        cdef double adj_width = self.width * rand() / float(RAND_MAX)
        cdef int i, size = self.domain.shape[0]
        for i in range(size):
            result[i] = self.mutation_epsilon * exp(-((self.domain[i] - value)*(self.domain[i] - value))/adj_width)
        return result

cdef class GaussianDistributionMutator(Mutator):
    """ A Mutator implementation for use in mutating the entire vector according to a Gaussian distribution while
        ensuring that the area under the curve remains constant
    """
    #The epsilon to be used in mutations and the width of the distribution
    cdef double mutation_epsilon, width
    #The domain of the vectors to be used
    cdef double [:] domain
    def __init__(self, double mutation_epsilon, np.ndarray[double, ndim=1] domain, double width):
        self.mutation_epsilon = mutation_epsilon
        self.domain = domain
        self.width = width
    
    cdef double [:] cyDoMutation(self, double [:] vector):
        """ The Cython code used to mutate the given vector.
        
        Args:
            vector: the vector to be mutated
            
        Returns:
            A point mutated vector
        """
        cdef double [:] mutant = np.copy(vector)
        cdef int pos_up = int(rand() / float(RAND_MAX) * vector.shape[0]), pos_down = int(rand() / float(RAND_MAX) * vector.shape[0])
        cdef adjusted_epsilon = self.mutation_epsilon
        cdef int i, size = mutant.shape[0]
        cdef double min = mutant[0]
        for i in range(size):
            if (mutant[i] < min):
                min = mutant[i]
        if (min - adjusted_epsilon < 0 ):
            adjusted_epsilon = min
        cdef double width = self.width * rand() / float(RAND_MAX)
        while width < 0.001:
            width = self.width * rand() /float(RAND_MAX)
        cdef double [:] perturbation_up = self.helper(adjusted_epsilon, self.domain[pos_up],width), perturbation_down = self.helper(adjusted_epsilon, self.domain[pos_down],width)
        for i in range(mutant.shape[0]):
            mutant[i] += perturbation_up[i] - perturbation_down[i]
        return mutant
        
    def doMutation(self, np.ndarray[double, ndim=1] vector):
        """ A wrapper function to allow Python code to access the optimised Cython code. """
        return np.asarray(self.cyDoMutation(vector))
        
        
    cdef double [:] helper(self, double eps, double value, double width):
        """ The helper function tasked with creating an array representing a Gaussian distribution centred
            on the given value.
            
        Args:
            eps: The multiplier to be used for the distribution
            value: The centre of the distribution.
            width: The width of the distribution to be used
            
        Returns:
            A memoryview representing a Gaussian distribution.
        """
        cdef double [:] result = np.zeros_like(self.domain)
        cdef int i, size = self.domain.shape[0]
        for i in range(size):
            result[i] = eps * exp(-(self.domain[i] - value)**2/width)
        return result

cdef int cy_within_bounds(double [:] vector, double lower, double upper):
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
    #The floating point tolerance to be used
    cdef double atol
    #The fitness function object to be used
    cdef fitness_function
    #The mutator to be used
    cdef mutator
    def __init__(self, fitness_function, mutator, double atol = 1e-8):
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
        cdef double [:] mutant = self.mutator.doMutation(resident_surface)
        cdef double fitness_resident, fitness_mutant
        [fitness_resident, fitness_mutant] = self.fitness_function.get(
            resident_surface, mutant)
        if fitness_resident < fitness_mutant and abs(fitness_resident - fitness_mutant) > self.atol:
            return np.asarray(mutant), (1==1)
        else:
            return resident_surface, (1==0)
            
            
def setup(int seed):
    srand(seed)
