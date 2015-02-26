paramless
=========

In this project we try out evolution of function valued traits in both monomorphic and heteromorphic populations. 

The set-up is similar to the framework introduced in [Dieckmann et al.](http://www.sciencedirect.com/science/article/pii/S0022519305005266).

Evolving functions are widespread in [symbolic regression](http://en.wikipedia.org/wiki/Symbolic_regression) and [genetic programming](http://en.wikipedia.org/wiki/Genetic_programming). The difference here is that evolution happens on the functional space itself, and not on the symbolic space. This facilitates controlling for constraints in mutations (e.g. make sure that all mutants are probability distributions) and also means that no primitives or parameters are required ex-ante. Therefore paramless.

**Currently only works smoothly on Linux, some environment setup may be required for other platforms**

Installation instructions
-------------------------
To install the module for use, navigate to the src directory and use the command:

    python setup.py install
This will build the module and copy it to the appropriate directory for use.

Usage
------
There are two modes of use available within this project, the first involving a monomorphic population and the second having a heteromorphic population. The usages of both are briefly explained here. For more detail, see the example notebooks given (see below) as well as the documentation in the source code.

**Monomorphic:**
To set up an evolution run of a monomorphic population, 3 main details need to be specified, in the form of classes. These are the fitness function, the mutator and the evolution process.
To specify the fitness function, a class with a get(vector) function returning the fitness of the vector must be implemented. As an example there is an existing one, _DistanceFitnessFunction_.
The mutator is specified in a similar way, in the form of a class with a doMutation(vector) function returning the mutated vector. 4 implementations of these exist in the project, being two forms of a point mutator and a Gaussian mutator. The first form is a regular mutation while the second maintains the area under the function, useful in cases involving distributions.
Finally, the evolver must be specified as a class implementing a do_step(resident) function which returns the resident at the end of the step. For this the _StandardEvolver_ has been supplied but a custom process can also be used. Typically this class has both the mutator and fitness function as a member type.
To simplify the implementation of these classes, if needed, abstract classes have been supplied. These are the _FitnessFunction_, _Mutator_ and _Evolver_ classes. Subclassing these classes in an implementation will ensure that the necessary functions are implemented or a NotImplementedError will be raised.

Running the project is as simple as calling the _evolve_ function in paramless.py with the following arguments:

* initial_surface: The vector to be used as the initial population.
* evolver: The evolver to be used.
* iterations: The number of iterations the run should complete.
* return_time_series: An optional argument specifying whether a time series should be constructed and returned for the run. The time series is represented as a dict
* seed: Another optional argument allowing for the random seed to be set.
When completed, it will return the resident population at the end of the run, as well as a time series if requested.

**Heteromorphic:**
To represent a heteromorphic population, two dicts are used, a popullation dict to map from a type id to a count of that type in the population and definition dict to map between the id and the vector used as its function. As such, the setup of a heteromorphic run is different in a number of ways.
Firstly, the fitness function class's get function now takes a definition dict and returns a new dict mapping the type id to its fitness. Again, an example class, _ModelDistanceFitnessFunction_, is supplied.
As the mutation is calculated on one individual at a time, no change is needed for the mutator.
The evolver however needs to now take and return the population dict and definition dict for its do_step function. Here, the _MoranEvolver_ and _WrightFisherEvolver_ classes are supplied, implementing the Moran and Wright-Fisher processes respectively.

To set up and run an evolution run, the evolver will need to be called directly within a separate script. This has been done in metabolic_investment.py, an example showing the metabolic investment notebook implemented as a heteromorphic population model.
This example also shows how a JSON file can be used to easily load configuration options for a specific run without having to enter large amounts of arguments at the command line or edit the script. To call it, use:

    python metabolic_investment.py metabolic_test.json
This will create a CSV file named in the JSON file containing the progress of the model over time and will also print the definition of the type which is most common at the completion of the run.


Some examples
-------------
 * [Evolving fixed target functions](http://nbviewer.ipython.org/github/AlexRothwell7/paramless/blob/master/src/on_a_line_examples.ipynb)
 * [Evolution of metabolic investment](http://nbviewer.ipython.org/github/AlexRothwell7/paramless/blob/master/src/evolution%20of%20metabolic%20investment.ipynb)
 * [Evolution of seasonal flowering](http://nbviewer.ipython.org/github/AlexRothwell7/paramless/blob/master/src/seasonal_flowering.ipynb)
 * [Improving runtime with Cython](http://nbviewer.ipython.org/github/AlexRothwell7/paramless/blob/master/src/cython_usage_example.ipynb)
