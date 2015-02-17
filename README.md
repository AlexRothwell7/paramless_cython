paramless
=========

In this project we try out evolution of function valued traits in monomorphic populations. 

The set-up is similar to the framework introduced in [Dieckmann et al.](http://www.sciencedirect.com/science/article/pii/S0022519305005266).

Evolving functions are widespread in symbolic regression and [genetic programming](http://en.wikipedia.org/wiki/Genetic_programming). The difference here is that evolution happens on the functional space itself, and not on the symbolic space. This facilitates controlling for constraints in mutations (e.g. make sure that all mutants are probability distributions) and also means that no primitives or parameters are required ex-ante. Therefore paramless.

Some examples
-------------
 * [Evolving fixed target functions](http://nbviewer.ipython.org/github/AlexRothwell7/paramless/blob/master/notebooks/on_a_line_examples.ipynb)
 * [Evolution of metabolic investment](http://nbviewer.ipython.org/github/AlexRothwell7/paramless/blob/master/notebooks/evolution%20of%20metabolic%20investment.ipynb)
 * [Evolution of seasonal flowering](http://nbviewer.ipython.org/github/AlexRothwell7/paramless/blob/master/notebooks/seasonal_flowering_fast.ipynb)
 * [Improving runtime with Cython] (http://nbviewer.ipython.org/github/AlexRothwell7/paramless/blob/master/notebooks/cython_usage_example.ipynb)


