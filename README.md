# MemRePs: learning task specifications with membership-respecting preference orders

This codebase provides an algorithm and implementation for the concept classes
provided in the paper "Learning Task Specifications from Preference and 
Membership Queries." 

## Installation Instructions

This project uses the [poetry](https://poetry.eustace.io/) python package/dependency management tool. Please familarize yourself with it
 and then, within this directory, run:
 
 ```
poetry install
```

We recommend setting up a separate python environment (using python 3.9)
 when installing to ensure no dependencies are misspecified.

## Usage

Within the `memreps` folder in this directory, the following files are
present:

1. `memreps.py` - This is the primary file that encodes the memrep algorithm described in the paper. We use
an existing implementation of the contextual multi-armed bandit algorithm for query selection. Within this
file, we define a 'learner' that will select queries based on the current evidence
(assumptions) it has, and will update those assumptions based on the response provided
by the oracle.
2. `explicit.py`, `finite.py`, and `implicit.py` - These files provide class abstractions of both
explicitly and implicitly defined concept classes, with function signatures for
the methods needed to define concept classes (such as membership of an atom within
a concept.)
3. `grid.py`, `monotone.py` - These files provide an example instantiation of a concept
class to be learned by the memreps algorithm, in the domain of monotone predicate
families (as elaborated on in section 4 of our paper.)
4. `dfa_learning.py` - This file provides an example instantiation of a concept class
to be learned by the memreps algorithm, in the setting of learning Deterministic
Finite Automata (DFA).

To run, we provide test files that both ensure the correctness of the memreps algorithm
itself as well as provide examples of target concepts to be learned by the memreps
algorithm. In particular, `tests/test_monotone.py/test_monotone_memreps` and
`tests/test_dfa_learning.py/test_gridworld_dfa` provide the two target concepts
that we learn in our experiments. These methods can be run standalone, or the 
entire test suite can be run with

```
poetry run pytest
```

The individual tests for each domain can be run as follows:
```
poetry run pytest -k monotone_memreps
poetry run pytest -k grid
```

Note that performance of query efficiency will vary across runs due to the 
randomness of our algorithm, as well as the randomness of the preference order
set in each experiment. The random seeds were set as outlined in the code
using the specified libraries.

For the results in our paper, all experiments were run on a single machine with a 2.4 GHz 8-Core Intel Core i9
processor. 

We have additionally provided plotting code, in `plot.py`, that can be run to reproduce the figures presented in 
the paper. Specifically, `run_experiment` will generate figures running the memreps algorithm on a list of 
specified cost ratios (membership relative to preference.) The number of trials to average results over can be 
modified, as well as the flag `monotone_exp` to evaluate either the monotone predicate setting or the DFA setting.