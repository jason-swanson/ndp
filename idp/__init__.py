r'''A package for working with the iterated Dirichlet process.

# Description

Consider a situation in which there are several agents, all from the
same population. Each agent undertakes a sequence of actions. These
actions are chosen according to the agent's particular tendencies.
Although different agents have different tendencies, there may be
patterns in the population.

We observe a certain set of agents over a certain amount of time. Based
on these observations, we want to make probabilistic forecasts about two
things:
* The future behavior of the agents we have observed.
* The behavior of a new (unobserved) agent from the population.

An iterated Dirichlet process is a mathematical tool that can model
this. The mathematical details are provided in the preprint, "[The
iterated Dirichlet process and applications to Bayesian
inference](https://arxiv.org/abs/2505.00451)," by Evan Donald and Jason
Swanson. (In this rest of this documentation, we refer to this preprint
as the "Paper.") This package, `idp`, is a Python implementation of the
iterated Dirichlet process (IDP).

Below are four examples using the `idp` package.

# The pressed penny machine

{}

# Flicking thumbtacks

{}

# Amazon reviews

{}

# Video game leaderboards

{}

'''

from collections import Counter
import numpy as np
from idp import tools, lboard
from idp.examples import coinexpl, tackexpl, revexpl, gameexpl

__all__ = ['tools', 'lboard', 'idpModel']

def idpModel(colConc, rowConc, baseMeas=None, rowCounts=None, data=None):
    r'''Creates an IDP model in which the agents have $L$ possible
    actions. These actions are represented by the integers $0, \ldots,
    L - 1$.

    **Arguments:**

    * **colConc** (`float`): The column concentration of the IDP.
    * **rowConc** (`float`): The row concentration of the IDP.
    * **baseMeas** (`array_like` or `int`): An array with one row that
      represents the base measure of the IDP. The value of `baseMeas[l]`
      is the prior probability that an agent will perform action `l`. If
      `baseMeas` is passed an integer `L`, then the uniform measure on
      $\\{{0, \ldots, L - 1\\}}$ will be used. Any list of nonnegative
      numbers with a positive sum may be passed to this argument. If it
      does not already sum to 1, then the function will normalize it so
      that it does.
    * **rowCounts** (`array_like`): An array of observations. The value
        of `rowCounts[m][l]` is the number of times action `l` was
        observed from agent `m`.
    * **data** (`list[list[int]]`): A jagged array of observations. The
        value of `data[m][n]` is the `n`-th observed action of agent
        `m`. If `rowCounts` is not provided, then it will be built from
        `data`.

    **Returns:**

    * `idp.tools.IDPModel`: The IDP model built from the given
        parameters.

    **Raises:**

    * **ValueError:** If `rowCounts` and `data` are both provided, or if
        neither are provided.

    '''
    if rowCounts is not None and data is not None:
        raise ValueError('Cannot provide both `rowCounts` and `data`')
    if rowCounts is None and data is None:
        raise ValueError('Must provide either `rowCounts` or `data`')

    if isinstance(baseMeas, int):
        baseMeas = [1 / baseMeas] * baseMeas
    arr = np.array(baseMeas)
    baseMeas = arr / np.sum(arr)

    if rowCounts is None:
        rowCounts = [
            [Counter(row)[state] for state in range(len(baseMeas))]
            for row in data
        ]
    rowCounts = np.array(rowCounts)

    return tools.IDPModel(colConc, rowConc, baseMeas, rowCounts)

__doc__ = __doc__.format(
    coinexpl.__doc__,
    tackexpl.__doc__,
    revexpl.__doc__,
    gameexpl.__doc__
)
