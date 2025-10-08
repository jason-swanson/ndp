'''The `NDPModel` class and related objects.

'''

from random import choices
import numpy as np
from scipy.special import gammaln, gammainc
from scipy.stats import pareto, gamma, beta, dirichlet
from scipy.stats import rv_continuous, gaussian_kde, ecdf
from statsmodels.distributions.empirical_distribution import ECDFDiscrete

__all__ = ['logBeta', 'gamer', 'Measure', 'IDPModel', 'WeightedSim']

def logBeta(alpha):
    '''Returns the logarithm of the multivariate beta function.

    **Arguments:**

    * **alpha** (`list`[`float`]): A list of arguments for the function.

    **Returns:**

    * `float`: The the natural logarithm of value of the multivariate
        beta function, evaluated at the given list of arguments.

    '''
    return np.sum(gammaln(alpha)) - gammaln(np.sum(alpha))

# pylint: disable=invalid-name, missing-class-docstring
# pylint: disable=arguments-differ
# pylint: disable=too-many-arguments, too-many-positional-arguments

class gamer_gen(rv_continuous):
    def _pdf(self, x, r, a):
        return r / a ** r * np.exp(gammaln(a + r) - gammaln(a)) * (
            x ** (-r - 1) * gammainc(a + r, a * x)
        )

    def _rvs(self, r, a, size=None, random_state=None):
        meanScores = pareto.rvs(
            r,
            size=size,
            random_state=random_state
        )
        return gamma.rvs(
            a,
            scale=meanScores / a,
            random_state=random_state
        )

    def _munp(self, n, r, a):
        if r <= n:
            return np.inf
        return r / a ** n * np.exp(gammaln(a + n) - gammaln(a)) / (r - n)

# pylint: enable=invalid-name, missing-class-docstring
# pylint: enable=arguments-differ
# pylint: enable=too-many-arguments, too-many-positional-arguments

gamer = gamer_gen(a=0.0, name='gamer')
r'''A gamer distribution as in Section 6.5 of the Paper.

As an instance of the `scipy.stats.rv_continuous` class, `gamer` object
inherits from it a collection of generic methods (see below for the full
list), and completes them with details specific for this particular
distribution.

The probability density function of `gamer` is:
$$
  f(x) = \frac r {a^r \, \Gamma(a)} \,
    x^{-r - 1} \, \int_0^{ax} y^{a + r - 1} e^{-y} \, dy
$$
for $x \ge 0$, $r > 0$, and $a > 0$. $\Gamma$ is the gamma function
(`scipy.special.gamma`).

`gamer` takes $r$ and $a$ as shape parameters.

The probability density above is defined in the "standardized" form. To
shift and/or scale the distribution use the `loc` and `scale`
parameters. Specifically, `gamer.pdf(x, r, a, loc, scale)` is
identically equivalent to `gamer.pdf(y, r, a) / scale` with
`y = (x - loc) / scale`.

In the notation of the Paper, `gamer` represents the
$\mathrm{Gamer}(r, 1, a)$ distribution. To obtain the
$\mathrm{Gamer}(r, c, a)$ distribution, set the `scale` parameter to
$c$.

**Methods:**

* **rvs(r, a, loc=0, scale=1, size=1, random_state=None)**:
    Random variates.
* **pdf(x, r, a, loc=0, scale=1)**:
    Probability density function.
* **logpdf(x, r, a, loc=0, scale=1)**:
    Log of the probability density function.
* **cdf(x, r, a, loc=0, scale=1)**:
    Cumulative distribution function.
* **logcdf(x, r, a, loc=0, scale=1)**:
    Log of the cumulative distribution function.
* **sf(x, r, a, loc=0, scale=1)**:
    Survival function (also defined as ``1 - cdf``).
* **logsf(x, r, a, loc=0, scale=1)**:
    Log of the survival function.
* **ppf(q, r, a, loc=0, scale=1)**:
    Percent point function (inverse of ``cdf`` --- percentiles).
* **isf(q, r, a, loc=0, scale=1)**:
    Inverse survival function (inverse of ``sf``).
* **moment(order, r, a, loc=0, scale=1)**:
    Non-central moment of the specified order.
* **stats(r, a, loc=0, scale=1, moments='mv')**:
    Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
* **entropy(r, a, loc=0, scale=1)**:
    (Differential) entropy of the RV.
* **fit(data)**:
    Parameter estimates for generic data. See
    `scipy.stats.rv_continuous.fit` for detailed documentation of the
    keyword arguments.
* **expect(func, args=(r, a), loc=0, scale=1, lb=None, ub=None,
conditional=False, **kwds)**:
    Expected value of a function (of one argument) with respect to the
    distribution.
* **median(r, a, loc=0, scale=1)**:
    Median of the distribution.
* **mean(r, a, loc=0, scale=1)**:
    Mean of the distribution.
* **var(r, a, loc=0, scale=1)**:
    Variance of the distribution.
* **std(r, a, loc=0, scale=1)**:
    Standard deviation of the distribution.
* **interval(confidence, r, a, loc=0, scale=1)**:
    Confidence interval with equal areas around the median.

'''

class Measure:
    r'''A finite, signed Borel measure on the real line.

    `Measure` objects can be added to each other and multiplied on the
    left by a `float`. These operations return new `Measure` objects.
    When adding two instances, the `bandwidth` of the result is the
    maximum of the bandwidths of the summands.

    '''
    def __init__(self, cdf, pdf, mean=None, totalMass=1):
        '''Constructs a `Measure` object from the given arguments.

        '''
        self._cdf = cdf
        self._pdf = pdf
        self._mean = mean
        self._totalMass = totalMass

    @property
    def cdf(self):
        r'''`func`: The distribution function of the measure. If $\mu$
        is the measure, then this is the function $t \mapsto
        \mu((-\infty, t])$.
        '''
        return self._cdf

    @property
    def pdf(self):
        '''`func`: The density of the measure, which is the derivative
        of `cdf`. If `cdf` is not differentiable, then an approximate
        density should be provided.
        '''
        return self._pdf

    @property
    def mean(self):
        r'''`float` or `None`: If the mean is stored in the `Measure`
        instance, then it should equal $\int_{-\infty}^\infty x \,
        dF(x)$, where $F$ is the distribution function. The mean of the
        sum of two `Measure` instances will be `None` if either mean is
        `None`.
        '''
        return self._mean

    @property
    def totalMass(self):
        r'''`float`: The total mass of the measure. If $\mu$ is the
        measure, then this $\mu(\mathbb{R})$.
        '''
        return self._totalMass

    @property
    def bandwidth(self):
        '''`float`: If `pdf` is an approximate density created with the
        `gaussian_kde` function from the `scipy.stats` module, then this
        is the bandwidth that was used to create it. Is 0 otherwise. See
        `NDPModel.law` for more details.
        '''
        try:
            return self.pdf.factor
        except AttributeError:
            return 0

    def __rmul__(self, scalar):
        def pdf(t):
            return scalar * self.pdf(t)
        pdf.factor = self.bandwidth
        mean = (
            None if self._mean is None
            else scalar * self._mean
        )
        return Measure(
            lambda t: scalar * self._cdf(t),
            pdf,
            mean,
            scalar * self._totalMass
        )

    def __add__(self, other):
        def pdf(t):
            return self._pdf(t) + other.pdf(t)
        pdf.factor = max(self.bandwidth, other.bandwidth)
        mean = (
            None if self._mean is None or other.mean is None
            else self._mean + other.mean
        )
        return Measure(
            lambda t: self._cdf(t) + other.cdf(t),
            pdf,
            mean,
            self._totalMass + other.totalMass
        )

class IDPModel:
    r'''An NDP model on the state space $S = \\{0, ..., L - 1\\}$.

    The NDP model consists of both an NDP and some observed data. See
    Section 6.1 of the Paper for details.

    '''

    def __init__(self, colConc, rowConc, baseMeas, rowCounts):
        '''Uses the given arguments to set the corresponding properties.
        The `baseMeas` and `rowCounts` arguments should be numpy arrays
        of the same shape as the corresponding properties.

        '''
        self._idp = {
            'colConc': colConc,
            'rowConc': rowConc,
            'baseMeas': baseMeas
        }
        self._rowCounts = rowCounts

        rowConc = self._idp['rowConc']
        baseMeas = self._idp['baseMeas']
        logDenom = logBeta(rowConc * baseMeas)
        self._likelihoods = [
            np.exp(logBeta(rowConc * baseMeas + row) - logDenom)
            for row in rowCounts
        ]

        self._logScale = 0
        self._weightedSims = []

        self._savedSims = []
        '''The state of `self._weightedSims` just prior to the last
        successful call to `setEss`. Is reset by `addSims`, `clearSims`,
        and `restoreSims`.
        '''

        self._allSims = []
        '''The list of all simulations ever created by `addSims`.'''

    @property
    def colConc(self):
        r'''`float`: The column concentration parameter. Should be
        positive. Denoted by $\kappa$ in Section 6.1 of the Paper.
        '''
        return self._idp['colConc']

    @property
    def rowConc(self):
        r'''`float`: The row concentration parameter. Should be
        positive. Denoted by $\varepsilon$ in Section 6.1 of the Paper.
        '''
        return self._idp['rowConc']

    @property
    def baseMeas(self):
        r'''`tuple`(`float`): The base probability vector. Should be a
        sequence of positive numbers that sum to 1. Denoted by
        $\boldsymbol{p} = (p_0, \ldots, p_{L - 1})$ in Section 6.1 of
        the Paper, where $p_\ell = \varrho(\\{\ell\\})$.
        '''
        return tuple(self._idp['baseMeas'])

    @property
    def rowCounts(self):
        r'''`tuple`(`tuple`(`int`)): The row counts for the observed
        data. If `row` is $m - 1$ and `state` is $\ell$, then
        `rowCounts`[`row`][`state`] is denoted by $\overline{y}_{m\ell}$
        in Section 6.1 of the Paper. It is the number of observed
        instances of `state` in the given row of data.
        '''
        return tuple(tuple(row) for row in self._rowCounts)

    @property
    def likelihoods(self):
        r'''`tuple`(`float`): The prior likelihoods. If `row` equals
        $m - 1$, then `likelihoods`[`row`] is denoted by
        $\varrho_N(A_m)$ in Section 6.1 of the Paper. It is the prior
        probability of seeing the row counts for the given row.
        '''
        return tuple(self._likelihoods)

    @property
    def logScale(self):
        r'''`float`: The log scale factor to use for weighted
        simulations. Denoted by $\log c$ in Section 6.1 of the Paper.
        Defaults to 0. Altering this value runs the `clearSims` method.
        '''
        return self._logScale

    @logScale.setter
    def logScale(self, value):
        self._logScale = value
        self.clearSims()

    @property
    def weightedSims(self):
        r'''`tuple`(`WeightedSim`): A tuple of distinct (and therefore
        independent) `WeightedSim` instances. `weightedSims`[k] is
        denoted by $(t^k, \boldsymbol{\theta}_M^{*, k})$ in Section 6.1
        of the Paper, and has total weight $V_k$. This property is empty
        if the `addSims` method has never been run.
        '''
        return tuple(self._weightedSims)

    @property
    def ess(self):
        '''`float`: The effective sample size, approximated with the
        sample variance. Denoted by $K_e''$ in Section 4.2 of the Paper.
        '''
        numSims = len(self._weightedSims)
        if numSims == 0:
            return 0
        essPop = (
            sum(sim.totalWeight      for sim in self._weightedSims) ** 2 /
            sum(sim.totalWeight ** 2 for sim in self._weightedSims)
        )
        if essPop == numSims:
            return numSims
        return essPop * (numSims - 1) / (numSims - essPop / numSims)

    def addSims(self, num):
        '''Creates new `WeightedSim` instances and adds them to the
        `weightedSims` property.

        **Arguments:**
        
        * **num** (`int`): The number of distinct `WeightedSim`
            instances to create.

        '''
        newSims = [WeightedSim(self) for _ in range(num)]
        self._weightedSims.extend(newSims)
        self._savedSims = self._weightedSims.copy()
        self._allSims.extend(newSims)

    def clearSims(self):
        '''Deletes all weighted simulations contained in the
        `weightedSims` property.

        '''
        self._weightedSims = []
        self._savedSims = []

    def restoreSims(self):
        '''Restores all `WeightedSim` instances ever created by the
        `addWeigtedSims` method.

        '''
        self._weightedSims = self._allSims.copy()
        self._savedSims = self._weightedSims.copy()

    def setEss(self, minEss, maxDel=None):
        '''Tries to set the `ess` to the given value by deleting
        simulations from `weightedSims`. Simulations are deleted in
        order of their `WeightedSim.totalWeight` property, starting with
        the largest.

        **Arguments:**

        * **minEss** (`float`): The minimum value of `ess` to achieve.
            The method will stop deleting simulations when `ess` reaches
            or exceeds this value.
        * **maxDel** (`int`): If provided, the method will not delete
            more than this number of simulations.

        **Returns:**

        * `int`: The number of simulations deleted.

        **Raises:**

        * **ValueError:** If
            1. `minEss` does not exceed `ess`;
            2. `minEss` is not less than the total number of simulations
                in `weightedSims`;
            3. `maxDel` is negative or zero; or
            4. the method is unable to achieve the given value for
                `ess`. In this case, all deletions are reversed and the
                `weightedSims` property is left unaltered.

        '''
        savedSims = self._weightedSims.copy()
        # Temporarily stored. Will be transferred to `self._savedSims`
        # if the method finished without error.

        if self.ess >= minEss:
            raise ValueError(f'ESS is already {self.ess}')
        if minEss >= len(self._weightedSims):
            raise ValueError('Not enough simulations')
        if maxDel is not None and maxDel <= 0:
            raise ValueError('`maxDel` must be positive')

        # make `minKeep`, the minimum number of simulations to keep
        if maxDel is None:
            minKeep = minEss
        else:
            minKeep = len(self._weightedSims) - maxDel

        # try to reach `minEss` without deleting too many sims
        sortedSims = sorted(
            self._weightedSims,
            key=lambda sim: sim.totalWeight,
            reverse=True
        )
        for sim in sortedSims:
            self._weightedSims.remove(sim)
            if self.ess >= minEss or len(self._weightedSims) <= minKeep:
                break

        # check for success
        if self.ess < minEss:
            self._weightedSims = savedSims # undo deletions
            raise ValueError(f'Cannot achieve an ESS of {minEss}')

        self._savedSims = savedSims # on success, save for undo

        return len(savedSims) - len(self._weightedSims)

    def undoSetEss(self):
        '''Undoes the last successful call to `setEss`. Cannot undo more
        than one call. Cannot undo if `addSims`,
        `clearSims`, or `restoreSims` has been called since the
        last successful call to `setEss`.

        **Raises:**

        * **ValueError:** If undo is not allowed.

        '''
        if self._weightedSims == self._savedSims:
            raise ValueError('Nothing to undo')
        self._weightedSims = self._savedSims
        self._savedSims = self._weightedSims.copy()

    def prior(self, mapping, numSamples=10000, bandwidth=None):
        r'''Let the random vector $(X_0, X_1, \ldots, X_{L - 1})$ have a
        Dirichlet distribution with parameters `colConc` * `baseMeas`.
        This method returns the distribution of a random variable of the
        form $Y = f(X_0, X_1, \ldots, X_{L - 1})$. The distribution of
        $Y$ is a Borel probability measure on the real line. This method
        therefore returns a `Measure` object.

        If a function is passed to the `mapping` argument, then the cdf
        and pdf of $Y$ are estimated by sampling and Gaussian kernel
        density estimation (KDE). See `agentLaw` for details about
        KDE.

        This method is used in the implementation of (8.24), (8.26),
        (9.6), and the final displayed equation in Section 9 of the
        paper.

        **Arguments:**

        * **mapping** (`func` or `int`): The function to apply to the
            random vector $(X_0, X_1, \ldots, X_{L - 1})$. Denoted by
            $f$ in the above description. Passing this argument an
            integer $\ell$ is equivalent to using the function $f(X_0,
            X_1, \ldots, X_{L - 1}) = X_\ell$.
        * **numSamples** (`int`): The number of samples to use when
            estimating the cdf and pdf.
        * **bandwidth** (`float`): The bandwidth to use in the KDE.

        **Returns:**

        * `Measure`: The distribution of $Y$.

        '''
        if not callable(mapping):
            prior = beta(
                self._idp['rowConc'] * self._idp['baseMeas'][mapping],
                self._idp['rowConc'] * (1 - self._idp['baseMeas'][mapping])
            )
            return Measure(prior.cdf, prior.pdf, prior.mean())
        rawSamples = dirichlet.rvs(
            self._idp['rowConc'] * self._idp['baseMeas'],
            numSamples
        )
        samples = [mapping(sample) for sample in rawSamples]
        pdf = gaussian_kde(samples, bandwidth)
        return Measure(
            ecdf(samples).cdf.evaluate,
            pdf,
            np.average(samples),
            pdf.factor
        )

    def law(self, mapping, newAgent=False, bandwidth=None):
        r'''Let $\theta_{m\ell}$ be the unknown probability that agent
        $m$ will perform action $\ell$, and let $\theta$ be the matrix
        with entries $\theta_{m\ell}$. Let $\vartheta_\ell$ be the
        unknown probability that a new, unobserved agent will perform
        action $\ell$ This method returns the law (or distribution) of a
        random variable of the form $\Theta = f(\vartheta_0, \ldots,
        \vartheta_{L - 1})$, or of the form $\Theta = f(\theta)$. To get
        the former, set `newAgent` to `True`. To get the latter, set
        `newAgent` to `False`. In either case, set `mapping` to $f$.

        The law of $\Theta$ is a Borel probability measure on the
        real line. This method therefore returns a `Measure` object.
        However, the law of $\Theta$ has a discrete component, so
        it does not have a density. The `Measure.pdf` attribute of the
        returned object is an approximate density, created with Gaussian
        kernel density estimation (KDE). More specifically, it treats a
        point mass $\delta_x$ as having a Gaussian density with mean $x$
        and standard deviation $h$, where $h$ is the "bandwidth" of the
        estimate.

        This method implements either (5.12) or (5.10) in the Paper,
        depending on whether `newAgent` is `True` or `False`,
        respectively.

        **Arguments:**

        * **mapping** (`func`): The function $f$ in the above
            description. This function should take a single `ndarray`
            object as an argument. If `newAgent` is `True`, the array
            should have the same shape as `baseMeas`. Otherwise, it
            should have the same shape as `rowCounts`.
        * **newAgent** (`bool`): Determines the form of $\Theta$ in the
            above description.
        * **bandwidth** (`float` or `None`): The bandwidth to use in the
            Gaussian KDE. If `None`, the bandwidth will be calculated
            using [Scott's Rule](https://doi.org/10.1002/9780470316849).
            The calculated value can be retrieved by accessing the
            `Measure.bandwidth` property of the returned object. This
            argument is passed directly to the `gaussian_kde` function
            in the `scipy.stats` module. See that documentation for
            further details.

        **Returns:**

        * `Measure`: The law of $\Theta$.

        **Raises:**

        * **ValueError**: If the `weightedSims` property is empty.

        '''
        if not self._weightedSims:
            raise ValueError('No weighted simulations found')

        numRows = len(self._rowCounts)

        if not newAgent:
            # Implement (8.10).
            samples = np.array([
                mapping(weightedSim.rowDistSims)
                for weightedSim in self._weightedSims
            ])
            weights = np.array([
                weightedSim.totalWeight
                for weightedSim in self._weightedSims
            ])
        else:
            # Implement (8.24). In this case, the `sample` list is
            # ordered as
            # \[
            #   \theta_1^{*, 1}, \ldots, \theta_M^{*, 1},
            #   \theta_1^{*, 2}, \ldots, \theta_M^{*, 2},
            #   \ldots,
            #   \theta_1^{*, K}, \ldots, \theta_M^{*, K}
            # \]
            samples = np.array([
                mapping(weightedSim.rowDistSims[m])
                for weightedSim in self._weightedSims # outer loop
                for m in range(numRows)  # inner loop
            ])
            weights = np.concatenate([
                [weightedSim.totalWeight] * numRows
                for weightedSim in self._weightedSims
            ])

        # Create the sample law
        sampleCDF = ECDFDiscrete(samples, weights)
        samplePDF = gaussian_kde(samples, bandwidth, weights)
        sampleLaw = Measure(
            sampleCDF,
            samplePDF,
            np.average(samples, weights=weights),
            samplePDF.factor
        )

        # If implementing (8.10), return only the sample law. Otherwise,
        # return a mixture as in (8.24).
        if not newAgent:
            return sampleLaw
        colConc = self._idp['colConc']
        return (
            (colConc / (colConc + numRows)) * self.prior(mapping) +
            (numRows / (colConc + numRows)) * sampleLaw
        )

    def agentLaw(self, rowNum=None, mapping=1, bandwidth=None):
        r'''A syntactic alternative for `law` in the case that a single
        agent is being considered. Using notation from the documentation
        for `law`, if $m$ is the agent's index, then this method returns
        the law of a random variable of the form $\Theta = 
        f(\theta_{m0}, \theta_{m1}, \ldots, \theta_{m, L - 1})$.

        **Arguments:**

        * **rowNum** (`int` or `None`): The index of the agent to
            consider. If `None`, then a new, unobserved agent is
            considered. Passing this argument an integer that is
            negative or greater than or equal to the length of
            `rowCounts` is equivalent to passing it `None`.
        * **mapping** (`func` or `int`): The function $f$ in the above
            description. Passing this argument an integer $\ell$ is
            equivalent to using the function $f(\theta_{m0},
            \theta_{m1}, \ldots, \theta_{m, L - 1}) = \theta_{m\ell}$.
        * **bandwidth** (`float` or `None`): Passed directly to `law`.
            See that documentation for further details.

        **Returns:**

        * `Measure`: The law of $\Theta$.

        '''
        func = (
            mapping if callable(mapping) else lambda rowDist: rowDist[mapping]
        )
        rowNum = -1 if rowNum is None else rowNum

        if 0 <= rowNum < len(self._rowCounts):
            return self.law(
                lambda rowDists: func(rowDists[rowNum]),
                False,
                bandwidth
            )
        return self.law(func, True, bandwidth)

class WeightedSim:
    r'''A weighted simulation for an `NDPModel` instance.

    In the Paper, a weighted simulation is determined by the pair $(t,
    \boldsymbol{\theta}_M^*)$ and has total weight $V$.

    '''

    def __init__(self, idpModel):
        r'''Constructs a weighted simulation for the given NDP model by
        computing the `rowWeights`, `rowDistSims`, and `totalWeight`
        properties.

        **Arguments:**

        * **ndpModel** (`NDPModel`): The NDP model for which the
            weighted simulation is being created.

        **Raises:**

        * **ValueError**: If the computed value for the `totalWeight`
            property is so small or so large that its square evaluates
            to `0.0` or `np.inf`. (The square is needed in the
            `NDPModel.ess` property).

        '''
        self._rowWeights = []
        self._rowDistSims = []
        self._logWeight = 0 # the natural logarithm of the total weight

        # If $L$ is the number of states in the NDP, then the case
        # $L = 2$ is separated from the case $2 < L < \infty$ to speed
        # up code execution.
        baseMeas = np.array(idpModel.baseMeas)
        numStates = len(baseMeas)

        # expand properties if row counts have been added
        for m, counts in enumerate(idpModel.rowCounts):

            # build `rowWeights[m]`
            self._rowWeights.append([])
            for i in range(m):
                if numStates == 2:
                    self._rowWeights[m].append(
                        self._rowDistSims[i][0] ** counts[0] *
                        self._rowDistSims[i][1] ** counts[1]
                    )
                else:
                    weight = self._rowDistSims[i][0] ** counts[0]
                    for state in range(1, numStates):
                        weight *= self._rowDistSims[i][state] ** counts[state]
                    self._rowWeights[m].append(weight)
            self._rowWeights[m].append(
                idpModel.colConc * idpModel.likelihoods[m]
            )

            # build `rowDistSims[m]`
            choice = choices(range(m + 1), weights=self._rowWeights[m])[0]
            if choice < m:
                self._rowDistSims.append(self._rowDistSims[choice])
            else:
                if numStates == 2:
                    rowDistSim = beta.rvs(
                        idpModel.rowConc * baseMeas[1] + counts[1],
                        idpModel.rowConc * baseMeas[0] + counts[0]
                    )
                    self._rowDistSims.append([1 - rowDistSim, rowDistSim])
                else:
                    self._rowDistSims.append(
                        # vector sum fails for lists, works for ndarray
                        dirichlet.rvs(
                            idpModel.rowConc * baseMeas + counts
                        )[0]
                    )

            # build `totalWeight`
            #
            # The m-th iteration of the `for` loop multiplies the
            # previous value of `totalWeight` by the (m + 1)-th factor
            # in the definition of $V$ in Section 6.1 of the Paper.
            self._logWeight += (
                idpModel.logScale +
                np.log(m + 1) -
                np.log(idpModel.colConc + m) +
                np.log(sum(self._rowWeights[m]))
            )

        # check that total weight is not out of bounds
        if np.exp(2 * self._logWeight) in (0, np.inf):
            adj = -self._logWeight/len(idpModel.rowCounts)
            raise ValueError(
                'Simulation weight out of bounds.\n' +
                f'Try adjusting log scale factor by {adj}'
            )

    @property
    def rowWeights(self):
        '''`tuple`(`tuple`(`float`)): The triangular array of row
        weights for the weighted simulation. Denoted by $t$ in Section
        6.1 of the Paper.
        '''
        return tuple(tuple(row) for row in self._rowWeights)

    @property
    def rowDistSims(self):
        r'''`tuple`(`tuple`(`float`)): A tuple of simulated row
        distributions. If `row` is $m - 1$ and `state` is $\ell$, then
        `rowDistSims`[`row`][`state`] is denoted by $\theta_{m\ell}^*
        = u_m^*(\\{\ell\\})$ in Section 6.1 of the Paper.
        '''
        return tuple(self._rowDistSims)

    @property
    def totalWeight(self):
        '''`float`: The total weight of the simulation. Denoted by $V$
        in Section 6.1 of the Paper.
        '''
        return np.exp(self._logWeight)
