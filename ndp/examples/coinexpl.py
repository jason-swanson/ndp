r'''Imagine a pressed penny machine, like those found in museums or
tourist attractions. For a fee, the machine presses a penny into a
commemorative souvenir. Now imagine the machine is broken, so that it
mangles all the pennies we feed it. Each pressed penny it creates is
mangled in its own way. Each has its own probability of landing on heads
when flipped. In this situation, the agents are the pennies and the
actions are the heads and tails that they produce.

Now suppose we create seven mangled pennies and flip each one 5 times,
giving us the following results:

| Coin # | 1st Flip | 2nd Flip | 3rd Flip | 4th Flip | 5th Flip |
|:------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| 0      | H        | H        | H        | H        | T        |
| 1      | H        | T        | H        | H        | H        |
| 2      | T        | H        | H        | T        | H        |
| 3      | H        | H        | T        | H        | H        |
| 4      | T        | T        | T        | H        | T        |
| 5      | T        | H        | H        | H        | H        |
| 6      | H        | T        | T        | H        | H        |

Of the 35 flips, 23 of them (or about 65.7%) were heads. In fact, 6 of
the 7 coins landed mostly on heads. The machine clearly seems
predisposed to creating pennies that are biased towards heads.

Coin #4, though, produced only one head. Is this coin different from the
others and actually biased toward tails? Or was it mere chance that its
flips turned out that way? For instance, suppose all 7 coins had a 60%
chance of landing on heads. In that case, there would still be a 43%
chance that at least one of them would produce four tails. How should we
balance these competing explanations and arrive at some concrete
probabilities?

One way to answer this is to model the situation with an NDP. We begin
by using the `ndpModel` function to create an instance of the
`ndp.tools.NDPModel` class. We then run 10,000 simulations.

```python
>>> from ndp import ndpModel
>>> model = ndpModel(1, 1, 2, data=[
...     [1, 1, 1, 1, 0],
...     [1, 0, 1, 1, 1],
...     [0, 1, 1, 0, 1],
...     [1, 1, 0, 1, 1],
...     [0, 0, 0, 1, 0],
...     [0, 1, 1, 1, 1],
...     [1, 0, 0, 1, 1]
... ])
>>> model.addSims(10000)
```

## New coins

With these simulations in place, the first question we ask is the
following. If we were to get a new coin from this machine, how would we
expect it to behave? The new coin would have some random probability of
heads, $\theta$. We can get the law (or distribution) of $\theta$ with
the `ndp.tools.NDPModel.agentLaw` method.

```python
>>> law = model.agentLaw()
```

The probability that a new coin will land on heads on its first flip is
simply the mean of $\theta$.

```python
>>> law.mean
0.6330354619750681
```

The following will plot the cumulative distribution function of
$\theta$:

```python
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> t = np.linspace(0, 1, 200)
>>> plt.plot(t, law.cdf(t))
>>> plt.show()
```
![image](images/coins_cdf.svg)

The random variable $\theta$ has a discrete component, so it does not
have a probability density function. But we can plot an approximate
density as follows:

```python
>>> plt.plot(t, law.pdf(t))
>>> plt.show()
```
![image](images/coins_pdf.svg)

The approximate density is computed by smoothing out the discrete point
masses. The degree of smoothing is determined by a parameter called the
"bandwidth". (See `ndp.tools.NDPModel.law` for details.) This bandwidth
is computed automatically. We can view the computed bandwidth, and we
can also set it manually.

```python
>>> law.bandwidth
0.1186811817471998
>>> law = model.agentLaw(bandwidth=0.001)
>>> plt.plot(t, law.pdf(t))
>>> plt.show()
```
![image](images/coins_pdf_coarse.svg)

## Observed coins

Everything we did above can also be done for the seven coins that we
have already observed. Consider, for instance, Coin #4. It has an
unknown probability of heads that we might denote by $\theta_4$. We can
view the distribution of $\theta_4$, given the flips we have already
observed.

```python
>>> law = model.agentLaw(4)
>>> plt.plot(t, law.pdf(t))
>>> plt.show()
```
![image](images/coins_pdf_4.svg)

We can also find the probability that Coin #4, if flipped again, will
land on heads:

```python
>>> law.mean
0.4610399210826454
```

Or we can find the probability that Coin #4 is actually biased toward
tails.

```python
>>> law.cdf(0.5)
0.4813587213542915
```

## Effective sample size

The above probabilities and graphs are not calculated by direct
simulation, but by using [importance
sampling](https://en.wikipedia.org/wiki/Importance_sampling). Although
the sample size is 10,000, the "effective sample size" is lower:

```python
>>> model.ess
6067.021983020713
```

What this means is that the accuracy of our calculations is about the
same as if we did 6,067 direct simulations.

'''

# pylint: disable=wrong-import-position
# pylint: disable=wrong-import-order

EXTENSIONS = ['png', 'jpg', 'pdf', 'svg', 'eps', 'ps']

if __name__ == '__main__':
    from ndp import ndpModel
    model = ndpModel(1, 1, 2, data=[
        [1, 1, 1, 1, 0],
        [1, 0, 1, 1, 1],
        [0, 1, 1, 0, 1],
        [1, 1, 0, 1, 1],
        [0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1],
        [1, 0, 0, 1, 1]
    ])
    model.addSims(10000)

    law = model.agentLaw()

    print(law.mean)

    import numpy as np
    import matplotlib.pyplot as plt
    t = np.linspace(0, 1, 200)
    plt.plot(t, law.cdf(t))
    for ext in EXTENSIONS:
        plt.savefig('coins_cdf.' + ext)
    plt.show()
    plt.close()

    plt.plot(t, law.pdf(t))
    for ext in EXTENSIONS:
        plt.savefig('coins_pdf.' + ext)
    plt.show()
    plt.close()

    print(law.bandwidth)
    law = model.agentLaw(bandwidth=0.001)
    plt.plot(t, law.pdf(t))
    for ext in EXTENSIONS:
        plt.savefig('coins_pdf_coarse.' + ext)
    plt.show()
    plt.close()

    law = model.agentLaw(4)
    plt.plot(t, law.pdf(t))
    for ext in EXTENSIONS:
        plt.savefig('coins_pdf_4.' + ext)
    plt.show()
    plt.close()

    print(law.mean)

    print(law.cdf(0.5))

    print(model.ess)
