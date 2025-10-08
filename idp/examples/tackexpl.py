r'''Imagine a box of 320 thumbtacks. We flick each thumbtack 9 times. If
it lands point up, we call it a success. Point down is a failure.
Because of the imperfections, each thumbtack has its own probability of
success. The number of successes in 9 flicks for all 320 thumbtacks are:

>   7, 4, 6, 6, 6, 6, 8, 6, 5, 8, 6, 3, 3, 7, 8, 4, 5, 5, 7, 8, 5, 7, 6,
    5, 3, 2, 7, 7, 9, 6, 4, 6, 4, 7, 3, 7, 6, 6, 6, 5, 6, 6, 5, 6, 5, 6,
    7, 9, 9, 5, 6, 4, 6, 4, 7, 6, 8, 7, 7, 2, 7, 7, 4, 6, 2, 4, 7, 7, 2,
    3, 4, 4, 4, 6, 8, 8, 5, 6, 6, 6, 5, 3, 8, 6, 5, 8, 6, 6, 3, 5, 8, 5,
    5, 5, 5, 6, 3, 6, 8, 6, 6, 6, 8, 5, 6, 4, 6, 8, 7, 8, 9, 4, 4, 4, 4,
    6, 7, 1, 5, 6, 7, 2, 3, 4, 7, 5, 6, 5, 2, 7, 8, 6, 5, 8, 4, 8, 3, 8,
    6, 4, 7, 7, 4, 5, 2, 3, 7, 7, 4, 5, 2, 3, 7, 4, 6, 8, 6, 4, 6, 2, 4,
    4, 7, 7, 6, 6, 6, 8, 7, 4, 4, 8, 9, 4, 4, 3, 6, 7, 7, 5, 5, 8, 5, 5,
    5, 6, 9, 1, 7, 3, 3, 5, 7, 7, 6, 8, 8, 8, 8, 7, 5, 8, 7, 8, 5, 5, 8,
    8, 7, 4, 6, 5, 9, 8, 6, 8, 9, 9, 8, 8, 9, 5, 8, 6, 3, 5, 9, 8, 8, 7,
    6, 8, 5, 9, 7, 6, 5, 8, 5, 8, 4, 8, 8, 7, 7, 5, 4, 2, 4, 5, 9, 8, 8,
    5, 7, 7, 2, 6, 2, 7, 6, 5, 4, 4, 6, 9, 3, 9, 4, 4, 1, 7, 4, 4, 5, 9,
    4, 7, 7, 8, 4, 6, 7, 8, 7, 4, 3, 5, 7, 7, 4, 4, 6, 4, 4, 2, 9, 9, 8,
    6, 8, 8, 4, 5, 7, 5, 4, 6, 8, 7, 6, 6, 8, 6, 9, 6, 7, 6, 6, 6

This is exactly the data considered in
[Liu (1996)](https://doi.org/10.1214/aos/1032526949). (This data
originally came from an experiment by Beckett and Diaconis in 1994. In
the original experiment, there were not 320 thumbtacks. Rather, there
were 16 thumbtacks, 2 flickers, and 10 surfaces.)

To study this situation, we first create the model and perform 10,000
simulations.

```python
>>> import json
>>> with open('tacks.json', 'r') as f:
...    data = json.load(f)
...    rowCounts = [[9 - successes, successes] for successes in data]
>>> model = ndpModel(1, 2, 2, rowCounts)
>>> model.addSims(10000) # took ~4m on a Macbook
>>> model.ess
243.581441084223
```

Here, we use a row concentration parameter of $2$ and a base measure
that assigns probability $1/2$ to both $0$ and $1$. This matches the
distribution used by Liu. What we call the column concentration
parameter is denoted by $c$ in Liu's analysis. Liu considers the cases
$c = 0.1$, $c = 1$, $c = 5$, and $c = 10$. Here, we are taking $c = 1$.

We also checked our effective sample size, which is a good habit
whenever we add new weighted simulations. For comparison, Liu reported
an effective sample size of 227 in the case $c = 1$.

We now look at the approximate density of the posterior probability of
success.

```python
>>> law = model.agentLaw()
>>> plt.plot(t, law.pdf(t))
>>> law.bandwidth
0.10510725412303899
>>> plt.show()
```
![image](images/tacks_pdf_1.svg)

This graph look about the same as Liu's, but with minor differences. It
is difficult, though, to make a direct comparison. Liu also uses
Gaussian kernel smoothing, but does not provide any details about it.
For example, we do not know the bandwidth used to produce Liu's graphs.

We now try $c = 10$.

```python
>>> model = ndpModel(10, 2, 2, rowCounts)
>>> model.addSims(10000)
>>> model.ess
387.47462091947494
>>> plt.plot(t, model.newRowDistPDF(t))
>>> law.bandwidth
0.09578841832899217
>>> plt.show()
```
![image](images/tacks_pdf_10.svg)

This graph also looks similar to Liu's. For comparison, Liu reported an
effective sample size of 300 in this case.

'''

# pylint: disable=wrong-import-position
# pylint: disable=wrong-import-order

if __name__ == '__main__':
    from idp import idpModel
    import numpy as np
    import matplotlib.pyplot as plt
    t = np.linspace(0, 1, 200)

    EXTENSIONS = ['png', 'jpg', 'pdf', 'svg', 'eps', 'ps']

    import json
    # pylint: disable-next=unspecified-encoding
    with open('tackexpl.json', 'r') as f: 
        data = json.load(f)
        rowCounts = [[9 - successes, successes] for successes in data]
    model = idpModel(1, 2, 2, rowCounts)
    model.addSims(10000)
    # %time
    # CPU times: user 3min 32s, sys: 18 s, total: 3min 50s
    # Wall time: 3min 57s
    print(model.ess) # 243.581441084223

    law = model.agentLaw()
    # %time
    # CPU times: user 4.5 s, sys: 217 ms, total: 4.72 s
    # Wall time: 4.8 s
    plt.plot(t, law.pdf(t))
    # %time
    # CPU times: user 2.66 s, sys: 1.64 s, total: 4.3 s
    # Wall time: 2.48 s
    print(law.bandwidth) # 0.10510725412303899
    for ext in EXTENSIONS:
        plt.savefig('tacks_pdf_1.' + ext)
    plt.show()
    plt.close()

    # restart IPython before running the rest

    model = idpModel(10, 2, 2, rowCounts)
    model.addSims(10000)
    # %time
    # CPU times: user 3min 39s, sys: 40.2 s, total: 4min 19s
    # Wall time: 5min 44s # probably so high because I didn't restart
    print(model.ess) # 387.47462091947494
    law = model.agentLaw()
    # %time
    # CPU times: user 3.22 s, sys: 812 ms, total: 4.03 s
    # Wall time: 4.71 s
    plt.plot(t, law.pdf(t))
    # %time
    # CPU times: user 3.98 s, sys: 52.9 ms, total: 4.04 s
    # Wall time: 2.32 s
    print(law.bandwidth) # 0.09578841832899217
    for ext in EXTENSIONS:
        plt.savefig('tacks_pdf_10.' + ext)
    plt.show()
    plt.close()
