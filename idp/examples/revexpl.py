r'''[Liu (1996)](https://doi.org/10.1214/aos/1032526949) only covers
agents with two possible actions, such as coins and thumbtacks. The NDP,
though, can handle agents whose range of possible actions is arbitrary.

Imagine, then, that we discover a seller on Amazon that has 50 products.
Their products have an average rating of 2.4 stars out of 5. Some
products have almost 100 ratings, while others have only a few. On
average, the products have 23 ratings each. In this case, the agents are
the products and the actions are the ratings that each product earns.
Each individual rating must be a whole number of stars between 1 and 5,
inclusive. Hence, each action has 5 possible outcomes.

The Google Sheets file, [`revexpl.csv`](https://docs.google.com/
spreadsheets/d/1BharWtVT43-BlpGYlBer7u3UleAihOgZK2qMJArD7eU/
edit?usp=sharing), contains all the data for this example. This file has
51 rows (one header row and one row for each product). The rows are
ordered so that the products with the most reviews are on top. The
format of the file looks like this:

| 1 star | 2 stars | 3 stars | 4 stars | 5 stars | # reviews | average |
|:------:|:-------:|:-------:|:-------:|:-------:|:---------:|:-------:|
| 9      | 25      | 15      | 41      | 0       | 90        | 2.98    |
| 21     | 28      | 18      | 1       | 3       | 71        | 2.11    |
| 16     | 11      | 21      | 11      | 0       | 59        | 2.46    |
| 3      | 9       | 37      | 0       | 3       | 52        | 2.83    |
| ...    | ...     | ...     | ...     | ...     | ...       | ...     |
| 2      | 1       | 0       | 0       | 0       | 3         | 1.33    |
| 0      | 0       | 2       | 0       | 0       | 2         | 3       |
| 0      | 1       | 0       | 1       | 0       | 2         | 3       |
| 0      | 0       | 1       | 1       | 0       | 2         | 3.5     |

After downloading the file to the current working directory, we import
the data and create the NDP model.

```python
>>> import csv
>>> with open('reviews.csv') as f:
...     sheet = list(csv.reader(f))
...     rowCounts = [
...         [int(val) for val in row[:5]] for row in sheet[1:]
...     ]
>>> model = ndpModel(1, 5, 5, rowCounts)
```

In this example, unlike the previous ones, when we try to create a
weighted simulation, we get an error.

```python
>>> model.addSims(1)
Traceback (most recent call last):
  File "ndp.py", line 395, in addSims
    self._weightedSims += [WeightedSim(self) for _ in range(num)]
  File "ndp.py", line 585, in __init__
    self.updateRowCounts()
  File "ndp.py", line 694, in updateRowCounts
    raise ValueError(
ValueError: Simulation weight out of bounds.
Try adjusting log scale factor by 28.773115195642152
```

The reason for this error is that the simulation weight was too small
for the code to handle. The error message recommends that we correct for
this by adjusting the log scale factor. Following the recommendation, we
are able to create simulations.

```python
>>> model.logScale += 28.8
>>> model.addSims(10000) # took ~17s
>>> model.ess
7.841271686848992
```

But now we have a new problem. The effective sample size is only
about 8. To fix this, we can adjust the column concentration.

```python
>>> model = ndpModel(10, 5, 5, rowCounts)
>>> model.logScale += 28.8
>>> model.addSims(10000)
>>> model.ess
43.80517839891516
```

This is a much better number, so we go ahead and generate even more
simulations.

```python
>>> model.addSims(90000) # took ~2.5m
>>> model.ess
560.6096774185303
```

We now have 100,000 simulations and an effective sample size of
about 561. We are ready to look at the statistical properties of this
model.

## New products

Let us first consider a hypothetical new product from this seller. This
product can be characterized by the vector $\theta = (\theta_1, \ldots,
\theta_5)$, where $\theta_\ell$ is the unknown probability that the
product will receive an $\ell$-star rating. The long-term average rating
of this product over many reviews will be $\Theta = \sum_{{\ell = 1}}^5
\ell \cdot \theta_\ell$. This unknown quantity $\Theta$ is a measure of
the quality of the product. We can get the law of $\Theta$ as follows.

```python
>>> def avg(theta):
...    return np.average(range(1, 6), weights=theta)
>>> law = model.agentLaw(None, avg) # took ~40s
>>> law.mean
2.535621660810854
>>> t = np.linspace(1, 5, 200)
>>> plt.plot(t, law.pdf(t))
>>> plt.show()
```
![image](images/reviews_pdf.svg)

The average ratings of future products is $E[\Theta]$. According to the
above, this is about 2.5. But the graph shows a bimodal distribution. We
expect, then, that future products will tend to cluster around 2 and 3
stars.

## Products with few reviews

We can do the same for products that already have ratings. Consider, for
instance, the product that occupies the last row of `reviews.csv`.

```python
>>> model.rowCounts[49]
(0, 0, 1, 1, 0)
```

This product has a 3.5-star average rating, but only 2 reviews. To see
the effect of these 2 reviews on the expected long-term rating, we
compute as we did above.

```python
>>> law = model.agentLaw(49, avg)
>>> law.mean
2.8256663976604957
>>> plt.plot(t, law.pdf(t))
>>> plt.show()
```
![image](images/reviews_pdf_49.svg)

Based on these 2 reviews, the expected long-term rating of this product
is about 2.83.

Similarly, we can consider the product with row index 25.

```python
>>> model.rowCounts[25]
(0, 3, 0, 6, 7)
>>> np.average(range(1, 6), weights=_)
4.0625
```

This product's average rating is higher than 4, but it only has 16
reviews. As above, we can see the effect of this as follows.

```python
>>> law = model.agentLaw(25, avg)
>>> law.mean
3.7975047283049452
>>> plt.plot(t, law.pdf(t))
>>> plt.show()
```
![image](images/reviews_pdf_25.svg)

This time the effect is more pronounced, as it should be. Based on these
16 reviews, the model expects this product's long-term rating to be
about 3.8.

'''

# pylint: disable=wrong-import-position
# pylint: disable=wrong-import-order
# pylint: disable=unused-import

if __name__ == '__main__':
    from idp import idpModel
    import numpy as np
    import matplotlib.pyplot as plt
    import json

    t = np.linspace(0, 1, 200)
    EXTENSIONS = ['png', 'jpg', 'pdf', 'svg', 'eps', 'ps']

    import csv
    # pylint: disable-next=unspecified-encoding
    with open('revexpl.csv') as f:
        sheet = list(csv.reader(f))
        rowCounts = [[int(val) for val in row[:5]] for row in sheet[1:]]
    model = idpModel(1, 5, 5, rowCounts)
    # model.addSims(1)
    # Traceback (most recent call last):
    #   File "tools.py", line 243, in addSims
    #     self._weightedSims += [WeightedSim(self) for _ in range(num)]
    #   File "tools.py", line 497, in __init__
    #     raise ValueError(
    # ValueError: Simulation weight out of bounds.
    # Try adjusting scale factor by 28.773115195642152

    model.logScale += 28.8
    model.addSims(10000)
    # %time
    # CPU times: user 16.7 s, sys: 146 ms, total: 16.8 s
    # Wall time: 16.8 s
    print(model.ess)
    # 2.5409335959370263
    # 21.227071192057817
    # 21.044935495378308
    # 22.36745196164331
    # 49.11110068697961
    # 7.841271686848992
    # 54.227868394484695
    # 6.954081687350993
    # 6.440874720710102
    # 2.7909339160305002
    # 17.151479680397365
    # 4.494145301503032
    # 14.581910032760046
    # 11.812530806848757
    # 3.456711897441024
    # 19.77470598816864
    # 7.94407670553619
    # 19.601866111755296
    # 9.628083583224756
    # 6.645118088548286
    # 12.507111958452892
    # 6.440533351406849
    # 12.529863701752635
    # 18.796623754300906
    # 2.068250099383208
    # 9.738680416899074
    # 45.81282631693935
    # 13.236790340851837
    # 6.537081348110968
    # 3.17520586210496
    # mean: 14.68267062279358
    # sample SD: 13.4323342552673

    model = idpModel(10, 5, 5, rowCounts)
    model.logScale += 28.8
    model.addSims(10000)
    # %time
    # CPU times: user 17.4 s, sys: 126 ms, total: 17.5 s
    # Wall time: 17.5 s
    print(model.ess)
    # 128.58687989662954
    # 31.915097166973442
    # 6.167795154813725
    # 64.0162158909806
    # 37.805775970591164
    # 13.138925903079
    # 16.7121384524134
    # 19.7981936942665
    # 30.47518018732239
    # 28.305327225237555
    # 17.35729767996366
    # 63.10130009074075
    # 12.958420040439183
    # 58.59518583381152
    # 58.983983667717524
    # 52.81392853687226
    # 62.47744733340391
    # 43.80517839891516
    # 30.30478971829668
    # 55.783094062365564
    # 6.371454457330428
    # 62.144631803774175
    # 26.665474934855684
    # 2.037172209252143
    # 4.299384147855736
    # 25.945301931096534
    # 36.93280328240885
    # 12.686585280294144
    # 60.41034197709455
    # 125.25990402977587
    # mean: 39.8618402986191
    # sample SD: 31.1849103309742

    model.addSims(90000)
    # %time
    # CPU times: user 2min 35s, sys: 1.1 s, total: 2min 36s
    # Wall time: 2min 36s
    print(model.ess) # 560.6096774185303

    def avg(theta): # pylint: disable=missing-function-docstring
        return np.average(range(1, 6), weights=theta)
    law = model.agentLaw(None, avg)
    # %time
    # CPU times: user 38 s, sys: 187 ms, total: 38.2 s
    # Wall time: 38.2 s
    print(law.mean) # 2.535621660810854
    t = np.linspace(1, 5, 200)
    plt.plot(t, law.pdf(t))
    # %time
    # CPU times: user 5.57 s, sys: 34.9 ms, total: 5.6 s
    # Wall time: 3.57 s
    for ext in EXTENSIONS:
        plt.savefig('reviews_pdf.' + ext)
    plt.show()
    plt.close()

    law = model.agentLaw(49, avg)
    print(law.mean) # 2.8256663976604957
    plt.plot(t, law.pdf(t))
    for ext in EXTENSIONS:
        plt.savefig('reviews_pdf_49.' + ext)
    plt.show()
    plt.close()

    law = model.agentLaw(25, avg)
    print(law.mean) # 3.7975047283049452
    plt.plot(t, law.pdf(t))
    for ext in EXTENSIONS:
        plt.savefig('reviews_pdf_25.' + ext)
    plt.show()
    plt.close()
