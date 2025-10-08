r'''Consider a single-player video game in which an individual player
tries to score as many points as possible before the game ends. A group
of 10 friends get together and play this game. Each friend plays the
game a different number of times. In this case, the agents are the
players and the actions are the scores they earn each time they play.

The 10 friends all have their own user names that they use when playing
the game. The user names are Asparagus Soda, Goat Radish, Potato Log,
Pumpkins, Running Stardust, Sweet Rolls, The Matrix, The Pianist Spider,
The Thing, and Vertigo Gal. We will consider two different scenarios for
this example. The data for these examples was generated using the
`ndp.lboard` submodule.

## Players with common scores

In our first scenario, the 10 friends generate the following scores:

* **Pumpkins:**             12, 21, 25, 25, 26, 27, 30, 33, 34, 34, 36,
                            42, 44, 44, 48, 55, 67, 69
* **Potato Log:**           18, 21, 21, 22, 23, 25, 29, 29, 32, 33, 47,
                            53, 54, 56, 57, 65, 75
* **The Thing:**            10, 16, 16, 19, 19, 25, 25, 26, 29, 32, 35,
                            37, 42, 44, 59, 60
* **Running Stardust:**     23, 38, 62, 71, 138, 149, 151
* **Sweet Rolls:**          15, 23, 56, 71, 98, 130
* **Vertigo Gal:**          10, 30, 40, 56, 87, 92
* **Asparagus Soda:**       17, 43, 55
* **The Matrix:**           11, 15
* **Goat Radish:**          38
* **The Pianist Spider:**   32

The Google Sheets file, [`gameexpl1.csv`](https://docs.google.com/
spreadsheets/d/1-mUExDE7FG_XfcjM4fHXqPT1-lmnB0weX1JVAy1ItYk/
edit?usp=sharing), contains all the data for this example. This file has
11 rows (one header row and one row for each player). The rows are
ordered so that the players with the most attempts are on top. After
downloading the file to the current working directory, we import the
data.

```python
>>> with open('gameexpl1.csv') as f:
>>>     sheet = list(csv.reader(f))
>>>     data = [
>>>         [int(val) for val in row[7:] if val != '']
>>>         for row in sheet[1:]
>>>     ]
```

To get an overview of the data, we can place the 10 players in a
leaderboard, ranked by their high score.

| rank | name               | hi score | avg score | # games |
|:----:|--------------------|:--------:|:---------:|:-------:|
|   1  | Running Stardust   |    151   |     90    |    7    |
|   2  | Sweet Rolls        |    130   |     66    |    6    |
|   3  | Vertigo Gal        |    92    |     52    |    6    |
|   4  | Potato Log         |    75    |     39    |    17   |
|   5  | Pumpkins           |    69    |     37    |    18   |
|   6  | The Thing          |    60    |     31    |    16   |
|   7  | Asparagus Soda     |    55    |     38    |    3    |
|   8  | Goat Radish        |    38    |     38    |    1    |
|   9  | The Pianist Spider |    32    |     32    |    1    |
|  10  | The Matrix         |    15    |     13    |    2    |

We are interested in ranking the players, not by their high score, but
by their skill at the game. To be precise, we imagine that each player
plays the game indefinitely and take their skill level to be measured by
their long-term average score. We will use an NDP model to estimate this
long-term average.

Our first task is to choose a base measure for our model. The base
measure represents our prior distribution on the score of a 
player. Because it a video game, a uniform base measure would not make
sense. But because it is only a hypothetical video game, there is no
particular base measure that would be most appropriate. Somewhat
arbitrarily, then, we will use the `ndp.tools.gamer` distribution as our
base measure.

The gamer distribution has shape parameters $r$ and $a$, and a scale
parameter $c$. We choose $r = 7/3$ so that our distribution has a 
law decay that approximately matches the distribution of scores on the
[global Tetris leaderboard](https://kirjava.xyz/
tetris-leaderboard/?board=NTSC&platform=Console&proof=Video). We choose
$c = 28$ so that our distribution has a mean of about $50$. Our choice
of $a = 3$ is arbitrary and meant to reflect a game in which the player
has three "lives."

Finally, since the gamer distribution is continuous, we must discretize
it so it can be used with the NDP model. We will use a discrete
approximation that is supported on the integers between 0 and 499.

```python
>>> from ndp.tools import gamer
>>> g = gamer(7/3, 3, scale=28)
>>> baseMeas = [g.cdf(n + 0.5) - g.cdf(n - 0.5) for n in range(499)]
>>> baseMeas.append(g.sf(499.5))
```

Having built the base measure, we now build our NDP model, generate
simulations, and check the effective sample size. As in the Amazon
reviews example, we need to adjust the model's log scale factor in order
to build the model without error.

```python
>>> model = ndpModel(1, 1, baseMeas, data=data)
>>> model.logScale += 42
>>> model.addSims(40000) # took ~3m
>>> model.ess
326.717352110184
```

We now build the list `laws` so that `laws[m]` is the distribution of
the long-term average score of player `m`.

```python
>>> def avg(theta):
>>>     return np.average(range(500), weights=theta)
>>> laws = [model.agentLaw(m, avg) for m in range(10)] # took ~10s
```

Using `laws`, we can view the expected long-term average score of each
player, according the NDP model.

```python
>>> for m in range(10):
>>>     print(f'{{sheet[m + 1][3]}}: {{laws[m].mean}}')
Pumpkins: 37.8769311756713
Potato Log: 39.475686400565
The Thing: 32.3193302196986
Running Stardust: 79.6472672018615
Sweet Rolls: 54.5136744371402
Vertigo Gal: 52.0993770879504
Asparagus Soda: 40.2712946915477
The Matrix: 42.5543167395948
Goat Radish: 71.3566832439191
The Pianist Spider: 37.4693206147343
```

If we round these expected values to the nearest integer and add them to
our leaderboard table, we can better visualize their relationship to the
data.

| rank | name               | hi score | avg score | NDP avg | # games |
|:----:|--------------------|:--------:|:---------:|:-------:|:-------:|
|   1  | Running Stardust   |    151   |     90    |    80   |    7    |
|   2  | Sweet Rolls        |    130   |     66    |    55   |    6    |
|   3  | Vertigo Gal        |    92    |     52    |    52   |    6    |
|   4  | Potato Log         |    75    |     39    |    39   |    17   |
|   5  | Pumpkins           |    69    |     37    |    38   |    18   |
|   6  | The Thing          |    60    |     31    |    32   |    16   |
|   7  | Asparagus Soda     |    55    |     38    |    40   |    3    |
|   8  | Goat Radish        |    38    |     38    |    71   |    1    |
|   9  | The Pianist Spider |    32    |     32    |    37   |    1    |
|  10  | The Matrix         |    15    |     13    |    43   |    2    |

Looking at this table, we can see at least two players whose numbers
seem unusual. The first is Goat Radish. They played only one game and
scored a 38, which is a relatively low score compared to the rest of the
group. And yet the NDP model has given them an expected long-term
average score of 71. Not only is this counterintuitive, it's also
inconsistent with how the model treated The Pianist Spider.

The reason for this behavior is that there is only one other player that
managed to score exactly 38 in one of their games: Running Stardust. So
from the model's perspective, there's a reasonable chance that Goat
Radish and Running Stardust will have similar scores in the long run.

Our human intuition is able to dismiss this line of reasoning because we
know, for instance, that there is very little difference between a score
of 38 and 39. Had Goat Radish scored a 39 instead, our predictions
should not change much. But we only know this because we are viewing the
positive real numbers as more than just a set. We are viewing them as a
totally ordered set with the Euclidean metric. The NDP model is not
designed to utilize these properties of the state space. From its
perspective, the number "38" is just a label. It's nothing more than the
name of a particular element of the state space.

We see similar behavior in the model's forecast for The Matrix, who
scored an 11 and a 15 in their two games. No one else scored an 11, but
exactly one other player managed to score exactly 15, and that was Sweet
Rolls. Just as with Goat Radish, this causes the model to generate an
unintuitively high value for The Matrix's long-term average score.

We can test this explanation by changing Goat Radish's 38 to a 39, and
The Matrix's 15 to a 14, and then rerunning the model.

```python
>>> data[8][0] = 39
>>> data[7][1] = 14
>>> model = ndpModel(1, 1, baseMeas, data=data)
>>> model.logScale += 42
>>> model.addSims(40000)
>>> model.ess
22.32478257656367
```

This time, our simulations produced an unexpectedly low effective sample
size. This can happen because of one or two outlier simulations whose
weight is massive compared to the others. We should not proceed until we
deal with this. We could deal with it by simply generating more
simulations. To save time, though, we will deal with it by deleting
these outlier simulations.

```python
>>> model.setEss(200)
2
>>> model.ess
1098.9123742787074
```

Out of 40,000 simulations, we deleted the 2 heaviest simulations and
this gave us an effective sample size of over 1000. We now generate the
expected long-term averages.

```python
>>> laws = [model.agentLaw(m, avg) for m in range(10)]
>>> for m in range(10):
>>>     print(f"{{sheet[m + 1][3]}}: {{laws[m].mean}}")
Pumpkins: 38.029840087762494
Potato Log: 39.450632512977556
The Thing: 31.92326020525782
Running Stardust: 84.41440312210776
Sweet Rolls: 62.72904055941699
Vertigo Gal: 51.208412372436186
Asparagus Soda: 39.92390947459972
The Matrix: 28.21322997473585
Goat Radish: 43.52445544679143
The Pianist Spider: 37.37972848216658
```

Updating our table, we have the following.

| rank | name               | hi score | avg score | NDP avg | # games |
|:----:|--------------------|:--------:|:---------:|:-------:|:-------:|
|   1  | Running Stardust   |    151   |     90    |    84   |    7    |
|   2  | Sweet Rolls        |    130   |     66    |    62   |    6    |
|   3  | Vertigo Gal        |    92    |     52    |    51   |    6    |
|   4  | Potato Log         |    75    |     39    |    39   |    17   |
|   5  | Pumpkins           |    69    |     37    |    38   |    18   |
|   6  | The Thing          |    60    |     31    |    31   |    16   |
|   7  | Asparagus Soda     |    55    |     38    |    39   |    3    |
|   8  | Goat Radish        |    38    |     38    |    43   |    1    |
|   9  | The Pianist Spider |    32    |     32    |    37   |    1    |
|  10  | The Matrix         |    15    |     13    |    28   |    2    |

We now see that Goat Radish and The Matrix have lower, more reasonable
long-term averages according to the model. Likewise, Running Stardust
and Sweet Rolls have slightly higher averages. In the previous run of
the model, their averages were brought down because of their
associations with Goat Radish and The Matrix.

## Players with few scores

In our second scenario, the 10 friends generate the following scores:

* **Vertigo Gal:**          45, 100, 118, 121, 125, 130, 133, 145, 161,
                            173, 173, 187, 190, 192, 193, 200, 220, 223,
                            256, 275, 314, 354, 388, 475, 524
* **Potato Log:**           4, 13, 13, 16, 19, 19, 19, 19, 23, 24, 25,
                            26, 31, 38, 41, 43, 44, 47, 51, 87
* **The Thing:**            4, 6, 9, 19, 25, 27, 28, 38, 39, 40
* **The Matrix:**           13, 15, 17, 32, 32, 61, 78
* **Running Stardust:**     21, 23, 51, 61, 65
* **Goat Radish:**          23, 25, 34, 51
* **Pumpkins:**             49, 65, 84, 117
* **Sweet Rolls:**          26, 65
* **Asparagus Soda:**       86
* **The Pianist Spider:**   62

The Google Sheets file, [`gameexpl2.csv`](https://docs.google.com/
spreadsheets/d/1H-e4Jj1_T_gnWEHKA9RssfemSWbarXl-tsbnp3_kHcQ/
edit?usp=sharing), contains all the data for this example. This file has
the same structure as the previous file. Again, we download it to the
current working directory and import the data.

```python
>>> with open('gameexpl2.csv') as f:
>>>     sheet = list(csv.reader(f))
>>>     data = [
>>>         [int(val) for val in row[7:] if val != '']
>>>         for row in sheet[1:]
>>>     ]
```

Placing the players in a ranked leaderboard, we have the following.

| rank | name               | hi score | avg score | # games |
|:----:|--------------------|:--------:|:---------:|:-------:|
|   1  | Vertigo Gal        |    475   |    207    |    25   |
|   2  | Pumpkins           |    117   |     79    |    4    |
|   3  | Potato Log         |    87    |     30    |    20   |
|   4  | Asparagus Soda     |    86    |     86    |    1    |
|   5  | The Matrix         |    78    |     35    |    7    |
|   6  | Running Stardust   |    65    |     44    |    5    |
|   6  | Sweet Rolls        |    65    |     46    |    2    |
|   8  | The Pianist Spider |    62    |     62    |    1    |
|   9  | Goat Radish        |    51    |     33    |    4    |
|  10  | The Thing          |    40    |     24    |    10   |

In this example, we focus our attention on Asparagus Soda, who it
situated at #4 on the leaderboard, but played the game only once. The
question is, do they deserve to be at #4? For example, Potato Log, who
is at #3, played the game 20 times and only managed to get a high score
of 87. Asparagus Soda almost matched that high score in a single
attempt. Intuitively, it seems clear that Asparagus Soda is the better
player and should rank higher than Potato Log.

It is less clear how Asparagus Soda compares to Pumpkins. Neither of
them made a lot of attempts, but Asparagus Soda has the higher average
score. Which one is more likely to have the higher long-term average
score? If they had a contest where they each played a single game and
the higher score wins, who should we bet on?

### Asparagus Soda vs. Potato Log

To answer these questions, we begin by creating an NDP model. We then
use the model to rank the players by their expected long-term average
score.

```python
>>> model = ndpModel(1, 1, baseMeas, data=data)
>>> model.logScale += 42
>>> model.addSims(40000)
>>> model.ess
38.995295056814456
>>> model.setEss(200)
26
>>> model.ess
206.80979715975414
>>> laws = [model.agentLaw(m, avg) for m in range(10)]
>>> for m in range(10):
>>>     print(f"{{sheet[m + 1][3]}}: {{laws[m].mean}}")
Vertigo Gal: 198.06866488887442
Potato Log: 31.183512061427916
The Thing: 26.333237442129764
The Matrix: 37.73199694794612
Running Stardust: 45.27946864625936
Goat Radish: 34.91069387324785
Pumpkins: 72.21882232954187
Sweet Rolls: 52.23247042222365
Asparagus Soda: 67.8642177970131
The Pianist Spider: 56.75360824901325
```

Rounding these outputs to the nearest integer and adding them to our
leaderboard, we have the following.

| rank | name               | hi score | avg score | NDP avg | # games |
|:----:|--------------------|:--------:|:---------:|:-------:|:-------:|
|   1  | Vertigo Gal        |    475   |    207    |   198   |    25   |
|   2  | Pumpkins           |    117   |     79    |    72   |    4    |
|   3  | Potato Log         |    87    |     30    |    31   |    20   |
|   4  | Asparagus Soda     |    86    |     86    |    67   |    1    |
|   5  | The Matrix         |    78    |     35    |    37   |    7    |
|   6  | Running Stardust   |    65    |     44    |    45   |    5    |
|   6  | Sweet Rolls        |    65    |     46    |    52   |    2    |
|   8  | The Pianist Spider |    62    |     62    |    56   |    1    |
|   9  | Goat Radish        |    51    |     33    |    34   |    4    |
|  10  | The Thing          |    40    |     24    |    26   |    10   |

The NDP model gives Asparagus Soda a much higher expected long-term
average score than Potato Log. The former is 67 and the latter is 31.
This confirms our intuition that Asparagus Soda is the better player.
But because Asparagus Soda played only one game, the model should have a
lot more uncertainty about its result of 67 than it does about its
result of 31. If we want to see this uncertainty, then we need to dive
deeper into the `laws` list.

Recall that in the `data` variable, the players are listed in descending
order according to the number of times they played. The exact order is
listed at the beginning of this example. So Vertigo Gal is Player 0,
Potato Log is Player 1, The Thing is Player 2, and so on until reaching
The Pianist Spider, who is Player 9. If $\Theta_m$ is the long-term
average score of Player $m$, then `laws[m]` is the distribution of
$\Theta_m$. The table above is showing us the means, $E[\Theta_m]$. To
visualize the uncertainty around these means we can look at the
densities. Since Asparagus Soda is Player 8, we begin by looking at the
density of $\Theta_8$.

```python
>>> t = np.linspace(1, 500, 500)
>>> plt.plot(t, laws[8].pdf(t)) # Asparagus Soda
>>> plt.show()
```
<a name='asparagus'>![image](images/asparagus.svg)</a>

We can then compare this to the density of $\Theta_1$, which is Potato
Log's long-term average score.

```python
>>> plt.plot(t, laws[1].pdf(t)) # Potato Log
>>> plt.show()
```
![image](images/potato.svg)

Visually it is clear that the model is much more certain about Potato
Log's long-term average than it is about Asparagus Soda's. How does this
uncertainty affect the probability that Asparagus Soda is the better
player? To answer this, we must compute $P(\Theta_8 > \Theta_1)$. The
`laws` list does not have enough information to compute this, since it
only provides us with the marginal distributions. We need the joint
distribution of $\Theta_8$ and $\Theta_1$. Or rather, it suffices to
know the distribution of $\Theta_8 - \Theta_1$. We compute that as
follows.

```python
>>> diffLaw = model.law(lambda theta: avg(theta[8]) - avg(theta[1]))
>>> diffLaw.cdf(0)
0.0491471502270436
```

In the above, `diffLaw` is the distribution of $\Theta_8 - \Theta_1$.
The second line shows us that $P(\Theta_8 - \Theta_1 \le 0) \approx
0.05$. In other words, according to the model, there is a 95% chance
that Asparagus Soda is a better player than Potato Log.

Now suppose the two of them had a contest in which they each played the
game once and the higher score wins. What is the probability that
Asparagus Soda would win this contest?

To model such a contest, let $\theta_{{1i}}$ and $\theta_{{2i}}$ be the
(unknown) probabilities that the first and second players (respectively)
score $i$ points in the contest. Then the probability that the first
player wins is $\sum_{{j < i}} \theta_{{1i}} \theta_{{2j}}$. We encode
this function as follows.

```python
>>> def contest(theta_1, theta_2):
>>>     return sum(sum(theta_1[i] * theta_2[:i]) for i in range(1, 500))
```

Now let $C_{{mn}}$ be the (unknown) probability that Player $m$ beats
Player $n$ in such a contest. The following line of code constructs the
distribution of $C_{{81}}$, which is the (unknown) probability that
Asparagus Soda beats Potato Log in this single-game contest. This line
of code took about 4 minutes to execute on a Macbook.

```python
>>> contProb = model.law(lambda theta: contest(theta[8], theta[1]))
```

The actual probability that Asparagus Soda wins the contest is given by
$E[C_{{81}}]$, which we compute as follows.

```python
>>> contProb.mean
0.7862715276930023
```

So according to the model, Asparagus Soda has about a 79% chance of
beating Potato Log in a contest involving a single play of the game. To
see the model's uncertainty around this probability, we can look at the
density of $C_{{81}}$.

```python
>>> t = np.linspace(0, 1, 200)
>>> plt.plot(t, contProb.pdf(t))
>>> plt.show()
```
<a name='C81'>![image](images/aspvspot.svg)</a>

### Asparagus Soda vs. Pumpkins

We now turn our attention to comparing Asparagus Soda, who played only
once, to Pumpkins, who played four times. Their expected long-term
average scores, according to the the NDP model, are 67 and 72,
respectively. Since Pumpkins is Player 6, we can view the model's
uncertainty around the value 72 by looking at the density of $\Theta_6$.

```python
>>> t = np.linspace(1, 500, 500)
>>> plt.plot(t, laws[6].pdf(t)) # Pumpkins
>>> plt.show()
```
![image](images/pumpkins.svg)

Visually comparing this to [the corresponding graph for Asparagus
Soda](#asparagus), we see that the two expected long-term averages have
similar degrees of uncertainty.

To see the probability that Pumpkins is the better player, we compute as
before:

```python
>>> diffLaw = model.law(lambda theta: avg(theta[8]) - avg(theta[6]))
>>> diffLaw.cdf(0)
0.6249224285816907
```

This means there is a 62% chance that Pumpkins is the better player.

We can also consider a single-game contest between Asparagus Soda and
Pumpkins. As above, we compute $E[C_{{86}}]$:

```python
>>> contProb = model.law(lambda theta: contest(theta[8], theta[6]))
>>> contProb.mean
0.4836523602067427
```

This means that Asparagus Soda has a 48% chance of beating Pumpkins in a
single-game contest. To see the model's uncertainty around this
probability, we can look at the density of $C_{{86}}$.

```python
>>> t = np.linspace(0, 1, 200)
>>> plt.plot(t, contProb.pdf(t))
>>> plt.show()
```
![image](images/aspvspum.svg)

Comparing this to [the density of $C_{{81}}$](#C81), we see that the
model is much more uncertain about this match-up than it was about
Asparagus Soda vs. Potato Log.

'''

# pylint: disable=wrong-import-position
# pylint: disable=wrong-import-order
# pylint: disable=unused-import
# pylint: disable=ungrouped-imports

if __name__ == '__main__':
    from idp import idpModel
    import numpy as np
    import matplotlib.pyplot as plt
    import json
    import csv

    EXTENSIONS = ['png', 'jpg', 'pdf', 'svg', 'eps', 'ps']

    # pylint: disable-next=unspecified-encoding
    with open('gameexpl1.csv') as f:
        sheet = list(csv.reader(f))
        data = [
            [int(val) for val in row[7:] if val != '']
            for row in sheet[1:]
        ]

    from idp.tools import gamer
    g = gamer(7/3, 3, scale=28)
    baseMeas = [g.cdf(n + 0.5) - g.cdf(n - 0.5) for n in range(499)]
    baseMeas.append(g.sf(499.5))

    model = idpModel(1, 1, baseMeas, data=data)
    model.logScale += 42
    model.addSims(40000)
    # %time
    # CPU times: user 2min 49s, sys: 741 ms, total: 2min 50s
    # Wall time: 2min 50s
    print(model.ess) # 326.717352110184

    def avg(theta): # pylint: disable=missing-function-docstring
        return np.average(range(500), weights=theta)
    laws = [model.agentLaw(m, avg) for m in range(10)]
    # %time
    # CPU times: user 9.8 s, sys: 91.8 ms, total: 9.89 s
    # Wall time: 9.89 s

    for m in range(10):
        print(f'{sheet[m + 1][3]}: {laws[m].mean}')
    # Pumpkins: 37.8769311756713
    # Potato Log: 39.475686400565
    # The Thing: 32.3193302196986
    # Running Stardust: 79.6472672018615
    # Sweet Rolls: 54.5136744371402
    # Vertigo Gal: 52.0993770879504
    # Asparagus Soda: 40.2712946915477
    # The Matrix: 42.5543167395948
    # Goat Radish: 71.3566832439191
    # The Pianist Spider: 37.4693206147343

    data[8][0] = 39
    data[7][1] = 14
    model = idpModel(1, 1, baseMeas, data=data)
    model.logScale += 42
    model.addSims(40000)
    # %time
    # CPU times: user 2min 50s, sys: 784 ms, total: 2min 51s
    # Wall time: 2min 51s
    print(model.ess) # 22.32478257656367
    print(model.setEss(200)) # 2
    print(model.ess) # 1098.9123742787074
    laws = [model.agentLaw(m, avg) for m in range(10)]
    # %time
    # CPU times: user 9.84 s, sys: 100 ms, total: 9.94 s
    # Wall time: 9.94 s
    for m in range(10):
        print(f"{sheet[m + 1][3]}: {laws[m].mean}")
    # Pumpkins: 38.029840087762494
    # Potato Log: 39.450632512977556
    # The Thing: 31.92326020525782
    # Running Stardust: 84.41440312210776
    # Sweet Rolls: 62.72904055941699
    # Vertigo Gal: 51.208412372436186
    # Asparagus Soda: 39.92390947459972
    # The Matrix: 28.21322997473585
    # Goat Radish: 43.52445544679143
    # The Pianist Spider: 37.37972848216658

    # pylint: disable-next=unspecified-encoding
    with open('gameexpl2.csv') as f:
        sheet = list(csv.reader(f))
        data = [
            [int(val) for val in row[7:] if val != '']
            for row in sheet[1:]
        ]

    model = idpModel(1, 1, baseMeas, data=data)
    model.logScale += 42
    model.addSims(40000)
    # %time
    # CPU times: user 2min 51s, sys: 700 ms, total: 2min 51s
    # Wall time: 2min 51s
    print(model.ess) # 38.995295056814456
    print(model.setEss(200)) # 26
    print(model.ess) # 206.80979715975414
    laws = [model.agentLaw(m, avg) for m in range(10)]
    # %time
    # CPU times: user 9.8 s, sys: 96.6 ms, total: 9.89 s
    # Wall time: 9.9 s
    for m in range(10):
        print(f'{sheet[m + 1][3]}: {laws[m].mean}')
    # Vertigo Gal: 198.06866488887442
    # Potato Log: 31.183512061427916
    # The Thing: 26.333237442129764
    # The Matrix: 37.73199694794612
    # Running Stardust: 45.27946864625936
    # Goat Radish: 34.91069387324785
    # Pumpkins: 72.21882232954187
    # Sweet Rolls: 52.23247042222365
    # Asparagus Soda: 67.8642177970131
    # The Pianist Spider: 56.75360824901325

    t = np.linspace(1, 500, 500)
    plt.plot(t, laws[8].pdf(t))
    for ext in EXTENSIONS:
        plt.savefig('asparagus.' + ext)
    plt.show()
    plt.close()

    plt.plot(t, laws[1].pdf(t))
    for ext in EXTENSIONS:
        plt.savefig('potato.' + ext)
    plt.show()
    plt.close()

    diffLaw = model.law(lambda theta: avg(theta[8]) - avg(theta[1]))
    print(diffLaw.cdf(0)) # 0.0491471502270436

    # pylint: disable-next=missing-function-docstring
    def contest(theta_1, theta_2): # pylint: disable=invalid-name
        return sum(sum(theta_1[i] * theta_2[:i]) for i in range(1, 500))

    contProb = model.law(lambda theta: contest(theta[8], theta[1]))
    # %time
    # CPU times: user 3min 56s, sys: 770 ms, total: 3min 57s
    # Wall time: 3min 57s

    print(contProb.mean) # 0.7862715276930023

    t = np.linspace(0, 1, 200)
    plt.plot(t, contProb.pdf(t))
    for ext in EXTENSIONS:
        plt.savefig('aspvspot.' + ext)
    plt.show()
    plt.close()

    t = np.linspace(1, 500, 500)
    plt.plot(t, laws[6].pdf(t))
    for ext in EXTENSIONS:
        plt.savefig('pumpkins.' + ext)
    plt.show()
    plt.close()

    diffLaw = model.law(lambda theta: avg(theta[8]) - avg(theta[6]))
    print(diffLaw.cdf(0)) # 0.6249224285816907

    contProb = model.law(lambda theta: contest(theta[8], theta[6]))
    # %time
    # CPU times: user 3min 56s, sys: 510 ms, total: 3min 56s
    # Wall time: 3min 56s
    print(contProb.mean) # 0.4836523602067427

    t = np.linspace(0, 1, 200)
    plt.plot(t, contProb.pdf(t))
    for ext in EXTENSIONS:
        plt.savefig('aspvspum.' + ext)
    plt.show()
    plt.close()
