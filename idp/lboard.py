r'''The `Leaderboard` class and related objects.

# Description

This module is used to create csv files of data for the video game
leaderboards example in the package, `idp`. In the following
documentation, we illustrate this by creating five different csv files.
Each file will contain the data for 10 players.

We first construct a `Leaderboard` instance with 10 players. Each player
on the leaderboard is a `Player` object that contains data about the
player as well as a list of historical scores for that player. The
number of scores on the list is random and the marginal distribution of
each score is the `gamer` distribution from the `idp.tools` module. To
construct such a player, we must decide on the mean number of scores on
the list, and also the shape and scale parameters of the `gamer`
distribution.

The shape parameters are $r$ and $a$. We will let $a = 3$, so that in
our game, the player has 3 failures before the game is over. The
parameter $r$ determines the decay rate of skill level in the general
population. We want to set $r$ so that this decay rate matches those
seen in `TETRIS_SCORES`.

If we fit the scores in `TETRIS_SCORES` to a power law distribution, we
get the following:

```python
>>> from idp.lboard import *
>>> from scipy.stats import pareto
>>> pareto.fit(TETRIS_SCORES, floc=0)[0]
2.361453182701752
```

The resulting exponent is approximately $7/3$. We therefore set
$r = 7/3$.

We will set the scale parameter $c$ so that the average score is
approximately $50$. Since the mean of the `gamer` distribution is
$cr/(r - 1)$, we will set $c = 28$, which gives a mean of $49$.

Finally, we set the mean number of historical scores for each player to
be $2.4\sqrt{10}$, which is about 7.6. To explain this, note that if
there are 10 players, and every game generates a unique score, then we
will see $24\sqrt{10}$ unique scores. It is then as if we have 10
players rolling a $24\sqrt{10}$-sided die $2.4\sqrt{10}$ times. If we
compare this to 320 thumbtacks being flicked (2 outcomes) 9 times, we
see that
$$
    10 \cdot 24\sqrt{10} \cdot 2.4\sqrt{10}
        = 320 \cdot 2 \cdot 9.
$$
In this way, a 10-player leaderboard ought to generate a model of
comparable complexity to Liu's thumbtack example.

The `makeBoard` function will construct our leaderboard for us. This
function randomly regenerates its data until the total number of scores
in the leaderboard is between 72 and 80. Before calling this function,
we can check how many times it is expected to loop.

```python
>>> from math import sqrt
>>> meanRunTime(2.4 * sqrt(10))
7.895576353625911
```

Since this number seems reasonable, we will go ahead and create a
leaderboard.

```python
>>> lboard = makeBoard((7/3, 3), 28, 2.4 * sqrt(10))
```

Although this would be enough to get us started, we want to add one more
column to this spreadsheet. We want to add a column that gives the
player's long-term average score, according to an IDP model. We will do
this by updating the `estAvg` property of each `Player` object in the
leaderboard.

We first build the base measure of our IDP model.

```python
>>> from idp.tools import gamer
>>> g = gamer(7/3, 3, scale=28)
>>> baseMeas = [g.cdf(n + 0.5) - g.cdf(n - 0.5) for n in range(499)]
>>> baseMeas.append(g.sf(499.5))
```

We then extract the data from our leaderboard for the IDP model.

```python
>>> players = sorted(lboard.players, key=lboard.playCountRank)
>>> data = [player.scores for player in players]
```

Finally, we build the IDP model and add 40,000 weighted simulations.

```python
>>> from idp import idpModel
>>> model = idpModel(1, 1, baseMeas, data=data)
>>> model.scale += 42
>>> model.addSims(40000) # took ~3m on a Macbook
>>> model.ess
326.717352110184
```

We now compute the estimated law for each player.

```python
>>> import numpy as np
>>> def avg(theta):
>>>     return np.average(range(500), weights=theta)
>>> laws = [model.agentLaw(m, avg) for m in range(10)] # took ~10s
```

Lastly, we add the estimated mean to each `Player` object and generate
the CSV file.

```python
>>> for m in range(10):
>>>     players[m].estAvg = laws[m].mean
>>> makeCSV(lboard, 'gameexpl1')
```

So that we have several scenarios to choose from, we generate four more
CSV files in the same way.

```python
>>> for i in range(2, 6):
>>>     lboard = makeBoard((7/3, 3), 28, 2.4 * sqrt(10))
>>>     players = sorted(lboard.players, key=lboard.playCountRank)
>>>     data = [player.scores for player in players]
>>>     model = idpModel(1, 1, baseMeas, data=data)
>>>     model.scale += 42
>>>     model.addSims(40000)
>>>     print(model.ess)
>>>     laws = [model.agentLaw(m, avg) for m in range(10)]
>>>     for m in range(10):
>>>         players[m].estAvg = laws[m].mean
>>>     makeCSV(lboard, f'gameexpl{i}')
12.953846545567364
21.65174613922022
116.00270773736477
4.515612542210865
```

'''

import os
import csv
import numpy as np
from scipy.stats import pareto, gamma, nbinom, rankdata

__all__ = [
    'TETRIS_SCORES',
    'USER_NAMES',
    'meanRunTime',
    'makeBoard',
    'makeCSV',
    'Player',
    'Leaderboard'
]

TETRIS_SCORES = np.array([
    29486164, 16700760, 13793540, 12409180, 11966100,  8063900,
    7220241,  7081880,  6787420,  6563440,  6529560,  6492500,
    6249920,  5435960,  4899280,  4890220,  4570640,  4222920,
    4213540,  3835120,  3740500,  3600460,  3222400,  3067100,
    2790920,  2743060,  2605320,  2529080,  2433160,  2382340,
    2373940,  2302480,  2281848,  2275996,  2153480,  2114180,
    2108820,  2077552,  2043580,  1834120,  1777456,  1768400,
    1705680,  1702640,  1696224,  1659860,  1657560,  1649320,
    1638000,  1632505,  1626880,  1608500,  1606732,  1554880,
    1537800,  1484501,  1483360,  1476400,  1472600,  1442340,
    1435280,  1412260,  1406260,  1404800,  1390000,  1388900,
    1386260,  1379220,  1374100,  1372600,  1372600,  1371040,
    1362703,  1357480,  1352620,  1350742,  1349060,  1344740,
    1333731,  1318660,  1316900,  1316360,  1312940,  1308962,
    1307370,  1305040,  1304500,  1301740,  1301080,  1300840,
    1295260,  1291493,  1291320,  1290870,  1287470,  1279920,
    1278140,  1276120,  1275602,  1274731,  1273920,  1272350,
    1266260,  1264796,  1264240,  1264020,  1263780,  1257920,
    1254915,  1254580,  1254374,  1252920,  1252240,  1251654,
    1251620,  1250180,  1247817,  1245200,  1240420,  1233940,
    1232740,  1228552,  1227040,  1223380,  1222363,  1220700,
    1219520,  1217852,  1217440,  1214942,  1214154,  1213660,
    1211680,  1210200,  1208900,  1206280,  1204040,  1202460,
    1200120,  1199692,  1196360,  1195860,  1190760,  1183460,
    1182458,  1182320,  1179234,  1178960,  1177147,  1175740,
    1170600,  1168399,  1167431,  1162380,  1157959,  1154920,
    1154647,  1151840,  1151040,  1143420,  1142423,  1141500,
    1137880,  1137515,  1130820,  1126760,  1126320,  1123215,
    1120300,  1116229,  1115500,  1112480,  1110940,  1110720,
    1108700,  1107400,  1107200,  1102342
])
'''`ndarray`: This 1 x 178 array is the actual global leaderboard for
Tetris.
'''

USER_NAMES = [
    'Asparagus Soda',
    'Goat Radish',
    'Potato Log',
    'Pumpkins',
    'Running Stardust',
    'Sweet Rolls',
    'The Matrix',
    'The Pianist Spider',
    'The Thing',
    'Vertigo Gal'
]
'''`list`[`str`]: The list of user names to use in the leaderboard
example.
'''

def meanRunTime(mean):
    r'''Returns the expected number of trials until the first success,
    where a trial involves generating a negative binomial random
    variable $X$ and success occurs when $72 \le X \le 80$.

    **Arguments:**

    * **mean** (`float`): Used to determine the distribution of $X$. If
        $\mu$ equals `mean`, then $X$ has a negative binomial
        distribution with $n = 10$ and $p = 1/\mu$.

    **Returns:**

    * `float`: $1/P(72 \le X \le 80)$.

    '''
    totalPlayCount = nbinom(10, 1/mean)
    return 1/(totalPlayCount.cdf(80) - totalPlayCount.cdf(71))

def makeBoard(args, scale, mean):
    r'''Makes a leaderboard of 10 players for analyzing with the
    iterated Dirichlet process. The names of the players are taken from
    the constant, `USER_NAMES`. The `args`, `scale`, and `mean`
    arguments are passed directly to the `Player` constructor for each
    player on the leaderboard.

    This function will regenerate the data if the total number of games
    is not between $72$ and $80$, inclusive.

    To explain why, let $T$ be the total number of games played, and let
    $n$ be the total number of thumbtacks used in the thumbtack example.
    Assuming every game produces a unique score, we will have $T$ trials
    with $T$ observed outcomes. In the thumbtack example, we have $9n$
    trials with $2$ observed outcomes. For these to be of comparable
    complexity, we want $T^2 = 18n$.

    In the thumbtack example, we used $n = 320$. To put bounds on the
    possible complexity of this example, we add and subtract $10\%$,
    requiring that $288 \le n \le 352$. This gives approximately
    $72 \le T \le 80$.

    **Returns:**

    * `Leaderboard`: The randomly generated leaderboard.

    '''
    while True:
        lboard = Leaderboard([
            Player(name, args, scale, mean)
            for name in USER_NAMES
        ])
        if 72 <= sum(player.playCount for player in lboard.players) <= 80:
            return lboard

def makeCSV(lboard, filename='gameexpl'):
    r'''Makes a csv file filled with data for the leaderboards example.
    Will not overwrite a previously existing file.

    The following example code will generate 5 scenarios worth of data:

    ```python
    >>> from idp.gamedata import makeCSV
    >>> for i in range(1, 6):
    ...     makeCSV(f'gameexpl{i}')
    ```

    **Arguments:**

    * **lboard** (`Leaderboard`): The leaderboard used to generate the
        csv file.
    * **filename** (`str`): The filename to use, without extension.
        Defaults to "gameexpl".

    **Raises:**

    * **FileExistsError:** If the csv file already exist in the current
        working directory.

    '''
    if os.path.exists(filename + '.csv'):
        raise FileExistsError(f'{filename}.csv already exists')

    with open(filename + ".csv", "w", newline="", encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=lboard.fieldnames)
        writer.writeheader()
        writer.writerows(
            sorted(lboard.dictify(), key=lambda row: row['by avg'])
        )

class Player:
    '''A player in the hypothetical game.

    '''

    def __init__(self, name, args, scale, mean):
        r'''Creates a `Player` object with the given `name` attribute.

        The player's scores have a `gamer` distribution whose shape
        parameters are given by `args` and whose scale parameter is
        given by `scale`.

        **Arguments:**

        * **args** (`tuple`(`float`)): A tuple of length 2. The first
            number is the shape parameter of the Pareto distribution
            used to determine the `trueAvg` property. The second is the
            shape parameter of the gamma distribution used to determine
            the `scores` property.
        * **scale** (`float`): The scale parameter of the Pareto
            distribution used to determine the `trueAvg` property.
        * **mean** (`float`): The mean of the `playCount` attribute.

        '''
        self.name = name
        '''`str`: The user name of the player.'''

        self.playCount = np.random.geometric(1/mean)
        '''`int`: The number of times the player played. This attribute
        is random and has a geometric distribution. The value of this
        attribute is generated on object creation, and is then
        permanently fixed.
        '''

        self.trueAvg = pareto.rvs(args[0], scale=scale)
        '''`float`: The theoretical long-term average score of the
        player. Determined at instance creation by randomly sampling
        from a `scipy.pareto` distribution.
        '''

        rawScores = gamma.rvs(
            args[1],
            scale=self.trueAvg / args[1],
            size=self.playCount
        )
        self.scores = np.array([round(score) for score in rawScores])
        '''`list`[`int`]: The scores the player earned. Determined at
        instance creation by randomly sampling from a `scipy.gamma`
        distribution with mean `trueAvg`. Scores are rounded to the
        nearest integer.
        '''

        self.estAvg = self.avgScore
        '''`float`: An estimate of the player's theoretical long-term
        average score. Defaults to the mean of their actual scores, but
        can be replaced by something more sophisticated, such as an
        estimate created with the `idp.tools.IDPModel` class.
        '''

    @property
    def highScore(self):
        '''`int`: The player's highest earned score.'''
        return max(self.scores)

    @property
    def avgScore(self):
        '''`int`: The player's average earned score.'''
        return np.mean(self.scores)

    @property
    def diff(self):
        '''`int`: Returns `avgScore` minus `trueAvg`.'''
        return self.avgScore - self.trueAvg

class Leaderboard:
    '''A leaderboard in the hypothetical game.

    '''

    fields = [
        'player #',
        'rank',
        'by avg',
        'name',
        'hi score',
        'avg score',
        '# games',
        'true avg',
        'est avg',
        'diff'
    ]
    '''`list`[`str`]: The non-score field names used in the `dictify`
    method.
    '''

    types = {
        'name': str,
        'avg score': float,
        'true avg': float,
        'est avg': float,
        'diff': float
    }
    '''`dict`{`str`:`type`}: The non-integer types used in the `dictify`
    method.
    '''

    def __init__(self, players):
        '''Creates a `Leaderboard` instance, assigning the given
        argument to the `players` property.

        '''
        self.players = players
        '''`list`[`Player`]: The list of `Player` objects that are on
        the leaderboard.
        '''

        self.fieldnames = self.fields + [
            f'score {gameNum}'
            for gameNum in range(
                max(player.playCount for player in players)
            )
        ]
        '''`list`[`str`]: The full list of field names used in the
        `dictify` method.
        '''

    def playCountRank(self, player):
        '''Returns the ranking of the given player, according to how
        many games they played. The player with the most games has rank
        0, and the ranks increase from there. Every player has a unique
        rank, with ties going to the player that appears first in
        `players`.

        The play count rankings are intended to be used as indices in an
        ordered list of players.

        **Arguments:**

        * **player** (`Player`): The player whose rank to return.

        **Returns:**

        * `int`: The rank of the player.

        '''
        playCounts = np.array([player.playCount for player in self.players])
        countRanks = rankdata(-playCounts, method='ordinal')
        return countRanks[self.players.index(player)] - 1

    def highScoreRank(self, player):
        '''Returns the ranking of the given player, according to their
        high score. The player with the best high score has rank 1, and
        the ranks increase from there. Competition ranking is used to
        handle ties.

        **Arguments:**

        * **player** (`Player`): The player whose rank to return.

        **Returns:**

        * `int`: The rank of the player.

        '''
        highScores = np.array([player.highScore for player in self.players])
        highScoreRanks = rankdata(-highScores, method='min')
        return highScoreRanks[self.players.index(player)]

    def avgScoreRank(self, player):
        '''Returns the ranking of the given player, according to their
        average score. The player with the best average score has rank
        1, and the ranks increase from there. Average scores are rounded
        to the nearest integer before ranks are computed. Competition
        ranking is used to handle ties.

        **Arguments:**

        * **player** (`Player`): The player whose rank to return.

        **Returns:**

        * `int`: The rank of the player.

        '''
        avgScores = np.array(
            [round(player.avgScore) for player in self.players]
        )
        avgScoreRanks = rankdata(-avgScores, method='min')
        return avgScoreRanks[self.players.index(player)]

    def dictify(self):
        '''Creates and returns a list of dictionaries suitable for
        writing to a spreadsheet.

        **Returns:**

        * `list`[`dict`]: A list of dictionaries, one for each `Player`
            object in the `players` attribute.

        '''
        dictList = []
        for player in self.players:
            d = {
                'player #': self.playCountRank(player),
                'rank': self.highScoreRank(player),
                'by avg': self.avgScoreRank(player),
                'name': player.name,
                'hi score': player.highScore,
                'avg score': round(player.avgScore),
                '# games': player.playCount,
                'true avg': player.trueAvg,
                'est avg': player.estAvg,
                'diff': player.diff
            }
            d.update({
                f'score {gameNum}': score
                for gameNum, score in enumerate(player.scores)
            })
            dictList.append(d)
        return dictList
