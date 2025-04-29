'''Makes a csv file of data for the video game leaderboards example.

'''

# pylint: disable=wildcard-import
# pylint: disable=unused-wildcard-import
# pylint: disable=wrong-import-position
# pylint: disable=wrong-import-order

from lboard import *
from scipy.stats import pareto
print(pareto.fit(TETRIS_SCORES, floc=0)[0]) # 2.361453182701752

from math import sqrt
print(meanRunTime(2.4 * sqrt(10))) # 7.895576353625911

lboard = makeBoard((7/3, 3), 28, 2.4 * sqrt(10))

from idp.tools import gamer
g = gamer(7/3, 3, scale=28)
baseMeas = [g.cdf(n + 0.5) - g.cdf(n - 0.5) for n in range(499)]
baseMeas.append(g.sf(499.5))

players = sorted(lboard.players, key=lboard.playCountRank)
data = [player.scores for player in players]
from idp import idpModel
model = idpModel(1, 1, baseMeas, data=data)
model.scale += 42
model.addSims(40000)
# %time
# CPU times: user 2min 49s, sys: 741 ms, total: 2min 50s
# Wall time: 2min 50s
print(model.ess) # 326.717352110184

import numpy as np
def avg(theta): # pylint: disable=missing-function-docstring
    return np.average(range(500), weights=theta)
laws = [model.agentLaw(m, avg) for m in range(10)]
# %time
# CPU times: user 9.8 s, sys: 91.8 ms, total: 9.89 s
# Wall time: 9.89 s

for m in range(10):
    players[m].estAvg = laws[m].mean
makeCSV(lboard, 'gameexpl1')

for i in range(2, 6):
    lboard = makeBoard((7/3, 3), 28, 2.4 * sqrt(10))
    players = sorted(lboard.players, key=lboard.playCountRank)
    data = [player.scores for player in players]
    model = idpModel(1, 1, baseMeas, data=data)
    model.scale += 42
    model.addSims(40000)
    print(model.ess)
    laws = [model.agentLaw(m, avg) for m in range(10)]
    for m in range(10):
        players[m].estAvg = laws[m].mean
    makeCSV(lboard, f'gameexpl{i}')
