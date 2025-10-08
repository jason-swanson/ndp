'''Makes a csv file of data for the amazon reviews example.

'''

import os
import csv
from random import choices
from statistics import mean
import numpy as np
from scipy.stats import dirichlet

DICE_PRIOR = [5/3, 4/3, 1, 2/3, 1/3] # avg = 2.3333

def makeCSV():
    '''Makes the file `revexpl.csv`, filled with data for the amazon
    reviews example. Uses the `DICE_PRIOR` constant. Will not overwrite
    a previously existing file.

    **Raises:**

    * **FileExistsError:** If `revexpl.csv` already exists in the
        current working directory.

    '''
    if os.path.exists('revexpl.csv'):
        raise FileExistsError
    fieldnames = (
        ['product #'] +
        [f'{starCount} stars' for starCount in range(1, 6)] +
        ['# reviews', 'empirical mean', 'true mean', 'diff']
    )
    data = []

    samples = 0
    prodNum = 0

    while samples < 128 * 9:
        row = {'product #': prodNum}

        diceProbs = dirichlet(DICE_PRIOR).rvs()[0]

        numRolls = np.random.geometric(1/25)
        if samples + numRolls > 128 * 9:
            numRolls = 128 * 9 - samples
        samples += numRolls

        rolls = choices(range(5), weights=diceProbs, k=numRolls)

        row.update(
            {f'{state + 1} stars': rolls.count(state) for state in range(5)}
        )

        row['# reviews'] = numRolls
        row['empirical mean'] = 1 + mean(rolls)
        row['true mean'] = 1 + sum(
            val * prob for val, prob in enumerate(diceProbs)
        )
        row['diff'] = row['empirical mean'] - row['true mean']

        data.append(row)
        prodNum += 1

    with open("revexpl.csv", "w", newline="", encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

if __name__ == '__main__':
    makeCSV()
