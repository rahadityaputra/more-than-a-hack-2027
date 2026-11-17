import os, sys
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import games

for name, path in sorted(games.asset_paths(source='tactical').items()):
    print('%-14s %-46s %s' % (name, os.path.basename(path),
                              'ok' if os.path.isfile(path) else 'MISSING'))

import baseline

codes = baseline.build()   # 0 = home, 1 = away, -1 = no answer