"""
Import test/train data
"""

import pandas as pd

df = pd.read_csv('data/raw/train.csv')

print(df.head(5))
