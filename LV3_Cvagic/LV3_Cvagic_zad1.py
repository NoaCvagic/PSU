import pandas as pd
import numpy as np

mtcars = pd.read_csv('mtcars.csv')
print('\n1.')
print(mtcars.sort_values(by='mpg').head(5))
print('\n2.')
print(mtcars[mtcars.cyl == 8].sort_values(by='mpg').head(3))
print('\n3.')
print(mtcars[mtcars.cyl == 6].mpg.mean())
print('\n4.')
print(mtcars[(mtcars.cyl == 4) & (mtcars.wt >= 2.000) & (mtcars.wt <= 2.200)].mpg.mean())
print('\n5.')
print(mtcars.am.value_counts())
print('\n6.')
am1_hp100 = mtcars[(mtcars.am == 1) & (mtcars.hp > 100)]
print(len(am1_hp100))
print('\n7.')
ukupnoWT = mtcars['wt'].sum() * 1000
print(str(ukupnoWT) + ' kg')