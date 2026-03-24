import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

mtcars = pd.read_csv('mtcars.csv')
mtcars.groupby('cyl')['mpg'].mean().plot(kind='bar')
plt.title('Prosjecna potrosnja po broju cilindara')
plt.xlabel('Broj cilindara')
plt.ylabel('Prosjecna potrosnja (mpg)')
plt.show()

mtcars.boxplot(column='wt', by='cyl')
plt.title('Tezina automobila po broju cilindara')
plt.xlabel('Broj cilindara')
plt.ylabel('Tezina (wt)')
plt.show()

mtcars.boxplot(column='mpg', by='am')
plt.title('Potrosnja: rucni vs automatski')
plt.xlabel('Vrsta mjenjaca (0 = automatski, 1 = rucni)')
plt.ylabel('Potrosnja (mpg)')
plt.show()

rucni = mtcars[mtcars.am == 1]
auto = mtcars[mtcars.am == 0]

plt.scatter(rucni.hp, rucni.qsec, color='blue', label='Rucni')
plt.scatter(auto.hp, auto.qsec, color='red', label='Automatski')
plt.title('Ubrzanje vs Snaga')
plt.xlabel('Snaga (hp)')
plt.ylabel('Ubrzanje (qsec)')
plt.legend()
plt.show()
