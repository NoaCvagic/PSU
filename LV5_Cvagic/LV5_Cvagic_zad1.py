'''
Room occupancy classification 

R.Grbic, 2024.

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ucitaj podatke za ucenje
df = pd.read_csv('occupancy_processed.csv')

feature_names = ['S3_Temp', 'S5_CO2']
target_name = 'Room_Occupancy_Count'
class_names = ['Slobodna', 'Zauzeta']

X = df[feature_names].to_numpy()
y = df[target_name].to_numpy()

# Scatter plot
plt.figure()
for class_value in np.unique(y):
    mask = y == class_value
    plt.scatter(X[mask, 0], X[mask, 1], label=class_names[class_value])

plt.xlabel('S3_Temp')
plt.ylabel('S5_CO2')
plt.title('Zauzetost prostorije')
plt.legend()
plt.show()

#a) Primjecuje se da ima dvije klase i da se klasa zauzete prostorije pojavljuje pri većim vrijednostima CO2. 
# Također se primjećuje da klase nisu potpuno odvojene, već postoji određeno preklapanje između njih.
print("b) Ukupan broj uzoraka: ", len(df))

unique, counts = np.unique(y, return_counts=True)
for u, c in zip(unique, counts):
    print("c) Klasa", u, ":", c)