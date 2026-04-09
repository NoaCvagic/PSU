import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, max_error

# ucitavanje podataka
df = pd.read_csv('cars_processed.csv')
print(df.info())


print("1. Uklanjanje stupca 'name'")
df.drop(columns=['name'], inplace=True)
print("\n2. Podjela skupa na train i test u omjeru 80:20 ")
from sklearn.model_selection import train_test_split
x = df.drop('selling_price', axis=1)
y = df['selling_price']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=300)
print("\n3. Skaliranje ulaznih varijabli")
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
Xtrain = scaler.fit_transform(X_train)
Xtest = scaler.transform(X_test)
print("\n4. Izrada modela linearne regresije")
model = LinearRegression()
model.fit(Xtrain, y_train)
print("\n5. Evaluacija modela")
y_pred = model.predict(Xtest)
print(mean_absolute_error(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))
print(r2_score(y_test, y_pred))
print(max_error(y_test, y_pred))

