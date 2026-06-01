import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv("cars_processed.csv")

df = df.drop(['name'], axis=1)

df = pd.get_dummies(
    df,
    columns=['fuel',
             'seller_type',
             'transmission',
             'owner'],
    drop_first=True
)

print(df.dtypes)
X = df.drop('selling_price', axis=1)
y = df['selling_price']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=300
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print("R2 =", r2_score(y_test,y_pred))