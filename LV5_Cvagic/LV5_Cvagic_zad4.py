import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report
)

df = pd.read_csv("occupancy_processed.csv")

X = df[['S3_Temp', 'S5_CO2']]
y = df['Room_Occupancy_Count']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

logreg = LogisticRegression()

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

ConfusionMatrixDisplay(
    confusion_matrix(y_test, y_pred)
).plot()

print("Accuracy:",
      accuracy_score(y_test, y_pred))

print(classification_report(
      y_test,
      y_pred))

plt.show()