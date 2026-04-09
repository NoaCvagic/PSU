import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('cars_processed.csv')
print("Informacije o datasetu:")
print(df.info())


n_measurements = df.shape[0]
print(f"\n1. Broj automobila u datasetu: {n_measurements}")


print("\n2. Tipovi stupaca:")
print(df.dtypes)


max_price_car = df.loc[df['selling_price'].idxmax()]
min_price_car = df.loc[df['selling_price'].idxmin()]
print(f"\n3. Automobil s najvecom cijenom:\n{max_price_car}")
print(f"\nAutomobil s najmanjom cijenom:\n{min_price_car}")


cars_2012 = df[df['year'] == 2012].shape[0]
print(f"\n4. Broj automobila proizvedenih 2012. godine: {cars_2012}")


max_km_car = df.loc[df['km_driven'].idxmax()]
min_km_car = df.loc[df['km_driven'].idxmin()]
print(f"\n5. Automobil koji je presao najvise kilometara:\n{max_km_car}")
print(f"\nAutomobil koji je presao najmanje kilometara:\n{min_km_car}")


most_common_seats = df['seats'].mode()[0]
print(f"\n6. Najcesci broj sjedala: {most_common_seats}")


avg_km_diesel = df[df['fuel'] == 'Diesel']['km_driven'].mean()
avg_km_petrol = df[df['fuel'] == 'Petrol']['km_driven'].mean()
print(f"\n7. Prosjecna kilometraza:")
print(f"Diesel: {avg_km_diesel:.2f} km")
print(f"Petrol: {avg_km_petrol:.2f} km")



sns.relplot(data=df, x='km_driven', y='selling_price', hue='fuel')
plt.title('Kilometraza vs Cijena po tipu goriva')
plt.show()


df['selling_price'].hist(bins=30)
plt.xlabel('Prodajna cijena')
plt.ylabel('Frekvencija')
plt.title('Distribucija cijena automobila')
plt.show()


cat_cols = df.select_dtypes(include='object').columns.tolist()
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)


plt.figure(figsize=(12,8))
sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm', linewidths=1)
plt.title('Korelacija svih varijabli (numeričke i one-hot kodirane)')
plt.show()