import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

def non_func(x):
    y = 1.6345 - 0.6235*np.cos(0.6067*x) - 1.3501*np.sin(0.6067*x) - 1.1622 * np.cos(2*x*0.6067) - 0.9443*np.sin(2*x*0.6067)
    return y

def add_noise(y):
    np.random.seed(14)
    varNoise = np.max(y) - np.min(y)
    y_noisy = y + 0.1*varNoise*np.random.normal(0,1,len(y))
    return y_noisy
 
x = np.linspace(1,10,50)
y_true = non_func(x)
y_measured = add_noise(y_true)

x = x[:, np.newaxis]
y_measured = y_measured[:, np.newaxis]


np.random.seed(12)
indeksi = np.random.permutation(len(x))
indeksi_train = indeksi[0:int(np.floor(0.7*len(x)))]
indeksi_test = indeksi[int(np.floor(0.7*len(x)))+1:len(x)]

xtrain = x[indeksi_train,]
ytrain = y_measured[indeksi_train]

xtest = x[indeksi_test,]
ytest = y_measured[indeksi_test]

#stupnjevi polinoma
degrees = [2, 6, 15]

MSEtrain = []
MSEtest = []

plt.figure()

for d in degrees:
    
    poly = PolynomialFeatures(degree=d)

    xtrain_poly = poly.fit_transform(xtrain)
    xtest_poly = poly.transform(xtest)
    x_poly_all = poly.transform(x)

    model = lm.LinearRegression()
    model.fit(xtrain_poly, ytrain)

    #predikcije
    ytrain_pred = model.predict(xtrain_poly)
    ytest_pred = model.predict(xtest_poly)

    #MSE
    MSEtrain.append(mean_squared_error(ytrain, ytrain_pred))
    MSEtest.append(mean_squared_error(ytest, ytest_pred))

    #crtanje modela
    plt.plot(x, model.predict(x_poly_all), label=f"degree={d}")
    
plt.scatter(xtrain, ytrain, color='blue', label='train')

plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Usporedba modela razlicitih stupnjeva polinoma')
plt.show()

print("MSE za train skup:", MSEtrain)
print("MSE za test skup:", MSEtest)