from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def chargementDonnees(nomFichier):
    """
    [Description]


    Paramètres
    ----------
    nomFichier : string
        Le nom du fichier à charger, au format .csv

    Returns
    -------
    X : np.array
        L'ensemble des features
    Y : np.array
        La target

    Raises
    ------

    """

    #TODO: Charger les données en fonction des features intéressantes

    regSimple = pd.read_csv(nomFichier)
    print(regSimple.info())
    print(regSimple.describe())
    n = len(regSimple.columns)
    xColonne = regSimple.head().columns[0:-1]
    yColonne = regSimple.head().columns[-1]

    X = np.array(regSimple[xColonne]).reshape(-1, n - 1)
    # X = np.array(regSimple[["RM", "LSTAT"]]).reshape(-1, 3 - 1)
    Y = np.array(regSimple[yColonne]).reshape(-1, 1)
    return X, Y

def encodageValeursLitterales(X):
    le = LabelEncoder()
    Xencode = le.fit_transform(X)

    return Xencode

def miseAlEchelle(Xtrain):
    scaler = StandardScaler()
    scaler.fit(Xtrain)
    X_train_scaled = scaler.transform(Xtrain)
    
    return X_train_scaled

def decoupageDonneesAleatoires (X, Y, pourcentage):
    #TODO: Faire le découpage après le prétraitement

    n = len(X)
    indexTrainTest= list(range(n))

    np.random.shuffle(indexTrainTest)

    nTrainTest = int(round(n*pourcentage, 0))


    Xtrain = X[indexTrainTest[0:nTrainTest -1], :]
    Ytrain = Y[indexTrainTest[0:nTrainTest -1], :]
    Xtest = X[indexTrainTest[nTrainTest -1:n], :]
    Ytest = Y[indexTrainTest[nTrainTest -1:n], :]

    return Xtrain, Ytrain, Xtest, Ytest

def coefficientSimpleInit(X, Y):
    a = (Y[-1]-Y[0])/(X[-1]-X[0])
    b = Y[0] - a*X[0]
    
    return np.array([a, b])

def fonctionCout(X, Y, theta):
    n = len(X)
    cout = 1/(2*n)*(X.dot(theta) - Y)

    return cout

def ecritureMatricielleSysEqu(Xtrain):
    ones = np.array([1]*len(Xtrain)).reshape(-1, 1)
    Xones = np.append(Xtrain, ones, axis=1)

    return Xones

def gradient(Xtrain, Ytrain, theta):
    n = len(Xtrain)
    Xones = ecritureMatricielleSysEqu(Xtrain)
    dtheta = 1/n * (np.transpose(Xones)).dot(Xones.dot(theta)-Ytrain)

    return dtheta

def descenteGradient(X, Y, theta, alpha, n):
    dtheta = 0

    while n > 0:
        dtheta = gradient(X, Y, theta)
        theta = theta - alpha * dtheta
        n -= 1

    return theta

def modeleLineaireSimple(Xtrain, Ytrain, alpha, n):

    theta = coefficientSimpleInit(Xtrain, Ytrain)
    
    theta = descenteGradient(Xtrain, Ytrain, theta, alpha, n)

    return theta

def coeffDetermination(Xtest, Ytest, Ypred):
    n = len(Xtest)

    sommeVariance = 0
    moyenneCarreResiduts = 0
    yMean = np.mean(Ytest)

    for i in range(n):
        sommeVariance += (Ytest[i]-Ypred[i])**2
        moyenneCarreResiduts += (Ytest[i]-yMean)**2

    R = 1 - sommeVariance/moyenneCarreResiduts 

    return R    

def coefficientMultipleInit(Xtrain, Ytrain):
    n = len(Xtrain[0, :])
    Xones = ecritureMatricielleSysEqu(Xtrain[0:n, :])
    XtrainPlus = np.linalg.pinv(Xones)
    theta = XtrainPlus.dot(Ytrain[0:n, :]) + (np.identity(n+1) - XtrainPlus.dot(Xones)).dot(np.random.rand(n+1, 1))

    return theta

def modeleLineaireMultiple(Xtrain, Ytrain, alpha, n):
    theta = coefficientMultipleInit(Xtrain, Ytrain)
    
    theta = descenteGradient(Xtrain, Ytrain, theta, alpha, n)

    return theta

def ecritureXPoly(X, degre):
    Xpoly = np.empty((len(X), 0))
    while degre >= 0:
        Xdegre = np.array([x**degre for x in X]).reshape(-1, 1)
        Xpoly = np.append(Xpoly, Xdegre, axis = 1)
        degre -= 1

    return Xpoly

# def modelePolynomial(Xtrain, Ytrain, degre alpha, n):


if __name__ == "__main__":
    alpha = 1e-5
    n = 10000
    degre = 3

    # X, Y = chargementDonnees("premierModeleIA_app/static/Data_Reg/reg_simple.csv")
    # Xtrain, Ytrain, Xtest, Ytest = decoupageDonneesAleatoires(X, Y, 0.8)


    # theta = modeleLineaireSimple(Xtrain, Ytrain, alpha, n)

    # Ypred = ecritureMatricielleSysEqu(Xtest).dot(theta)

    # R = coeffDetermination(Xtest, Ytest, Ypred)

    # plt.scatter(Xtrain, Ytrain)
    # plt.scatter(Xtest, Ytest)
    # plt.plot(Xtest, Ypred)
    # plt.show()

    # print(R)

    # X, Y = chargementDonnees("premierModeleIA_app/static/Data_Reg/boston_house_prices.csv")
    # X = miseAlEchelle(X)
    # Y = miseAlEchelle(Y)

    # Xtrain, Ytrain, Xtest, Ytest = decoupageDonneesAleatoires(X, Y, 0.8)


    # theta = modeleLineaireMultiple(Xtrain, Ytrain, alpha, n)
    # # theta = coefficientMultipleInit(Xtrain, Ytrain)

    # Ypred = ecritureMatricielleSysEqu(Xtest).dot(theta)

    # R = coeffDetermination(Xtest, Ytest, Ypred)

    # ax = plt.axes(projection="3d")
    # # ax.scatter(Xtrain[:, 0], Xtrain[:, 1], Ytrain)
    # ax.scatter(Xtest[:, 0], Xtest[:, 1],  Ytest)
    # ax.scatter(Xtest[:, 0], Xtest[:, 1], Ypred)
    # plt.show()

    # print(R)
    # print(mean_squared_error(Ytest, Ypred))

    X, Y = chargementDonnees("Data_Reg/Position_Salaries.csv")

    X[:, 0] = encodageValeursLitterales(X[:, 0])

    Xpoly = ecritureXPoly(X[:, 1], 2)
    Xpoly = miseAlEchelle(Xpoly)
    Y = miseAlEchelle(Y)

    Xtrain, Ytrain, Xtest, Ytest = decoupageDonneesAleatoires(Xpoly, Y, 0.8)

    theta = modeleLineaireMultiple(Xtrain, Ytrain, alpha, n)
    # theta = coefficientMultipleInit(Xtrain, Ytrain)

    Ypred = ecritureMatricielleSysEqu(Xtest).dot(theta)

    # R = coeffDetermination(Xtest, Ytest, Ypred)

    # ax = plt.axes(projection="3d")
    # ax.scatter(Xtrain[:, 0], Xtrain[:, 1], Ytrain)
    plt.scatter(Xtrain[:, -2],  Ytrain)
    plt.scatter(Xtest[:, -2],  Ytest)
    plt.show()

    print(mean_squared_error(Ytest, Ypred))

    # print(R)