from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

PATH = "premierModeleIA_app/static/Data_Reg/"

def chargementDonnees(nomFichier):

    """
    [Description]
        Charge les données depuis un fichier CSV et retourne les features et la target sous forme de dataframe.

    Paramètres
    ----------
    nomFichier : string
        Le nom du fichier à charger, au format .csv

    Returns
    -------
    X : pd.DataFrame
        L'ensemble des features
    Y : pd.DataFrame
        La target

    Raises
    ------

    """

    regSimple = pd.read_csv(nomFichier)

    # print(regSimple.info())
    # print(regSimple.describe())

    xColonne = regSimple.head().columns[0:-1]
    yColonne = regSimple.head().columns[-1]

    X = regSimple[xColonne]
    Y = regSimple[yColonne]

    return X, Y

def pretraitementDonnees(X):
    """
    [Description]
        Permet de prétraîter les données en les encodant si une colonne est de type object et d'appliquer une normalisation des valeurs (la standardisation est problématique pour le dernier jeu de données utilisé.)

    Paramètres
    ----------
    X : pd.DataFrame
        Le dataframe à prétraîter (encodage au besoin et mise à l'échelle)

    Returns
    -------
    X : pd.DataFrame
        L'ensemble des features prétraîtées

    Raises
    ------

    """

    le = LabelEncoder()
    scaler1 = MinMaxScaler()
    scaler2 = StandardScaler()

    nShape = len(X.shape)

    if  nShape == 1:
        columns = [X.name]
        if (X.dtype == np.object_):
            X = le.fit_transform(X)
        X = pd.DataFrame(scaler1.fit_transform(np.array(X).reshape(-1, 1)), columns = columns)
        # X = pd.DataFrame(scaler2.fit_transform(np.array(X).reshape(-1, 1)), columns = columns)


    else:
        columns = X.columns
        nColonnes = len(columns)

        for col in X.columns:
            if (X[col].dtype == np.object_):
                X[col] = le.fit_transform(X[col])
            else:
                continue

        X = pd.DataFrame(scaler1.fit_transform(np.array(X).reshape(-1, nColonnes)), columns = columns)
        # X = pd.DataFrame(scaler2.fit_transform(np.array(X).reshape(-1, nColonnes)), columns = columns)

    return X

def decoupageDonneesAleatoires (X, Y, pourcentage):
    """
    [Description]
        Découpe les caractéristiques et la cible en train/test.

    Paramètres
    ----------
    X, Y : pd.DataFrame
        Les dataframes à découper selon un certain ratio
    pourcentage : float
        Le ratio à appliquer lors de la découpe.

    Returns
    -------
    Xtrain, Ytrain, Xtest, Ytest : np.array
        Les différents tableaux issus de la découpe.

    Raises
    ------

    """

    nLignes = len(X.values)

    if len(X.shape) == 2:
        nColonnes = X.shape[1]
    else:
        nColonnes = 1

    indexTrainTest= list(range(nLignes))

    np.random.shuffle(indexTrainTest)

    nTrainTest = int(round(nLignes*pourcentage, 0))

    if nColonnes == 1:
        Xtrain = np.array(X.iloc[indexTrainTest[0:nTrainTest -1]]).reshape(-1, 1)
        Xtest = np.array(X.iloc[indexTrainTest[nTrainTest -1:-1]]).reshape(-1, 1)
    else:
        Xtrain = np.array(X.iloc[indexTrainTest[0:nTrainTest -1], :]).reshape(-1, nColonnes)
        Xtest = np.array(X.iloc[indexTrainTest[nTrainTest -1:-1], :]).reshape(-1, nColonnes)

    Ytrain = np.array(Y.iloc[indexTrainTest[0:nTrainTest -1]]).reshape(-1, 1)
    Ytest = np.array(Y.iloc[indexTrainTest[nTrainTest -1:-1]]).reshape(-1, 1)

    return Xtrain, Ytrain, Xtest, Ytest

def coefficientInit(Xtrain, Ytrain):
    """
    [Description]
        Retourne une première approximation du coefficient du modèle.

    Paramètres
    ----------
    Xtrain, Ytrain : np.array
        Les tableaux permettant d'effectuer une première approche du coefficient de la régression
    

    Returns
    -------
    theta : np.array
        Coefficient de la régression initialisé

    Raises
    ------

    """

    if len(Xtrain[0]) == 1:
        a = (Ytrain[-1]-Ytrain[0])/(Xtrain[-1]-Xtrain[0])
        b = Ytrain[0] - a*Xtrain[0]
        theta = np.array([a, b])
    else :
        n = len(Xtrain[0])
        Xones = ecritureMatricielleSysEqu(Xtrain[0:n, :])
        XtrainPlus = np.linalg.pinv(Xones)
        theta = XtrainPlus.dot(Ytrain[0:n]) + (np.identity(n+1) - XtrainPlus.dot(Xones)).dot(np.random.rand(n+1, 1))

    return theta

def fonctionCout(Xtrain, Ytrain, theta):
    """
    [Description]
        Retourne la valeur du coût d'une étape de la descente de gradient.

    Paramètres
    ----------
    Xones, Ytrain : np.array
        Les tableaux permettant d'effectuer une première approche du coefficient de la régression
    theta :
        Coefficient de la régression.

    Returns
    -------
    cout : float
        Coût de la régression.

    Raises
    ------

    """
    Xones = ecritureMatricielleSysEqu(Xtrain)
    n = len(Xones)
    cout = 0
    erreurs = (Xones.dot(theta) - Ytrain)**2

    for erreur in erreurs:
        cout += erreur

    cout = 1/(2*n)*cout

    return cout

def ecritureMatricielleSysEqu(Xtrain):
    """
    [Description]
        Retourne la forme des caractéristiques permettant d'effectuer les calculs matriciels.

    Paramètres
    ----------
    Xtrain : np.array
        Le tableau à convertir pour effectuer les calculs matriciels.

    Returns
    -------
    Xones : np.array
        Tableau pour effectuer les calculs matriciels, avec une colonne de 1.

    Raises
    ------

    """
    
    ones = np.array([1]*len(Xtrain)).reshape(-1, 1)
    Xones = np.append(Xtrain, ones, axis=1)

    return Xones

def ecritureXPoly(X, degre):
    """
    [Description]
        Retourne la matrice des caractéristiques associée à la modélisation d'un problème polynomial.

    Paramètres
    ----------
    X : pd.DataFrame
        Le dataframe à convertir pour effectuer la régression polynomiale.
    degre : int
        La valeur du degré de la régression polynomiale.

    Returns
    -------
    Xpoly : pd.DataFrame
        Dataframe pour effectuer la régression polynomiale.

    Raises
    ------

    """

    Xpoly = pd.DataFrame()

    while degre >= 0:
        nomColonne = str(degre)

        Xdegre = [x**degre for x in X.values]
        Xpoly[nomColonne] = Xdegre
        degre -= 1

    return Xpoly

def gradient(Xtrain, Ytrain, theta):
    """
    [Description]
        Retourne la valeur du gradient.

    Paramètres
    ----------
    Xtrain, Ytrain : np.array
        Le tableau à convertir pour effectuer la régression polynomiale.
    theta : np.array
        Le coefficient de la régression.

    Returns
    -------
    dtheta : np.array
        La valeur du gradient.

    Raises
    ------

    """

    n = len(Xtrain)
    Xones = ecritureMatricielleSysEqu(Xtrain)
    dtheta = 1/n * (np.transpose(Xones)).dot(Xones.dot(theta)-Ytrain)

    return dtheta

def descenteGradient(X, Y, theta, alpha, n):
    """
    [Description]
        Retourne le coefficient final du modèle et la liste des coûts associés à chaque étape.

    Paramètres
    ----------
    X, Y : np.array
        Les tableaux pour effectuer la descente de gradient.
    theta : np.array
        Le coefficient de la régression.
    alpha : float
        Le coefficient d'apprentissage.
    n :
        Le nombre d'itération.
    
    Returns
    -------
    theta : np.array
        Le coefficient de la régression recalculé.

    Raises
    ------

    """

    dtheta = 0
    cout = 0
    couts = []

    while n > 0:
        dtheta = gradient(X, Y, theta)
        theta = theta - alpha * dtheta
        cout = fonctionCout(X, Y, theta)
        couts.append(cout)
        n -= 1


    return theta, couts

def modele(X, Y, alpha, n, modele, degre):
    """
    [Description]
        Affiche la fonction de coût et retourne l'ensemble des valeurs permettant de définir un modèle prédictif.

    Paramètres
    ----------
    X, Y : pd.DataFrame
        Les tableaux utilisé pour créer le modèle.
    alpha : float
        Le coefficient d'apprentissage.
    n :
        Le nombre d'itération.
    modele : str
        Le nom de la régression à appliquer.
    degre :
        Le degré de la régression si le modèle utilisé est le polynomial.
    
    Returns
    -------
    Xtrain, Ytrain, Xtest, Ytest : np.array
        Les tableaux des valeurs découpées.
    theta : np.array
        Le coefficient de la régression recalculé.

    Raises
    ------

    """

    global dataName

    if modele == "polynomial":
        degre = int(degre)
        X = ecritureXPoly(X, degre)

    Xtrain, Ytrain, Xtest, Ytest = decoupageDonneesAleatoires(X, Y, 0.8)

    theta = coefficientInit(Xtrain, Ytrain)
    
    theta, couts = descenteGradient(Xtrain, Ytrain, theta, alpha, n)

    plt.plot(couts)
    plt.title("Coût en fonction des itérations")
    plt.xlabel("Itérations")
    plt.ylabel("Coût")
    nomFigure = f"{dataName} - alpha{str(alpha)} - N{n} - Fonction coût"
    plt.savefig(nomFigure.replace(".", "_"))
    plt.show()

    return Xtrain, Ytrain, Xtest, Ytest, theta

def prediction(Xtest, theta):
    """
    [Description]
        Retourne le vecteur prédit à partir des valeurs résultant d'une régression.

    Paramètres
    ----------
    Xtest : np.array
        Les tableaux utilisé pour effectuer la prédiction.
    
    Returns
    -------
    Ypred : np.array
        Le tableau des prédictions.

    Raises
    ------

    """

    return ecritureMatricielleSysEqu(Xtest).dot(theta)

def coeffDetermination(Ytest, Ypred):
    """
    [Description]
        Retourne la valeur du coefficient de Pearson.

    Paramètres
    ----------
    Ytest, Ypred : np.array
        Les tableaux utilisé pour évaluer le modèle.
    
    Returns
    -------
    R : float
        Le coefficient de détermination du modèle.

    Raises
    ------

    """
    n = len(Ytest)

    sommeVariance = 0
    moyenneCarreResiduts = 0
    yMean = np.mean(Ytest)

    for i in range(n):
        sommeVariance += (Ytest[i]-Ypred[i])**2
        moyenneCarreResiduts += (Ytest[i]-yMean)**2

    R2 = 1 - sommeVariance/moyenneCarreResiduts 

    return R2   

def errQuadMoyenne(Ytest, Ypred):
    """
    [Description]
        Retourne la valeur de l'erreur quadratique moyenne.

    Paramètres
    ----------
    Ytest, Ypred : np.array
        Les tableaux utilisé pour évaluer le modèle.
    
    Returns
    -------
    mSE : float
        L'erreur quadratique moyenne de la prédiction.'

    Raises
    ------

    """
    mSE = mean_squared_error(Ytest, Ypred)

    return mSE

if __name__ == "__main__":
    alpha = 5e-1
    n = 100
    degre = 6

    '''
    Décommentez la ligne correspondante aux données souhaitées.
    '''
    dataName = "reg_simple.csv"
    # dataName = "boston_house_prices.csv"
    # dataName = "Position_Salaries.csv"

    dataNamePath = PATH + dataName
    X, Y = chargementDonnees(dataNamePath)
    X = pretraitementDonnees(X)
    Y = pretraitementDonnees(Y)

    '''
    Décommentez la ligne associée au modèle souhaité.
    '''
    Xtrain, Ytrain, Xtest, Ytest, theta = modele(X, Y, alpha, n, "simple linéaire", degre)
    # Xtrain, Ytrain, Xtest, Ytest, theta = modele(X[["RM", "LSTAT"]], Y, alpha, n, "multiple linéaire", degre)
    # Xtrain, Ytrain, Xtest, Ytest, theta = modele(X["Level"], Y, alpha, n, "polynomial", degre)

    Ypred = prediction(Xtest, theta)

    '''
    Décommentez ce paragraphe pour afficher les résultats du premier modèle
    '''
    plt.scatter(Xtrain, Ytrain)
    plt.scatter(Xtest, Ytest)
    plt.plot(X, prediction(X, theta))
    plt.title("Note en fonction du temps passé à réviser")
    plt.xlabel("Temps passé à réviser")
    plt.ylabel("Note obtenue")
    plt.legend(["entraînement", "test", "prédiction"])

    '''
    Décommentez ce paragraphe pour afficher les résultats du deuxième modèle
    '''
    # ax = plt.axes(projection="3d")
    # ax.scatter(Xtest[:, 0], Xtest[:, 1],  Ytest)
    # ax.scatter(Xtest[:, 0], Xtest[:, 1],  Ypred)
    # ax.set_title("Régression multiple")
    # ax.set_xlabel("RM")
    # ax.set_ylabel("LSTAT")
    # ax.set_zlabel("MEDV")
    # ax.view_init(45, 180)
    # plt.legend(["test", "prédiction"])
    

    '''
    Décommentez ce paragraphe pour afficher les résultats du troisième modèle
    '''
    # plt.scatter(Xtrain[:, -2],  Ytrain)
    # plt.scatter(Xtest[:, -2], Ytest)
    # plt.plot(X["Level"].values.reshape(-1,1), prediction(ecritureXPoly(X["Level"], degre), theta))
    # plt.title(f"Régression polynomiale de degré {degre}")
    # plt.xlabel("Level")
    # plt.ylabel("Salary")
    # plt.legend(["entraînement", "test", "prédiction"])

    '''
    Si vous souhaitez enregistrer une figure, n'oubliez pas de décommenter ce paragraphe
    '''
    # nomFigure = f"{dataName} - alpha{str(alpha)} - N{n} - Prédiction"
    # plt.savefig(nomFigure.replace(".", "_"))
    # plt.show()

    '''
    Si vous souhaitez afficher les coefficients d'erreur, n'oubliez de décommenter ce paragraphe'''
    # R2 = coeffDetermination(Ytest, Ypred)
    # print("MSE=" + str(errQuadMoyenne(Ytest, Ypred)))
    # print("R2=" + str(R2))