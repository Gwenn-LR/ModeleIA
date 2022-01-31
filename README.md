# Premier Modèle IA (Régression linéaire simple, multiple et polynomiale)

## Sommaire
1. [TODO](#todo)
   1. [Régression linéaire simple](#1)
   2. [Régression linéaire multiple](#2)
   3. [Régression poly](#3)
   4. [Scikit-Learn](#4)

#

## <div id="todo">TODO:</div>
- <div id="1"> Régression linéaire simple</div>


  - [x] Récupération des données
    - Importation de pandas
    - Chargement des fichiers
    - Découpage des données en test/train
  - [x] Visualisation des données
    - Importation de matplotlib
    - Affichage de la courbe
  - [x] Création du modèle (model(X,theta) )
    - Création de la fonction modeleLinSimple(X, theta)
    - Création de la fonction regression(X, Y, alpha, n_iterations)
  - [x] Fonction du coût (fonction_cout(X,Y,theta))
    - Création de la fonction produitMatriciel
    - Création de la fonction cout
  - [x] Gradient (gradient(X,Y,theta))
    - Création de la fonction transposee
    - Création de la fonction gradient
  - [x] Descente du gradient (descente_gradient(X,Y,theta,alpha,n_iterations))
    - Création de la fonction descenteGradient
    - Récurrence
  - [x] Evaluer votre modèle en utilisant le coefficient de détermination
    - Récupération des valeurs à tester
    - Création de la fonction coefDetermination(X_train, Y_train)
  - [x] Tracer la courbe de la fonction du coût selon les itérations
    - Création fonction coutIterations
    - Affichage selon plusieurs valeurs

>
- <div id="2"> Régression linéaire multiple</div>

  - [x] Implémentez un modèle de régression multiple sur la base de données issue du fichier nommé boston_house_prices.csv (sans utiliser la bibliothèque Scikit-learn).
     - Chargement des fichiers
     - Découpage des données en test/train
     - Affichage de la courbe
     - ~~Création de la fonction modeleLinMulti()~~
     - Utilisation de la fonction de création de modèle
  - [x] Évaluez les résultats obtenus en utilisant la fonction mean_squared_error de sklearn
    - Importation de sklearn
    - Création de la fonction errQuadMoyenne(X_train, Y_train)


>
- <div id="3"> Régression polynomiale</div>


  - [x] En utilisant les bibliothèques adéquates de Python, implémentez un modèle de régression polynomiale sur le jeu de données issu du fichier **Position_Salaire.csv **(sans utiliser la bibliothèque Scikit-learn).
     - Chargement des fichiers
     - Découpage des données en test/train
     - Affichage de la courbe
     - ~~Création de la fonction modelePoly()~~
     - Utilisation de la fonction de création de modèle
  - [x] Appliquez le même modèle sur le jeu de données issu du fichier data/qualite_vin_rouge.csv
    - Utilisation de la fonction modelePoly()
  - [x] Évaluez votre modèle.
    - Utilisation de la fonction coeffDetermination
    - Utilisation de la fonction errQuadMoyenne

>
- <div id ="4"> Scikit-Learn</div>


  - [ ] Refaire les 3 régressions avec le module Scikit-Learn
    - Regression linéaire simple
    - Regression linéaire multiple
    - Regression polynomiale
  - [ ] Comparez les résultats de prédiction avec la méthode normale
    - Regression linéaire simple
    - Regression linéaire multiple
    - Regression polynomiale
</div>
