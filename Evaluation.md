# Objectifs et évaluation --> 18/01/2022

## Objectifs :

1. (obligatoire) Implémenter en C++-20
    1. en utilisant la bibliothèque standard,
    2. sans utiliser de bibliothèques externes,
    3. sur votre machine et sur la cible embarquée (dans notre cas Raspberry Pi 4).
2. (obligatoire) Veiller à donner la preuve que le code fonctionne sur la cible.
3. (obligatoire) Évaluer (et mesurer) les complexités temporelle et mémoire de vos algorithmes.
4. (obligatoire) Vérifier la performance de chaque classificateur embarqué sur les fichiers de test sélectionnés avant l'entraînement afin de les comparer.
5. Programmer une extraction de paramètres en C++ performante :
    1. (obligatoire) compléter le fichier au_reading.h afin de [convertir les fichiers AU du dataset en DataVector en C++](https://en.wikipedia.org/wiki/Au_file_format).
    2. (obligatoire) efficace dans son exécution (1 Go de données musicales peuvent être traitées en 60 secondes ou moins),
    3. minimale en termes de paramètres significatifs générés pour l'apprentissage :
        1. (obligatoire) approche STFT,
        2. (facultatif) approche MFCC.
6. Programmer des arbres de décision :
    1. (obligatoire) générer automatiquement le code C++ à compiler pour exécuter la prise de décision sur la cible depuis le CART Python fourni ou Skearn,
    2. (facultatif) proposer une implémentation de CART en C++-20 à partir de celle proposée en Python,
    4. (facultatif) proposer une implémentation simple de l'exécution d'une Random Forest en C++-20 en se servant des codes générés automatiquement.
7. Programmer une SVM :
    1. (obligatoire) Élaborer une SVM optimale à l'aide de Scikit Learn, à partir des paramètres extraits en C++,
    2. (obligatoire) Implémenter la prédiction *un contre un* en C++-20 sur la cible.
8. Programmer un réseau de neurones :
    1. (obligatoire) Implémenter un ANN en C++,
    2. (facultatif) Implémenter un ANN en C++ en utilisant le paradigme objet,
9. (facultatif) Comparer vos implémentations avec TensorFlow sur Raspberry Pi,
10. (obligatoire) Comparer les approches et faire un tableau synthétique des caractéristiques et des performances de chacune des approches (CART, RF, SVM, ANN).

## Évaluation pour le 18/01/2022 (dépôt Moodle) :

- Rapport au format PDF police 11 de 10 pages maximum qui rend compte précisément de votre travail.
- Le code source (C++/Python/Matlab) 
