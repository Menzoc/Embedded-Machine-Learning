[[_TOC_]]

# Embedded Machine Learning

Ce projet GitLab est la base du code que vous devez rendre sur Moodle. Il faut en respecter l'esprit et l'organisation.

# Embarquer la prédiction issue de l'apprentissage automatique ?

L'idée fondamentale de ce cours est de poser la problématique
de l'**utilisation de l'apprentissage automatique sur une cible quelconque** (ordinateur, smartphone, carte avec ou sans OS, FPGA).

- Peut-on embarquer des algorithmes de prédiction issus de l'apprentissage automatique ?
- Si oui, comment faire ?
- Quels sont les éléments nécessaires à la prédiction embarquée ?
- Peut-on les rendre minimaux en termes de ressources (puissance de calcul, mémoire, énergie) ?

Ce cours tente d'apporter des réponses à ces questions en apparence simples.

La question de la minimalité des ressources utilisées pour exécuter un calcul rejoint la problématique de la consommation d'énergie (fabrication, fonctionnement et recyclage) et donc celle du développement durable.  

D'une manière générale, moins on fait appel aux appels systèmes d'un OS, moins on fait appel à des bibliothèques, moins on utilise de ressources, plus le système est minimal. Encore faut-il pouvoir garantir son efficacité...

Au niveau des algorithmes, il faudrait pouvoir isoler uniquement les éléments nécessaires à l'exécution de l'IA. Est-ce possible ? Il faut alors s'interroger sur les étapes nécessaires selon la nature de l'algorithme d'apprentissage choisi :

- Faut-il privilégier une approche algorithmique classique ou une approche de type apprentissage supervisé ou non supervisé ?
- Peut-on distinguer des phases dans ce processus (formatage des données d'entrée ou de sortie, extraction de paramètres, entraînement, prédiction) et peut-on les isoler ? Doit-on toutes les implémenter sur la cible ?
- Peut-on schématiser ce flux de données et préciser les interfaces nécessaires aux calculs ?

Au niveau des ressources nécessaires, on peut s'interroger :

- A-t-on besoin de langages de haut niveau interprétés ?
- Faut-il un OS ?
- Faut-il un processeur capable ?  
- Quelle puissance de calcul est requise ?

Une réponse simple à toute cette problématique est [TensorFlow Lite](https://www.tensorflow.org/lite/guide), le moteur d'interprétation de modèle de TensorFlow optimisé pour l'embarqué.  La liste de ces avantages est fascinante :

- compatibilité matérielle extrême (Android, iOS, Linux, microcrontôleurs),
- interfaçage logiciel immense (Java, C++, Python...),
- possibilité d'accélération matérielle,
- l'interpréteur de modèle est un exécutable de taille infiniment optimisée (~1 Mo).

**Alors pourquoi réinventer la roue ?**

- Pour comprendre en profondeur comment tout ceci fonctionne,

- S'affranchir des dépendances qui impliquent des évolutions pas nécessairement souhaitées à vos produits et parfois des (re)qualifications coûteuses,

- Maîtriser entièrement sa conception et son développement logiciel.
  
  Les conséquences de ces choix militants :

- création de compétences qui permettent de faire évoluer les entreprises,

- acquisition de souveraineté des produits,

- création de valeur.

# Contexte d'apprentissage

Pour illustrer ces problématiques, un contexte principal a été choisi : **la reconnaissance de styles musicaux**.

Il s'agit de pouvoir prédire, d'après un extrait musical de 30 secondes, le style de musique de celui-ci.

Nous disposons pour cela de plusieurs bases de données. La plus simple à utiliser, même si celle-ci n'est pas parfaite, est la collection GTZAN disponible sous différents formats (.au ou .wav). L'ensemble de données se compose de 1000 pistes audio de 30 secondes, de 10 styles différents, chacun représenté par 100 pistes. Les pistes sont toutes des fichiers audio monocanal, échantillonnés sur 16 bits à la fréquence de 22050 Hz.

Les styles musicaux sont les suivants :
0. blues
1. classique
2. country
3. disco
4. hiphop
5. jazz
6. metal
7. pop
8. reggae
9. rock

# Standards

Ce projet utilise nécessairement :

- un compilateur C++ à la norme 20,
- cmake (>=3.16),
- Python 3.9 et Scikit Learn.

CMake est utilisé pour construire les exécutables du projet. L'utilisation d'un compilateur local, d'un cross-compilateur ou d'un compilateur distant (sur la cible) est à paramétrer dans les préférences de CLion directement (Build > Toolchains).
On pourra ajouter dans chaque répertoire le code Python nécessaire au projet. Par exemple, dans le répertoire SVM, on peut ajouter un répertoire Python qui contient le code pour calculer les coefficients d'une SVM optimale, adaptée au problème.

**Il est très vivement recommandé d'utiliser CLion comme IDE et gcc comme compilateur.**.

# Cible

Nous avons choisi de vous faire travailler sur Raspberry PI 4, avec un OS Linux 64 bits et avec le standard C++-20.

Cette cible a le mérite d'être commode, peu onéreuse et malgré tout pertinente pour notre étude car le processeur est un Broadcom BCM2711, Quad core Cortex-A72 (ARM v8) 64-bit SoC @ 1.5GHz.
Le standard C++-20 permet d'aborder dans un cadre sûr et efficace les problématiques de calcul numérique.
Linux sera juste pratique dans le cadre de nos exercices et pourrait être supprimé des ressources nécessaires pour peu qu'on mette les mains dans le cambouis ;)

Il sera possible d'ajouter à ces éléments TensorFlow Lite et d'autres outils dédiés à l'IA et l'embarqué.

[L'image du système de la RPI4 se trouve ici !](https://filesender.renater.fr/?s=download&token=769beef2-7287-456f-a439-5c09e886e210).

# Compilation du projet

__Créer et se déplacer dans un dossier pour le build :__
```
mkdir build
cd build
```
__Configurer le projet :__
- Pour une architecture x86
    ```
    cmake env WORKING_DIR=./ ..
    ```
- Pour une architecture arm 64bit
    ```
    cmake --DCMAKE_C_COMPILER=/usr/bin/aarch64-linux-gnu-gcc-11 -DCMAKE_CXX_COMPILER=/usr/bin/aarch64-linux-gnu-g++-11 env WORKING_DIR=./ ..
    ```
> la variable d'environnement `WORKING_DIR` ( __/!\\__ ne pas oublier le slash à la fin __/!\\__) permet aux projets de savoir où se trouvent les différents csv, par exemple, le fichier csv d'entrainement via SFTF se trouve au chemin `${WORKING_DIR}datasets/music_features_stft_train.csv`.   
> La valeur par défaut est `../../../` pour simplifier le fonctionnement lors de la compilation via Clion

__Compiler :__
```
make
```

> __Les paramètres des modèles ayant été générés avec les données présente dans la branche master, pensez à bien tester les projets CART, RF, SVM et ANN avec ces même données pour avoir la même matrice de confusion qu'à l'entrainement!       
> En effet, une execution du projet EXTRACTION génerera un nouveau batch de données train/test différent de celui utilisé en entrainement et aura donc des prédictions faussés (présence potentielle de donnée utilisées en entrainement dans le batch de test ce qui augmentera la précision par exemple)__

# Répertoires

## datasets

Le dossier `datasets` contient les ensembles de données utilisées pour apprendre et tester les algorithmes implémentés.
Pour le dépôt Git, on n'a pas gardé tous les extraits musicaux, afin de na pas encombrer inutilement les disques.
[À vous de télécharger les données complètes ici au format AU.](http://marsyas.info/downloads/datasets.html)

## docker
Dossier contenant le dockerfile utilisé pour compiler le projet pour la carte raspberry-pi.

Pour compiler le projet en l'utilisant :
- `docker build -t clion/ubuntu/aarch64-toolchain:1.0 -f toolchain.dockerfile`
- Puis, il faut ajouter une toolchain Docker dans Clion (`settings/Build, Execution, Deployment/Toolchains`)
- Enfin, il faut ajouter une configuration Cmake dans Clion (`settings/Build, Execution, Deployment/Cmake`) et ne pas oublier de définir la variable d'environnement `WORKING_DIR` ( __/!\\__ ne pas oublier le slash à la fin __/!\\__).

## embedded_implementation
Les fichiers présents dans ce dossier contiennent la partie implémentation sur la cible embarqué en langage C++20.
### demo
Le dossier `demo` contient différents main qui ont pour but de tester chaque partie du projet lors du développement :
- **extractor_demo.cpp** sépare tous les fichiers au format `.au` du dossier `datasets/music` en données de test en d'entrainement, traite selon différent algorithme (**STFT** ou **MFCC**) et produit les fichier de paramètres au format CSV (dans `datasets/music_features_<algorithme>_<dataset>.csv`) formaté avec entêtes.
  
- **decision_tree_demo.cpp** récupères les features vector du fichier CSV créé dans `extractor_demo.cpp`, récupère les paramètres de l'arbre de décision à partir du fichier CSV créé dans la partie training (algorithme CART) puis créer l'arbre de décision lié et test les prédictions.
- **random_forest_demo.cpp** récupères les features vector du fichier CSV créé dans `extractor_demo.cpp`, récupère les paramètres des arbres de décision à partir des fichiers CSV créé dans la partie training (algorithme random forest) puis créer la forêt d'arbres de décision liée et test les prédictions.
- **one_vs_one_svm_demo.cpp** récupères les features vector du fichier CSV créé dans `extractor_demo.cpp`, récupère les paramètres des classificateurs linéaires à partir du fichier CSV créé dans la partie training (algorithme Machine à Vecteur de Support type One VS One et noyau linéaire) puis créer le modèle SVM et test les prédictions.
- **artificial_neural_network_demo.cpp** récupères les features vector du fichier CSV créé dans `extractor_demo.cpp`, récupère les poids des neurones à partir des fichiers CSV de chaque couche créés dans la partie training (ANN simple avec la fonction d'activation softmax pour la couche de sortie et relu pour les autres) puis créer le modèle ANN et test les prédictions.

### extraction
Le dossier `extraction` contient la définition de la classe pour l'extraction de paramètres d'un fichier au format `.au`.
Celle-ci est nécessaire sur la machine et sur la cible : en effet, pour prédire une classe d'appartenance, le système embarqué doit être capable de générer les paramètres.

L'utilisation de la méthode `get_csv_line()` permet d'obtenir la ligne au format CSV du fichié traité et la fonction static `get_csv_line_header()` permet d'obtenir l'en tête de ce fichier.   
Selon la méthode d'extraction utilisée, chaque colonne est nommée et le nom désigne un paramètre particulier.

Pour la méthode **stft** (fichier `"./dataset/music_features_stft_<dataset>.csv"`) la convention de nommage est la suivante :
- `BIN_AVG3` est la moyenne du bin 3 de la STFT pratiquée sur un échantillon de musique.
- `BIN_STDEV252` est l'écart type du bin 252 de la STFT pratiquée sur un échantillon de musique.
- `style` est le nom du style de musique associé au fichier de la ligne
- `file_name` est le path vers le fichier de musique associé à la ligne
> par exemple :
> ```
> BIN_AVG0,BIN_AVG0,BIN_AVG1,BIN_AVG2,...,BIN_STDEV252,BIN_STDEV253,BIN_STDEV254,BIN_STDEV255,style,file_name
> 0,0,0,0,0,0,0,0,...0,0,0,0,0,"blues","../../datasets/music/blues/blues.00000.au"
> ```
Pour la méthode **mfcc** (fichier `"./dataset/music_features_mfcc_<dataset>.csv"`) la convention de nommage est la suivante :
- `SIGNALENERGY_AVG` est la valeur moyenne de l'énergie du signal
- `BIN_AVG3` est la moyenne du bin 3 de la MFCC pratiquée sur un échantillon de musique.
- `SIGNALENERGY_STDEV`  est l'écart type de l'énergie du signal
- `BIN_STDEV11` est l'écart type du bin 11 de la MFCC pratiquée sur un échantillon de musique.
- `style` est le nom du style de musique associé au fichier de la ligne
- `file_name` est le path vers le fichier de musique associé à la ligne
> par exemple :
> ```
> SIGNALENERGY_AVG,BIN_AVG0,BIN_AVG1,...,BIN_AVG17,BIN_AVG18,SIGNALENERGY_STDEV,BIN_STDEV0,BIN_STDEV1,...,BIN_STDEV17,BIN_STDEV18,style,file_name
> 0,0,0,0,0,0,0,0,...0,0,0,0,0,"blues","../../datasets/music/blues/blues.00000.au"
> ```


### helpers
Le dossier `helpers` contient un ensemble de fonctionnalités utiles au développement :
- **file_helpers.h** Sélectionner des fichiers pour l'entraînement et le test et en garder la trace.
- **globals.h** Définir des variables globales et des types ad-hoc et avoir la possibilité de compiler rapidement en simple ou en double précision.
- **log.h** Simplifier l'affichage des logs avec un système de gravité du message, plus de détails sur l'origine du message et la possibilité de les désactiver simplement.
- **music_style_helpers.h** Manipuler facilement des styles sous la forme d'un énuméré plutôt que des entiers.
- **print_helpers.h** Afficher des vecteurs ou des tableaux facilement.
- **signal.h** Calculer une transformée de Fourier rapide.

### ml_algorithms
Le dossier `ml_algorithms` contient l'implémentation des algorithmes de machine learning en C++. 
Il comprend les fichiers suivants :
- **machine_learning_model.h** et **machine_learning_model.cpp** qui définissent la superclasse abstraite qui sera surchargé par tous les objets liés aux algorithmes de machine learning
- **decision_tree.h** et **decision_tree.cpp** qui définissent les classes d'un noeud et d'un arbre de décision
- **random_forest.h** et **random_forest.cpp** qui définissent la classe d'une forêt d'arbres de décision aléatoire
- **one_vs_one_svm.h** et **one_vs_one_svm.cpp** qui définissent les classes d'un classificateur linéaire et d'une machine à support de vecteur utilisant un noyau linéaire un mode one vs one
- **artificial_neural_network.h** et **artificial_neural_network.cpp** qui définissent les classes d'un neurone et d'un réseau de neurones
## training
Les fichiers présents dans ce dossier contiennent la partie entrainement de chaque type de neurone.    
Cet entrainement est fait dans des fichiers jupyter notebook utilisant un kernel python 3.8 et les librairies suivantes :
- pandas (1.3.4)
- numpy (1.20.2)
- sklearn (1.0.1)
- matplotlib (3.4.0)
- graphviz (0.19.1)
- tensorflow (2.7.0)

### decison_tree
Le dossier `decison_tree` contient les phases d'entrainement d'un modèle de type arbre de décision CART.

### random_forest
Le dossier `random_forest` contient les phases d'entrainement d'un modèle de type forêt d'arbre de décision aléatoire.

### support_vector_machine
Le dossier `support_vector_machine` contient les phases d'entrainement d'un modèle de type séparateur à vaste marge.

### artificial_neural_network
Le dossier `artificial_neural_network` contient les phases d'entrainement d'un modèle de type réseau de neurones artificiel.
