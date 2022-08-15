```
── datasets								//Données
│   ├──  music                 				// Fichier stockant la database
|   |   └── ...         					
|   ├── music_features_mfcc_test.csv      	// MFCC pour test
|   ├── music_features_mfcc_train.csv   	// MFCC pour train
|   ├── music_features_stft_test.csv		// STFT pour train
|	└── music_features_stft_train.csv		// STFT pour train
├── embedded_implementation
│   ├── demo							// main pour chaque partie
│	│	├── artificial_neural_network_demo.cpp //ANN
│	│	├── decision_tree_demo.cpp			// CART
│	│	├── extractor_demo.cpp				// Extractions
│	│	├── one_vs_one_demo.cpp				// SVM
│	│	└── random_forest_demo.cpp			// RF
│	├── extraction					    // Objet pour l'extraction
│	│	├── au_file_procssor.cpp			// Extractions
│	│	└── au_file_processor.h				// Extractions
│	├── helpers								// Nombreuses fonctions utiles
│	│	└── ...								
│	└── ml_algorithms					// Objet pour les modéles de ML
│		├── artificial_neural_network.cpp	// ANN 
│		├── artificial_neural_network.h		// ANN 
│		├── decision_tree.cpp				// CART 
│		├── decision_tree.h					// CART 
│		├── machine_learning_model.h		// Base des objets
│		├── one_vs_one_svm.cpp				// SVM 
│		├── one_vs_one_svm.h				// SVM
│		├── random_forest.cpp				// RF
│		└── random_forest.h					// RF 
└── training							// Entrainement python et modéles
    ├── artificial_neural_network			// ANN 
    |   └── ... 
 	├── decision_tree						// CART 
 	│	└── ...
	├── random_forest						// RF 
	│	└── ...
	└── support_vector_machine				// SVM 
		└── ...
```