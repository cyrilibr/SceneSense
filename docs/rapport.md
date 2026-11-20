# SceneSense : classification de scènes avec CNN et transfert d'apprentissage

## 1. Présentation du projet

Ce projet étudie la reconnaissance de scènes naturelles à partir d'images. L'objectif est de classer une photographie dans l'une des six catégories du dataset Intel : bâtiments, forêt, glacier, montagne, mer ou rue. Nous cherchons surtout à comprendre ce que changent les différentes méthodes vues en cours. Un bon score seul ne suffit pas : il faut regarder si le modèle apprend correctement et s'il sait reconnaître des photos qu'il n'a jamais vues.

Nous avons organisé le travail en cinq expériences. Les trois premières portent sur un CNN construit par nos soins. Les deux dernières utilisent MobileNetV2, un réseau déjà entraîné sur ImageNet. Cela permet de comparer un apprentissage complet et la réutilisation de caractéristiques visuelles existantes.

## 2. Données et préparation

Le dataset Intel Image Classification contient des images RGB de paysages naturels et de scènes urbaines. Il possède six dossiers de classes. Nous utilisons la partie `seg_train` pour constituer un ensemble d'apprentissage et un ensemble de validation, puis nous gardons `seg_test` pour la fin du projet.

Nous séparons 90 % des images de `seg_train` pour l'entraînement et 10 % pour la validation, de manière stratifiée. Cela permet de garder une représentation de chaque catégorie. Le découpage est sauvegardé dans `results/splits.csv` avec une graine fixée à 42. Tous les modèles utilisent donc exactement les mêmes photos dans chaque ensemble. Les images sont redimensionnées en 160 × 160 pixels et chargées par lots de 32.

Une attention particulière est portée à la normalisation. Le CNN prend des pixels entre 0 et 255 et les divise par 255 dans sa première couche. MobileNetV2 attend une entrée comprise entre -1 et 1 : la normalisation utilisée est donc `x / 127.5 - 1`. Il serait incorrect de réutiliser la même normalisation pour les deux architectures sans tenir compte de celle attendue par le modèle préentraîné.

## 3. Le CNN développé

L'architecture comporte trois convolutions de taille 3 × 3, respectivement avec 32, 64 et 128 filtres. Après chaque convolution, nous utilisons une fonction ReLU et un max-pooling 2 × 2. Le pooling réduit progressivement les dimensions des cartes de caractéristiques. Une couche GlobalAveragePooling transforme les cartes en un vecteur compact. Ce vecteur passe dans une couche dense de 128 neurones et dans une dernière couche dense de six neurones avec Softmax.

Le CNN apprend grâce à la perte *sparse categorical cross-entropy*. Le modèle calcule d'abord une prédiction, puis la différence entre les probabilités obtenues et la vraie étiquette. La rétropropagation sert ensuite à calculer les gradients à l'aide de la règle de dérivation en chaîne. L'optimiseur Adam met à jour les poids. Le taux d'apprentissage initial est fixé à 0,001. La validation est suivie à chaque époque pour détecter le surapprentissage.

## 4. Régularisation et expériences CNN

Dans E1, nous entraînons le CNN sans augmentation de données ni dropout. Ce premier essai sert de référence.

Dans E2, nous conservons l'architecture de E1 et ajoutons trois transformations aléatoires : retournement horizontal, petite rotation et zoom limité. Ces transformations n'ont lieu que pendant l'entraînement ; la validation et le test utilisent les photos sans ces modifications. Le but est d'encourager le modèle à reconnaître la scène même si son cadrage change légèrement.

Dans E3, nous reprenons E2 en ajoutant une couche de dropout de 0,30 juste avant la classification. Pendant l'entraînement, elle désactive aléatoirement une partie des activations. L'idée est d'éviter que le modèle s'appuie systématiquement sur les mêmes neurones. Nous comparons E2 et E3 pour étudier cet effet, sans prétendre que le dropout garantit une amélioration.

Nous utilisons aussi l'early stopping pour interrompre un entraînement dont la perte de validation ne s'améliore plus. Le meilleur checkpoint est conservé. Cela évite de retenir automatiquement la dernière époque, qui n'est pas forcément la meilleure.

## 5. Transfer learning

Dans E4, nous chargeons MobileNetV2 avec ses poids ImageNet, sans sa dernière couche de classification. Les couches convolutives existantes sont figées ; nous entraînons uniquement notre classifieur composé d'un global average pooling, d'un dropout et de six sorties Softmax. Le transfert d'apprentissage est particulièrement intéressant lorsque l'on souhaite réutiliser des caractéristiques déjà apprises, comme les bords, textures et formes.

Dans E5, nous repartons du meilleur checkpoint de E4. Nous autorisons l'entraînement des dernières couches du backbone, sauf les couches BatchNormalization, qui restent figées. Le backbone est appelé en mode inférence pour préserver le comportement des statistiques de normalisation. Nous recompilons le modèle avec un taux d'apprentissage beaucoup plus faible, 0,00001, afin d'ajuster les poids sans modifier brutalement les caractéristiques précédemment apprises.

## 6. Méthode de comparaison

Les cinq expériences partagent le même découpage et la même taille d'entrée. Pour chaque essai, le code conserve le meilleur modèle selon la perte de validation, un historique de l'entraînement et un fichier décrivant les paramètres de l'expérience.

Nous privilégions la perte de validation pour sélectionner le modèle final, sans regarder les résultats du test à ce stade. Une fois le choix fait, le modèle sélectionné est évalué sur `seg_test`. Nous calculons l'accuracy, le F1 macro et la matrice de confusion. Le F1 macro donne le même poids à chacune des six classes, ce qui évite de juger le modèle uniquement à partir des catégories les plus nombreuses.

L'analyse des erreurs est importante. Par exemple, une image montrant un sommet enneigé pourrait être confondue avec un glacier. Le fichier des prédictions indique, pour chaque photographie, sa vraie classe, sa prédiction et la confiance du modèle. Ces exemples permettent de discuter les limites sans se contenter d'un pourcentage global.

## 7. Ce que le projet permet de conclure

Le protocole permet de répondre à trois questions : l'augmentation de données améliore-t-elle la validation ? Le dropout aide-t-il le CNN à généraliser ? Et le transfert d'apprentissage donne-t-il un avantage suffisant pour justifier l'utilisation d'un réseau préentraîné ? La réponse doit venir des résultats mesurés, pas d'une hypothèse formulée à l'avance.

**Statut de la livraison :** les architectures, scripts, tests, notebooks, documentation et diapositives sont préparés. L'entraînement sur les images Intel et la mesure de ses résultats ne sont pas effectués dans cette livraison, car le dataset et le GPU ne sont pas disponibles dans l'environnement d'exécution. Le dépôt ne contient pas de scores fictifs. Les commandes du README produisent les fichiers de résultats lorsque l'entraînement est exécuté.

## Sources

1. Supports de cours fournis : Introduction to Deep Learning (CH1), Backpropagation (CH2), Regularization and Optimization in Deep Learning (CH3), Convolutional Neural Networks (CH4).
2. Intel Image Classification : https://www.kaggle.com/datasets/puneet6060/intel-image-classification
3. Guide TensorFlow : https://www.tensorflow.org/guide/keras/transfer_learning
4. MobileNetV2 : https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2
