# Préparer les questions individuelles de la séance finale

**Pourquoi un CNN ?** Parce qu'il applique des filtres locaux partagés sur toute l'image. Il garde la structure spatiale et possède généralement moins de paramètres qu'un réseau entièrement connecté sur les pixels.

**Que fait la convolution ?** Un filtre glisse sur l'image. À chaque position, il calcule un produit scalaire avec les pixels concernés, puis ajoute un biais. Chaque filtre produit une carte d'activation.

**Que signifie `padding="same"` ?** Avec une convolution 3×3 de stride 1, on ajoute suffisamment de pixels autour de l'entrée pour conserver les dimensions spatiales.

**Pourquoi ReLU ?** Pour introduire de la non-linéarité : `ReLU(x) = max(0, x)`. Sans activations non linéaires, plusieurs couches linéaires se réduiraient à une transformation linéaire.

**À quoi sert le pooling ?** Il réduit les dimensions des cartes de caractéristiques. Un max-pooling 2×2 de stride 2 divise normalement largeur et hauteur par deux.

**Combien de paramètres dans la première convolution ?** Pour 32 filtres de taille 3×3 et une image RGB : `(3×3×3 + 1)×32 = 896`, en comptant les biais.

**Pourquoi six sorties ?** Parce que nous avons six catégories exclusives. Softmax convertit les scores en probabilités dont la somme vaut un.

**Pourquoi la cross-entropy ?** Elle pénalise une probabilité faible attribuée à la bonne catégorie. L'optimiseur cherche à minimiser cette perte.

**Comment fonctionne la rétropropagation ?** Après la passe avant, elle applique la règle de la chaîne aux différentes couches pour obtenir les gradients des paramètres par rapport à la perte.

**Quel est le rôle du learning rate ?** Il règle l'amplitude des mises à jour. Trop grand, l'entraînement peut être instable ; trop faible, il peut progresser très lentement.

**Qu'est-ce que l'overfitting ?** Le modèle s'adapte excessivement à ses exemples d'entraînement. On peut le détecter si la perte d'entraînement continue de baisser tandis que celle de validation augmente.

**Pourquoi l'augmentation de données ?** Elle augmente la diversité des exemples vus en entraînement tout en conservant normalement leur étiquette.

**Pourquoi le dropout ?** Il désactive aléatoirement des activations pendant l'entraînement et peut limiter le surapprentissage.

**Quelle différence entre transfer learning et fine-tuning ?** Le transfer learning réutilise un modèle préentraîné. Dans E4, le backbone est figé ; dans E5, une partie de ses poids est aussi ajustée.

**Pourquoi ne pas normaliser MobileNetV2 en `[0,1]` ?** Le prétraitement attendu par ce backbone transforme les pixels en `[-1,1]`.

**Pourquoi garder BatchNormalization en inférence pendant le fine-tuning ?** Pour éviter de modifier de manière incontrôlée ses statistiques internes, apprises pendant le préentraînement.

**Pourquoi séparer validation et test ?** La validation sert à comparer les choix du projet. Le test sert à estimer les performances finales sur des images non utilisées pour ces choix.

**Pourquoi le macro-F1 ?** Il calcule la moyenne du F1 de chaque catégorie. Chaque classe compte de la même manière.

**Le transfert doit-il forcément gagner ?** Non. Ses résultats dépendent notamment de la proximité entre les images d'origine et celles du nouveau dataset, de la taille de ce dernier et du réglage de l'entraînement.

**Qu'est-ce qui garantit la reproductibilité ?** Un découpage fixe enregistré dans un CSV, une graine aléatoire, les configurations documentées et les checkpoints sauvegardés. Les résultats peuvent tout de même varier légèrement selon le matériel et les bibliothèques.
