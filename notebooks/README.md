# Ordre d'exécution des notebooks

Ouvrir ces notebooks depuis la racine du projet SceneSense dans Jupyter, ou placer le dossier dans `/content/SceneSense` avec Google Colab. Installer les dépendances et télécharger le dataset en suivant le README principal.

1. `01_dataset.ipynb` : inspection et découpage.
2. `02_custom_cnn.ipynb` : CNN de référence, augmentation, dropout.
3. `03_transfer_learning.ipynb` : MobileNetV2 gelé puis fine-tuning.
4. `04_results.ipynb` : choix final sur validation, puis test et analyse des erreurs.

Les cellules ne contiennent pas de sorties d'entraînement simulées. Elles écrivent les résultats mesurés dans `results/`.
