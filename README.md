# SceneSense — Classification d'images par CNN et transfer learning

A4 — Projet de Deep Learning • TensorFlow / Keras 3

**Question étudiée :** dans quelle mesure l'augmentation de données, le dropout et le transfert de connaissances améliorent-ils la classification de scènes naturelles par rapport à un CNN entraîné de zéro ?

Le dépôt contient **cinq expériences réellement entraînées**, les données de validation et de test, quatre notebooks, une présentation et un rapport final. Les résultats sont issus de l'exécution du 20 novembre 2026 : **92,63 % d'accuracy** et **0,9275 de macro-F1** sur le test final (3 000 images), pour le modèle MobileNetV2 fine-tuning.

## Travail demandé et réalisé

| Consigne de la séance | Réponse du projet |
| --- | --- |
| Choisir un dataset | Intel Image Classification, six classes |
| Développer un modèle CNN | CNN personnel avec 3 blocs convolutifs |
| Utiliser au moins un transfer learning | MobileNetV2 préentraîné sur ImageNet |
| Expérimenter, améliorer, comparer | Augmentation de données, dropout et fine-tuning |
| Présentation et questions individuelles | Diapositives, théorie, choix techniques, guide oral |

## Dataset

Source : [Intel Image Classification sur Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification).

Classes, dans l'ordre utilisé par le code : `buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`.
Le dossier original contient environ 14 000 images d'entraînement, 3 000 images de test étiquetées et un dossier de prédiction non étiqueté. Seules les images de `seg_train` et de `seg_test` sont utilisées.

Arborescence acceptée :

```text
data/intel-image-classification/
├── seg_train/
│   └── seg_train/
│       ├── buildings/ ...
│       ├── forest/ ...
│       └── ...
└── seg_test/
    └── seg_test/
        ├── buildings/ ...
        ├── forest/ ...
        └── ...
```

Les dossiers de classes directement sous `seg_train/` et `seg_test/` sont également pris en charge.

## Installation et exécution (Google Colab ou ordinateur)

Dans Colab, activer si possible **Exécution > Modifier le type d'exécution > GPU**, puis importer le dossier du projet et ouvrir un terminal dans sa racine. Pour une machine avec TensorFlow :

```bash
pip install -r requirements-colab.txt
pip install -e .
python scripts/download_dataset.py
python -m scenesense.cli prepare
python -m scenesense.cli train-all
python -m scenesense.cli plot
python -m scenesense.cli evaluate
```

Le téléchargement Kaggle nécessite une connexion internet. Si Kaggle demande une connexion ou un jeton, télécharger et décompresser le dataset dans le dossier `data/` manuellement, puis continuer à `prepare`.

**Si le dossier de données est ailleurs :** placer `--data-dir /chemin/vers/dataset` *avant* la sous-commande, par exemple `python -m scenesense.cli --data-dir /content/Intel prepare`.

**Pour entraîner une seule expérience :**

```bash
python -m scenesense.cli train cnn_baseline
python -m scenesense.cli train cnn_augmentation
python -m scenesense.cli train cnn_dropout
python -m scenesense.cli train mobilenet_frozen
python -m scenesense.cli train mobilenet_finetune
```

La dernière étape exige un checkpoint `mobilenet_frozen.keras` déjà entraîné. Elle affine les dernières couches du modèle préentraîné avec un taux d'apprentissage réduit.

**Pour prédire une image :**

```bash
python -m scenesense.cli predict chemin/vers/photo.jpg
```

## Expériences

| Code | Modèle | Modification isolée |
| --- | --- | --- |
| E1 | `cnn_baseline` | CNN de référence sans augmentation ni dropout |
| E2 | `cnn_augmentation` | Même CNN + flips / rotations / zooms |
| E3 | `cnn_dropout` | E2 + dropout de 0,3 |
| E4 | `mobilenet_frozen` | Backbone ImageNet figé + classifieur à six classes |
| E5 | `mobilenet_finetune` | E4 + adaptation des dernières couches |

Les trois expériences CNN utilisent le même générateur aléatoire pour leur initialisation, le même découpage et les mêmes hyperparamètres d'entraînement. Les modèles reçoivent des images de 160 × 160 pixels (RGB). L'entrée est normalisée en `[0,1]` pour le CNN et en `[-1,1]` pour MobileNetV2 : cette différence est indispensable au modèle préentraîné.

## Evaluation : pas de fuite de données

- Répartition stratifiée de `seg_train` : **90 % entraînement / 10 % validation** avec graine `42`.
- `seg_test` est réservé à une **évaluation finale unique** après sélection du modèle par la *validation loss*.
- Les checkpoints sont sauvegardés dans `results/models/`, les historiques en CSV et le rapport de classification dans `results/evaluation/`.
- Mesures : accuracy, macro-F1, matrice de confusion, loss de validation, durée d'entraînement et nombre de paramètres entraînables.
- Pour examiner les erreurs, consulter `predictions.csv` généré après l'évaluation.

**État actuel :** les cinq expériences ont été entraînées sur Intel. Les résultats mesurés sont fournis sous `results/` et détaillés dans le rapport final. Seul le modèle choisi sur la validation (MobileNetV2 fine-tuning) a été évalué sur le test. Les poids `.keras` sont conservés sur Google Drive et ne sont pas publiés sur GitHub.

## Documents

- `notebooks/01_dataset.ipynb` — dataset et découpage.
- `notebooks/02_custom_cnn.ipynb` — E1, E2, E3.
- `notebooks/03_transfer_learning.ipynb` — E4, E5.
- `notebooks/04_results.ipynb` — graphiques, confusion, erreurs.
- `docs/rapport.md` — rapport de résultats mesurés, en Markdown.
- `docs/SceneSense_rapport_final.docx` et `.pdf` — rapport final en français.
- `docs/SceneSense_resultats_mesures.md` — tableau des résultats.
- `docs/questions_oral.md` — révision des notions du cours.
- `slides/SceneSense_presentation_finale.pptx` et `.pdf` — présentation finale avec les chiffres réels.

## Tests

En local, le projet peut être testé avec Keras 3 et le backend PyTorch, sans installer TensorFlow :

```bash
pip install -e '.[torch,test]'
KERAS_BACKEND=torch pytest -q
```

Les tests couvrent le découpage, les données, la construction des modèles, et les règles du fine-tuning. Un smoke test d'entraînement est également prévu sur de petites images synthétiques, **sans valeur scientifique pour mesurer la précision sur Intel**.

## Références

- Supports du cours fournis : *Introduction to Deep Learning*, *Backpropagation*, *Regularization and Optimization*, *Convolutional Neural Networks*.
- [TensorFlow — Transfer learning and fine-tuning](https://www.tensorflow.org/guide/keras/transfer_learning)
- [TensorFlow — MobileNetV2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2)
- [Dataset Intel](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)

Les éléments de théorie sont expliqués simplement et les expérimentations sont séparées pour pouvoir discuter honnêtement de ce qui améliore — ou non — la généralisation.

## Résultats mesurés — 20 novembre 2026

Les 14 034 images `seg_train` ont été réparties en **12 632 images d'entraînement** et **1 402 images de validation**. Le test indépendant comprend **3 000 images**.

| Modèle | Accuracy validation | Perte de validation |
| --- | ---: | ---: |
| CNN simple | 86,45 % | 0,3748 |
| CNN + augmentation | 81,38 % | 0,5004 |
| CNN + dropout | 83,59 % | 0,4390 |
| MobileNetV2 gelé | 91,87 % | 0,2248 |
| **MobileNetV2 fine-tuning** | **92,30 %** | **0,2047** |

**Test final (MobileNetV2 fine-tuning) :** 92,63 % d'accuracy, macro-F1 = 0,9275, soit 2 779 images correctement classées. Sur les 221 erreurs, 97 concernent glacier/mountain et 72 buildings/street. Les classes ne sont pas toutes reconnues avec la même facilité : le rappel `forest` atteint 98,95 % contre 87,16 % pour `glacier`.

Les historiques, le tableau comparatif et la matrice de confusion sont disponibles dans `results/`. Les checkpoints `.keras` et les photos Intel n'y figurent pas : ils restent dans Google Drive ou doivent être téléchargés séparément.

## Reproduire les figures et la présentation depuis un entraînement

Après `train-all` puis `evaluate`, la commande `python scripts/export_results.py` peut régénérer les fichiers de résultats. La présentation et le rapport déjà finalisés sont disponibles dans `slides/SceneSense_presentation_finale.pptx` et `docs/SceneSense_rapport_final.docx`.

## GitHub

Dépôt du projet : [cyrilibr/SceneSense](https://github.com/cyrilibr/SceneSense).
