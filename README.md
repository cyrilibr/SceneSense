# SceneSense — CNN & Transfer Learning

Projet de Deep Learning, niveau A4 — TensorFlow / Keras.

Ce dépôt compare un réseau convolutif entraîné depuis zéro à un modèle MobileNetV2 préentraîné sur ImageNet, pour reconnaître six types de scènes naturelles.

**Question du projet :** dans quelle mesure la régularisation et le transfert d'apprentissage améliorent-ils la généralisation d'un modèle de classification d'images ?

## Consignes du professeur

| Attendu | Réalisation |
| --- | --- |
| Dataset | Intel Image Classification, six classes |
| CNN personnel | CNN avec trois convolutions, pooling et Softmax |
| Transfer learning | MobileNetV2 avec les poids ImageNet |
| Expériences complémentaires | Augmentation de données, dropout, fine-tuning |
| Présentation et oral | Diapositives, rapport, questions théoriques |

## Données

Téléchargement : [Intel Image Classification sur Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification).

Classes : `buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`.

La partie officielle `seg_train` est divisée en 90 % entraînement et 10 % validation par classe, avec une graine de 42. Le dossier officiel `seg_test` est conservé pour l'évaluation finale. La répartition est sauvegardée dans `results/splits.csv`.

Les images sont redimensionnées à 160 × 160 pixels, avec des lots de 32. Le CNN normalise les pixels en [0,1]. MobileNetV2 utilise la normalisation en [-1,1] attendue par ses poids ImageNet.

## Installation

Python 3.10 ou plus récent, de préférence avec GPU NVIDIA / Google Colab.

```bash
python -m pip install -r requirements-colab.txt
python -m pip install -e .
python scripts/download_dataset.py
python -m scenesense.cli prepare
```

Le téléchargement automatique demande une connexion Internet. Il est également possible de déposer directement les images dans `data/intel-image-classification/`. Les sous-dossiers d'origine `seg_train/seg_train/` et `seg_test/seg_test/` sont pris en charge, ainsi que les versions à un seul niveau.

Dans Colab, exécuter les commandes depuis la racine du dépôt cloné, et ouvrir les notebooks depuis cette racine. Si le dataset est stocké ailleurs, utiliser `--data-dir /chemin/vers/dataset` avant la sous-commande.

## Expériences

| Code | Modèle | Hypothèse testée |
| --- | --- | --- |
| E1 | `cnn_baseline` | CNN simple, sans régularisation supplémentaire |
| E2 | `cnn_augmentation` | Ajout de flips, rotations et zooms |
| E3 | `cnn_dropout` | Ajout du dropout 0,30 à E2 |
| E4 | `mobilenet_frozen` | Classifieur à six classes sur un MobileNetV2 figé |
| E5 | `mobilenet_finetune` | Ajustement des dernières couches du modèle E4 |

Le même découpage d'images est utilisé dans les cinq expériences. La validation loss sert à retenir le meilleur checkpoint, avec early stopping.

```bash
python -m scenesense.cli train-all
```

Ou séparer les essais :

```bash
python -m scenesense.cli train cnn_baseline
python -m scenesense.cli train cnn_augmentation
python -m scenesense.cli train cnn_dropout
python -m scenesense.cli train mobilenet_frozen
python -m scenesense.cli train mobilenet_finetune
```

E5 recharge le meilleur checkpoint E4 : toujours entraîner `mobilenet_frozen` avant `mobilenet_finetune`.

## Comparaison des modèles

```bash
python -m scenesense.cli plot
python -m scenesense.cli evaluate
python scripts/export_results.py
```

L'évaluation finale utilise **uniquement le modèle retenu grâce à la validation**, sans choisir sur le test. Les sorties comprennent accuracy, macro-F1, matrice de confusion et exemples d'erreurs. Le script `export_results.py` génère ensuite une présentation enrichie et un récapitulatif à partir des mesures réellement enregistrées.

Pour une prédiction unique :

```bash
python -m scenesense.cli predict chemin/vers/image.jpg
```

## Notebooks et documents

- `notebooks/01_dataset.ipynb` — exploration et séparation des images ;
- `notebooks/02_custom_cnn.ipynb` — expériences E1, E2, E3 ;
- `notebooks/03_transfer_learning.ipynb` — MobileNetV2 et fine-tuning ;
- `notebooks/04_results.ipynb` — comparaison et analyse des erreurs ;
- `docs/rapport.md` et `docs/SceneSense_rapport.docx` — compte rendu ;
- `docs/questions_oral.md` — révisions pour les questions individuelles ;
- `slides/SceneSense_presentation.pptx` — présentation modifiable.

Les documents et notebooks peuvent être régénérés avec les scripts `build_notebooks.py`, `build_report.py` et `build_slides.py`.

## Tests

```bash
pip install -e '.[torch,test]'
KERAS_BACKEND=torch pytest -q
python scripts/check_project.py
```

Les tests utilisent un petit dataset synthétique uniquement pour vérifier le fonctionnement du pipeline ; il ne permet pas d'estimer la précision sur Intel.

## Statut des résultats et honnêteté expérimentale

Le code et le protocole ont été préparés, mais **les modèles ne sont pas encore entraînés sur le jeu de données Intel complet**. Aucun score de classification présenté ici ne constitue un résultat scientifique mesuré. Les résultats seront produits par les commandes d'entraînement, avec un GPU et le dataset réel.

Les fichiers de données et les checkpoints ne sont pas commités : ils sont volumineux et générés séparément.

## Références

- Supports de cours : introduction au deep learning, backpropagation, régularisation et optimisation, CNN.
- [Dataset Intel](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
- [TensorFlow — Transfer learning and fine-tuning](https://www.tensorflow.org/guide/keras/transfer_learning)
- [TensorFlow — MobileNetV2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2)
