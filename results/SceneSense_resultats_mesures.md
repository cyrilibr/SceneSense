# SceneSense — Résultats mesurés

Le modèle final a été choisi en fonction de la perte de validation, sans consulter le test.

| Expérience | Validation accuracy | Validation loss | Durée (s) |
| --- | ---: | ---: | ---: |
| mobilenet_finetune | 92.30% | 0.2047 | 315.5 |
| mobilenet_frozen | 91.87% | 0.2248 | 472.4 |
| cnn_baseline | 86.45% | 0.3748 | 784.9 |
| cnn_dropout | 83.59% | 0.4390 | 734.8 |
| cnn_augmentation | 81.38% | 0.5004 | 786.7 |

**Modèle sélectionné :** mobilenet_finetune.
**Test final :** accuracy 92.63%, macro-F1 0.9275 sur 3000 images.

Le détail par classe et la matrice de confusion sont disponibles dans `results/evaluation/`.
