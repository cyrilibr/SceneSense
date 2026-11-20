# SceneSense — Compte rendu final (résultats réels)

## Objet
Classifier six scènes du dataset Intel et comparer un CNN construit depuis zéro à MobileNetV2 avec transfer learning.

## Données et méthode
12 632 images pour entraîner, 1 402 pour la validation et 3 000 images de test réservées. Entrées RGB 160 × 160, batch 32, graine 42. Modèle final choisi sur la **perte de validation**.

## Modèles et validation
| Expérience | Accuracy validation | Perte validation | Époques |
|---|---:|---:|---:|
| E1 — CNN simple | 86.45 % | 0.3748 | 25 |
| E2 — CNN + augmentation | 81.38 % | 0.5004 | 25 |
| E3 — CNN + dropout | 83.59 % | 0.4390 | 24 |
| E4 — MobileNetV2 gelé | 91.87 % | 0.2248 | 14 |
| E5 — MobileNetV2 ajusté | 92.30 % | 0.2047 | 9 |

Le CNN simple réussit mieux que ses variantes avec augmentation dans cette série d’expériences. Le dropout améliore E2 mais ne dépasse pas E1. MobileNetV2 gelé améliore E1 de 5,42 points, et le fine-tuning améliore encore E4 de 0,43 point.

## Test final : MobileNetV2 fine-tuning
- **Accuracy** : 92,63 % (2 779 / 3 000 images correctes).
- **Macro-F1** : 0,9275.
- **Rappel forest** : 469 / 474 = 98,95 %.
- **Rappel glacier** : 482 / 553 = 87,16 %.
- **Confusions majeures** : 97 erreurs glacier/mountain et 72 erreurs buildings/street, soit 169 / 221 erreurs.

## Discussion
Les résultats montrent le bénéfice du transfert de caractéristiques ImageNet. Le gain lié au fine-tuning est réel mais faible; l’étude est limitée à une exécution par configuration. Le test est utilisé uniquement pour le modèle choisi, pas pour arbitrer les hyperparamètres.

## Fichiers de preuve
Les fichiers de résultats sont dans `results/validation_summary.csv`, `results/*_info.json`, `results/*_history.csv` et `results/evaluation/test_mobilenet_finetune/`. Les modèles `.keras` sont conservés séparément dans Google Drive.

## Sources
Cours CH1 à CH4 ; dataset Intel sur Kaggle ; documentation TensorFlow/Keras.
