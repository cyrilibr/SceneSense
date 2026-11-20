"""Generate four clean explanatory notebooks from the tested package API."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "notebooks"
DEST.mkdir(exist_ok=True)


def markdown(t):
    return nbf.v4.new_markdown_cell(t)


def code(t):
    return nbf.v4.new_code_cell(t)


SETUP = '''from pathlib import Path
import sys
candidates = [Path.cwd(), Path.cwd().parent, Path('/content/SceneSense')]
ROOT = next((p for p in candidates if (p / 'pyproject.toml').is_file()), None)
if ROOT is None:
    raise FileNotFoundError('Open this notebook from the SceneSense project folder in Colab or Jupyter')
sys.path.insert(0, str(ROOT / 'src'))
from scenesense.config import Settings, CLASSES
settings = Settings(data_dir=ROOT / "data/intel-image-classification", output_dir=ROOT / "results")
print("Project:", ROOT)
print("Dataset:", settings.data_dir)'''

notebooks = {
    "01_dataset.ipynb": [
        markdown("# 01 · Découverte du dataset\n\n**SceneSense — A4 / Deep Learning.** Objectif : inspecter les données et enregistrer une séparation fixe. Les images du test ne participent pas à l'entraînement."),
        code(SETUP),
        markdown("## Construire les trois ensembles\n\nLa répartition est stratifiée par catégorie. La partie officielle `seg_test` reste indépendante."),
        code('''from scenesense.data import build_manifest, read_manifest
frame = build_manifest(settings.data_dir, settings.manifest_path,
                       settings.seed, settings.validation_fraction)
print(frame.groupby(["split", "label"]).size().unstack(fill_value=0).rename(columns=dict(enumerate(CLASSES))))
print("Saved split:", settings.manifest_path)'''),
        markdown("## Exemples d'images\n\nChaque catégorie a des caractéristiques visuelles distinctes, mais les classes glacier et montagne peuvent se ressembler."),
        code('''import matplotlib.pyplot as plt
from PIL import Image
fig, axes = plt.subplots(2, 3, figsize=(11, 7))
for ax, cls in zip(axes.ravel(), CLASSES):
    row = frame[(frame.label == CLASSES.index(cls)) & (frame.split == "train")].iloc[0]
    with Image.open(settings.data_dir / row.path) as im:
        ax.imshow(im.convert("RGB"))
    ax.set_title(cls.capitalize())
    ax.axis("off")
plt.tight_layout()
plt.show()'''),
    ],
    "02_custom_cnn.ipynb": [
        markdown("# 02 · CNN entraîné de zéro\n\nNous comparons une référence simple, l'augmentation de données puis le dropout. L'idée est de modifier un seul facteur à la fois."),
        code(SETUP),
        markdown("## Architecture\n\nTrois convolutions 3×3 (32, 64, 128 filtres), trois max-pooling 2×2, GlobalAveragePooling, Dense(128), Softmax(6)."),
        code('''from scenesense.models import build_custom_cnn, compile_model
model = compile_model(build_custom_cnn(image_size=settings.size))
model.summary()'''),
        markdown("## Entraîner E1, E2 et E3\n\nLe meilleur checkpoint est enregistré selon la perte de validation, indépendamment des données de test."),
        code('''from scenesense.data import read_manifest
from scenesense.training import train_experiment
assert settings.manifest_path.exists(), "Run notebook 01 first to prepare the fixed split"
cnn_results = []
for experiment in ("cnn_baseline", "cnn_augmentation", "cnn_dropout"):
    cnn_results.append(train_experiment(experiment, settings))'''),
        code('''import pandas as pd
display(pd.DataFrame(cnn_results)[["experiment", "validation_accuracy", "validation_loss", "epochs_completed", "duration_seconds"]])'''),
        markdown("## Commentaire\n\nComparer E1 à E2 pour isoler l'augmentation de données, puis E2 à E3 pour isoler le dropout. Si la validation ne s'améliore pas, expliquer ce résultat au lieu de le masquer."),
    ],
    "03_transfer_learning.ipynb": [
        markdown("# 03 · Transfer learning avec MobileNetV2\n\nE4 : réseau ImageNet gelé et nouvelle tête de classification. E5 : fine-tuning des dernières couches à learning rate réduit."),
        code(SETUP),
        markdown("## Inspecter la structure\n\nCette étape exige l'accès aux poids ImageNet lors de la première utilisation."),
        code('''from scenesense.models import build_mobilenet, compile_model
mobile = compile_model(build_mobilenet(image_size=settings.size))
mobile.summary()'''),
        markdown("## E4 · Backbone figé"),
        code('''from scenesense.training import train_experiment
frozen = train_experiment("mobilenet_frozen", settings)
print(frozen)'''),
        markdown("## E5 · Fine-tuning\n\nL'expérience E5 recharge le meilleur checkpoint E4, défige les 25 dernières couches pertinentes et garde BatchNormalization en mode inférence."),
        code('''finetuned = train_experiment("mobilenet_finetune", settings)
print(finetuned)'''),
    ],
    "04_results.ipynb": [
        markdown("# 04 · Evaluation et analyse des erreurs\n\nLe meilleur modèle est choisi sur la perte de validation. Le test n'est utilisé qu'après ce choix."),
        code(SETUP),
        markdown("## Résumé de la validation"),
        code('''from scenesense.evaluation import load_validation_results, plot_training, select_model, evaluate_final
import pandas as pd
from IPython.display import display
summary = load_validation_results(settings).sort_values("validation_loss")
display(summary[["experiment", "validation_accuracy", "validation_loss", "trainable_parameters", "duration_seconds"]])
print("Model chosen by validation loss:", select_model(settings))'''),
        code('''from PIL import Image
path = plot_training(settings)
display(Image.open(path))'''),
        markdown("## Test final\n\nN'exécuter cette étape qu'une fois les comparaisons de validation terminées."),
        code('''final = evaluate_final(settings)
print(final)'''),
        markdown("## Matrice de confusion et erreurs"),
        code('''from pathlib import Path
best = final["experiment"]
folder = settings.output_dir / "evaluation" / f"test_{best}"
display(Image.open(folder / "confusion_matrix.png"))
print((folder / "classification_report.txt").read_text())
errors = pd.read_csv(folder / "predictions.csv")
errors = errors[~errors.correct].sort_values("confidence", ascending=False)
display(errors.head(10))'''),
        code('''import matplotlib.pyplot as plt
if not errors.empty:
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for ax, (_, row) in zip(axes.flat, errors.head(6).iterrows()):
        with Image.open(settings.data_dir / row.path) as im:
            ax.imshow(im.convert("RGB"))
        ax.set_title(f"True: {CLASSES[int(row.label)]} / Pred: {CLASSES[int(row.prediction)]}", fontsize=9)
        ax.axis("off")
    for ax in axes.flat[len(errors.head(6)):]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()
else:
    print("No classification errors found in the test split.")'''),
    ],
}

for name, cells in notebooks.items():
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    })
    nbf.write(notebook, DEST / name)
    print("Created", name)
