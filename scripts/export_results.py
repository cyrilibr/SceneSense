"""Export measured results to a separate final presentation and summary file.

This script requires completed training and the reserved final test evaluation.
It never inserts artificial numbers into a student report.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

from scenesense.config import Settings
from scenesense.evaluation import load_validation_results, plot_training, select_model

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description='Export measured metrics into report and slides')
parser.add_argument('--results-dir', type=Path, default=ROOT / 'results')
parser.add_argument('--docs-dir', type=Path, default=ROOT / 'docs')
parser.add_argument('--slides-dir', type=Path, default=ROOT / 'slides')
args = parser.parse_args()
args.docs_dir.mkdir(parents=True, exist_ok=True)
args.slides_dir.mkdir(parents=True, exist_ok=True)
SETTINGS = Settings(data_dir=ROOT / 'data/intel-image-classification', output_dir=args.results_dir)
selected = select_model(SETTINGS)
metrics_path = SETTINGS.output_dir / 'evaluation' / f'test_{selected}' / 'metrics.json'
if not metrics_path.is_file():
    raise SystemExit('Run python -m scenesense.cli evaluate before exporting measured results.')
metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
valid = load_validation_results(SETTINGS).sort_values('validation_loss')
if len(valid) < 5:
    raise SystemExit('All five experiments must be trained before exporting the final deck.')

lines = [
    '# SceneSense — Résultats mesurés',
    '',
    'Le modèle final a été choisi en fonction de la perte de validation, sans consulter le test.',
    '',
    '| Expérience | Validation accuracy | Validation loss | Durée (s) |',
    '| --- | ---: | ---: | ---: |',
]
for _, r in valid.iterrows():
    lines.append(f'| {r.experiment} | {r.validation_accuracy:.2%} | {r.validation_loss:.4f} | {r.duration_seconds:.1f} |')
lines.extend(['', f'**Modèle sélectionné :** {selected}.',
              f'**Test final :** accuracy {metrics["accuracy"]:.2%}, macro-F1 {metrics["macro_f1"]:.4f} '
              f'sur {metrics["samples"]} images.',
              '', 'Le détail par classe et la matrice de confusion sont disponibles dans `results/evaluation/`.'])
out_md=args.docs_dir/'SceneSense_resultats_mesures.md'
out_md.write_text('\n'.join(lines)+'\n',encoding='utf-8')

prs=Presentation(ROOT/'slides/SceneSense_presentation.pptx')
navy=RGBColor(21,40,68)
for title in ('Résultats de validation', 'Test final et erreurs'):
    slide=prs.slides.add_slide(prs.slide_layouts[6])
    bg=slide.shapes.add_shape(1,0,0,prs.slide_width,prs.slide_height)
    bg.fill.solid();bg.fill.fore_color.rgb=RGBColor(255,255,255);bg.line.fill.background()
    t=slide.shapes.add_textbox(Inches(.75),Inches(.55),Inches(11),Inches(.8))
    p=t.text_frame.paragraphs[0];p.text=title;p.font.name='Aptos';p.font.bold=True;p.font.size=Pt(30);p.font.color.rgb=navy

slide=prs.slides[-2]
path=plot_training(SETTINGS)
slide.shapes.add_picture(str(path),Inches(.75),Inches(1.52),height=Inches(4.8))
box=slide.shapes.add_textbox(Inches(8.95),Inches(1.75),Inches(3.3),Inches(4.7))
frame=box.text_frame
frame.word_wrap=True
for i,(_,r) in enumerate(valid.iterrows()):
    p=frame.paragraphs[0] if i==0 else frame.add_paragraph()
    p.text=f'{r.experiment.replace("_"," ")}\naccuracy {r.validation_accuracy:.1%} • loss {r.validation_loss:.3f}'
    p.font.size=Pt(13);p.font.color.rgb=navy
    p.space_after=Pt(14)

slide=prs.slides[-1]
cm=SETTINGS.output_dir/'evaluation'/f'test_{selected}'/'confusion_matrix.png'
if not cm.is_file():
    raise SystemExit(f'Missing confusion matrix: {cm}')
slide.shapes.add_picture(str(cm), Inches(.78), Inches(1.47), height=Inches(5.5))
box=slide.shapes.add_textbox(Inches(7.65),Inches(2.2),Inches(4.6),Inches(2.9))
p=box.text_frame.paragraphs[0]
p.text=f'Modèle : {selected}\n\nAccuracy : {metrics["accuracy"]:.2%}\nMacro-F1 : {metrics["macro_f1"]:.3f}\nImages : {metrics["samples"]}'
p.font.name='Aptos';p.font.size=Pt(20);p.font.color.rgb=navy

out_pptx=args.slides_dir/'SceneSense_presentation_avec_resultats.pptx'
prs.save(out_pptx)
print(f'Created {out_md} and {out_pptx}')
