"""Make a black-and-white Aptos report for the student project."""
from pathlib import Path
import re
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT=Path(__file__).resolve().parents[1]
source=(ROOT/'docs/rapport.md').read_text(encoding='utf-8')
doc=Document()
sec=doc.sections[0]
sec.top_margin=Inches(.82);sec.bottom_margin=Inches(.8)
sec.left_margin=Inches(.92);sec.right_margin=Inches(.85)
styles=doc.styles
normal=styles['Normal']; normal.font.name='Aptos';normal.font.size=Pt(10.5);normal.font.color.rgb=RGBColor(22,22,22)
normal.paragraph_format.space_after=Pt(7)
normal.paragraph_format.line_spacing=1.14
for n,sz in [('Title',20),('Heading 1',13),('Heading 2',11)]:
    style=styles[n];style.font.name='Aptos Display' if n!='Heading 2' else 'Aptos';style.font.size=Pt(sz)
    style.font.bold=True;style.font.color.rgb=RGBColor(15,15,15)
    style.paragraph_format.space_after=Pt(8)
    style.paragraph_format.space_before=Pt(13 if n!='Title' else 5)

heading=doc.add_paragraph(style='Title')
heading.add_run('SceneSense')
sub=doc.add_paragraph()
sub.alignment=WD_ALIGN_PARAGRAPH.LEFT
run=sub.add_run('Classification de scènes naturelles par CNN et transfer learning')
run.bold=True;run.font.size=Pt(12);run.font.color.rgb=RGBColor(65,65,65)
meta=doc.add_paragraph()
meta.add_run('Projet de Deep Learning — Cycle ingénieur A4').italic=True
meta.paragraph_format.space_after=Pt(13)

lines=source.splitlines()
for line in lines:
    line=line.strip()
    if not line or line.startswith('# SceneSense :'):
        continue
    if line.startswith('## '):
        doc.add_heading(line[3:].strip(),level=1)
        continue
    p=doc.add_paragraph(style='Normal')
    line=line.replace('**','').replace('`','')
    p.add_run(line)

header=sec.header.paragraphs[0]
header.alignment=WD_ALIGN_PARAGRAPH.RIGHT
run=header.add_run('SCENESENSE  /  DEEP LEARNING')
run.font.name='Aptos';run.font.size=Pt(8);run.font.color.rgb=RGBColor(80,80,80)
footer=sec.footer.paragraphs[0]
footer.alignment=WD_ALIGN_PARAGRAPH.RIGHT
run=footer.add_run('Rapport de projet   ·   ')
run.font.size=Pt(8)
field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');footer._p.append(field)

out=ROOT/'docs/SceneSense_rapport.docx'
doc.save(out)
print(out)
