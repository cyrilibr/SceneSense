"""Create an editable, clean A4-level presentation (no invented experimental data)."""
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'slides/SceneSense_presentation.pptx'
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
NAVY = RGBColor(21, 40, 68)
INK = RGBColor(35, 45, 60)
GRAY = RGBColor(105, 117, 130)
CYAN = RGBColor(28, 138, 164)
PALE = RGBColor(234, 245, 248)
LIGHT = RGBColor(247, 249, 251)
WHITE = RGBColor(255,255,255)
SOFT = RGBColor(225, 231, 235)


def rect(slide,x,y,w,h,fill,rounded=False,line=None,radius=None):
    s=slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE,
                             Inches(x),Inches(y),Inches(w),Inches(h))
    s.fill.solid();s.fill.fore_color.rgb=fill
    s.line.fill.background() if line is None else None
    if line is not None:
        s.line.color.rgb=line
    if rounded:
        try:s.adjustments[0]=0.12
        except Exception:pass
    return s


def text(slide,txt,x,y,w,h,size=18,color=INK,bold=False,align='left',font='Aptos',margin=0.0):
    shape=slide.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h))
    tf=shape.text_frame;tf.clear();tf.word_wrap=True
    tf.margin_left=tf.margin_right=Inches(margin)
    tf.margin_top=tf.margin_bottom=Inches(0.02)
    tf.vertical_anchor=MSO_ANCHOR.MIDDLE
    for i,line in enumerate(txt.split('\n')):
        p=tf.paragraphs[0] if i==0 else tf.add_paragraph()
        p.text=line;p.font.name=font;p.font.size=Pt(size);p.font.bold=bold;p.font.color.rgb=color
        p.alignment={'left':PP_ALIGN.LEFT,'center':PP_ALIGN.CENTER,'right':PP_ALIGN.RIGHT}[align]
        p.space_after=Pt(3)
    return shape


def base(n,kicker,title,subtitle=None):
    slide=prs.slides.add_slide(prs.slide_layouts[6]);rect(slide,0,0,13.333,7.5,WHITE)
    rect(slide,0,0,13.333,0.09,CYAN)
    text(slide,kicker.upper(),0.72,0.40,10,0.23,size=10,color=CYAN,bold=True)
    text(slide,title,0.72,0.82,12.0,0.72,size=29,color=NAVY,bold=True)
    if subtitle:text(slide,subtitle,0.73,1.63,11.8,0.48,size=14,color=GRAY)
    rect(slide,0.72,7.04,11.86,0.015,SOFT)
    text(slide,'SCENESENSE  /  DEEP LEARNING  /  PROJET A4',0.72,7.12,8,0.22,size=9,color=GRAY)
    text(slide,str(n).zfill(2),12.1,7.1,0.45,0.22,size=10,color=GRAY,align='right')
    return slide


def card(s,x,y,w,h,title,body,accent=CYAN):
    rect(s,x,y,w,h,LIGHT,rounded=True)
    rect(s,x+0.15,y+0.19,0.06,0.32,accent)
    text(s,title,x+0.35,y+0.16,w-0.55,0.44,16,NAVY,True)
    text(s,body,x+0.35,y+0.73,w-0.7,h-0.87,12,GRAY)


# 1
s=prs.slides.add_slide(prs.slide_layouts[6]);rect(s,0,0,13.333,7.5,NAVY)
rect(s,0.73,0.80,0.62,0.07,CYAN)
text(s,'DEEP LEARNING  ·  PROJET A4',0.75,1.13,10,0.34,size=14,color=RGBColor(151,211,223),bold=True)
text(s,'SceneSense',0.73,2.03,10.9,0.95,size=49,color=WHITE,bold=True)
text(s,'Classification de scènes naturelles par CNN\net transfer learning',0.78,3.12,10.9,1.18,size=23,color=WHITE)
rect(s,0.76,5.52,11.7,0.02,RGBColor(62,86,117))
text(s,'6 catégories   |   5 expériences   |   1 protocole reproductible',0.79,5.82,11.1,0.53,size=18,color=RGBColor(194,211,228))
text(s,'Architecture · régularisation · MobileNetV2 · évaluation',0.79,6.56,11.9,0.37,size=12,color=RGBColor(169,193,213))
# 2
s=base(2,'Pourquoi ce projet ?','Question étudiée','Comprendre les choix du modèle, pas uniquement chercher une accuracy élevée.')
rect(s,0.75,2.37,11.85,1.35,PALE,rounded=True)
text(s,'Comment améliorer la généralisation d’un CNN sur des images inédites ?',1.16,2.65,11.2,0.85,size=23,color=NAVY,bold=True)
card(s,0.75,4.13,3.75,2.0,'01 · CNN','Construire une référence et observer son apprentissage.')
card(s,4.78,4.13,3.75,2.0,'02 · Régularisation','Tester des transformations et le dropout.')
card(s,8.81,4.13,3.75,2.0,'03 · Transfert','Comparer avec un réseau ImageNet préentraîné.')
# 3
s=base(3,'Dataset','Intel Image Classification','Images RGB de scènes naturelles et urbaines, réparties en six classes.')
classes=['BUILDINGS','FOREST','GLACIER','MOUNTAIN','SEA','STREET']
for i,c in enumerate(classes):
    x=0.77+(i%3)*4.05;y=2.25+(i//3)*1.37
    rect(s,x,y,3.74,1.13,PALE if i%2==0 else LIGHT,rounded=True)
    rect(s,x+0.23,y+0.24,0.14,0.64,CYAN if i%2==0 else NAVY)
    text(s,c,x+0.59,y+0.31,2.80,0.45,size=18,color=NAVY,bold=True)
text(s,'seg_train → 90 % entraînement / 10 % validation  •  seg_test → test final indépendant',0.84,5.31,11.7,0.45,size=16,color=INK)
text(s,'Même découpage pour les cinq expériences · Images redimensionnées à 160 × 160',0.84,6.09,11.6,0.36,size=13,color=GRAY)
# 4
s=base(4,'Architecture','Notre CNN : trois blocs convolutifs','Proche des notions du cours : convolution, ReLU, pooling, Dense et Softmax.')
steps=[('Entrée','160×160×3'),('Conv 32','3×3 + ReLU'),('Pool','80×80'),('Conv 64','3×3 + ReLU'),('Pool','40×40'),('Conv 128','3×3 + ReLU'),('Pool','20×20'),('Dense','128'),('Softmax','6 classes')]
for i,(name,dim) in enumerate(steps):
    x=0.69+(i%5)*2.49
    y=2.25 if i<5 else 4.53
    if i>=5:x=1.84+(i-5)*3.00
    w=2.20 if i<5 else 2.48
    rect(s,x,y,w,1.23,PALE if i not in (0,8) else NAVY,rounded=True)
    color=WHITE if i in (0,8) else NAVY
    text(s,name,x+0.08,y+0.24,w-0.16,0.34,size=16,color=color,bold=True,align='center')
    text(s,dim,x+0.08,y+0.66,w-0.16,0.27,size=11,color=RGBColor(208,225,234) if i in (0,8) else GRAY,align='center')
    if i<4:text(s,'→',x+w+0.08,y+0.49,0.22,0.34,18,CYAN,bold=True)
    if 5<=i<8:text(s,'→',x+w+0.10,y+0.49,0.25,0.34,18,CYAN,bold=True)
text(s,'GlobalAveragePooling entre le dernier pooling et Dense(128).',0.88,6.19,11,0.35,size=13,color=GRAY)
# 5
s=base(5,'Apprentissage','Ce qui se passe pendant l’entraînement','Passage avant → calcul de l’erreur → rétropropagation → mise à jour des poids.')
for i,(n,h,b) in enumerate([('01','Forward pass','L’image traverse le réseau pour produire six probabilités.'),('02','Cross-entropy','La perte mesure la qualité de la prédiction.'),('03','Backpropagation','Les gradients sont calculés par la règle de la chaîne.'),('04','Adam','Les poids sont ajustés avec un pas d’apprentissage.')]):
    x=0.74+(i%2)*6.05;y=2.24+(i//2)*1.90
    card(s,x,y,5.72,1.67,n+'  '+h,b)
text(s,'Taux initial : 0,001  ·  Batch size : 32  ·  Early stopping selon la validation loss',0.89,6.46,11.8,0.34,size=13,color=INK)
# 6
s=base(6,'Expériences','Des comparaisons contrôlées','Une amélioration est considérée comme une hypothèse à vérifier, pas comme un résultat acquis.')
for i,(name,body) in enumerate([('E1 — Référence','CNN sans augmentation ni dropout.'),('E2 — Augmentation','E1 + flip horizontal, rotation et zoom.'),('E3 — Dropout','E2 + dropout de 30 %.'),('E4 — Transfert','MobileNetV2 ImageNet, backbone gelé.'),('E5 — Fine-tuning','E4 avec dernières couches ajustables.')]):
    x=0.75+(i%3)*4.04;y=2.29+(i//3)*1.77
    card(s,x,y,3.77,1.49,name,body)
text(s,'Contrôle : mêmes ensembles d’images, même résolution, même méthode d’évaluation.',0.88,6.32,11.7,0.50,13,GRAY)
# 7
s=base(7,'Transfer learning','Pourquoi MobileNetV2 ?','Réutiliser des représentations visuelles déjà apprises sur ImageNet.')
card(s,0.77,2.31,5.55,2.48,'E4 · Extraction de caractéristiques','Backbone préentraîné et gelé. On entraîne uniquement le classifieur final à six classes.')
card(s,6.70,2.31,5.55,2.48,'E5 · Fine-tuning','On recharge E4, puis on adapte quelques couches hautes avec un learning rate de 10⁻⁵.')
rect(s,0.80,5.30,11.43,0.88,PALE,rounded=True)
text(s,'Point technique : MobileNetV2 reçoit des pixels normalisés en [-1 ; 1].',1.12,5.52,11.00,0.39,16,NAVY,True)
# 8
s=base(8,'Evaluation','Protocole pour éviter les biais','La sélection se fait sur la validation ; le test est gardé pour l’estimation finale.')
for i,(title,detail) in enumerate([('TRAIN','Ajuster les poids'),('VALIDATION','Comparer E1 à E5'),('TEST','Mesurer le modèle retenu')]):
    x=0.82+i*4.15
    rect(s,x,2.55,3.82,1.75,PALE if i!=2 else NAVY,rounded=True)
    color=NAVY if i!=2 else WHITE
    text(s,title,x+0.1,2.87,3.6,0.40,19,color,True,'center')
    text(s,detail,x+0.12,3.45,3.56,0.34,12,color,False,'center')
    if i<2:text(s,'→',x+3.84,3.18,0.27,0.30,18,CYAN,True)
text(s,'Mesures finales : accuracy, macro-F1, confusion matrix et exemples d’erreurs.',0.89,5.13,11.9,0.51,17,INK)
text(s,'Les sorties réelles sont générées automatiquement par les notebooks et les scripts.',0.90,5.79,11.6,0.45,13,GRAY)
# 9
s=base(9,'Livrables','Un projet que l’on peut reproduire','Les performances seront affichées à partir des exécutions réelles, jamais inventées.')
files=[('01','Notebooks','Exploration, entraînement, transfert, analyse'),('02','Code Python','Modules séparés : données, modèles, entraînement'),('03','Résultats','Courbes, classification report, matrice de confusion'),('04','Tests','Découpage, architecture, entraînement minimal')]
for i,(n,title,body) in enumerate(files):
    x=0.76+(i%2)*6.08;y=2.20+(i//2)*1.81
    card(s,x,y,5.66,1.63,n+' · '+title,body)
text(s,'Dossier exécutable : README.md + requirements-colab.txt + commandes CLI',0.91,6.27,11.8,0.32,13,INK)
# 10
s=base(10,'Conclusion','Ce que nous cherchons à démontrer')
rect(s,0.78,2.14,11.7,1.25,PALE,rounded=True)
text(s,'Un bon modèle est avant tout un modèle qui généralise.',1.11,2.43,11.1,0.64,24,NAVY,True)
for i,item in enumerate(['Comprendre les convolutions et les gradients', 'Mesurer l’effet réel de la régularisation', 'Comparer apprentissage complet et transfert', 'Expliquer les erreurs et les limites du modèle']):
    y=3.79+i*0.58
    rect(s,0.91,y+0.08,0.13,0.13,CYAN,rounded=True)
    text(s,item,1.29,y,10.6,0.39,17,INK)
text(s,'Sources : supports du cours · Dataset Intel (Kaggle) · documentation TensorFlow / Keras',0.87,6.36,11.7,0.32,11,GRAY)

OUT.parent.mkdir(exist_ok=True)
prs.save(OUT)
print(OUT)
