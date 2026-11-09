"""Build an assignment PDF with links and a genuine Virtual Lab screenshot.

Template: python generate_report.py
Final: python generate_report.py --name "Your Name" --bits-id "Your BITS ID" \
       --github "https://github.com/.../..." --app "https://...streamlit.app" \
       --screenshot "/path/to/your-virtual-lab-screenshot.png"

Never insert a classmate's screenshot or claim a deployment you did not perform.
"""
from argparse import ArgumentParser
from html import escape
from pathlib import Path
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
    Image, KeepTogether
)
from reportlab.lib.colors import HexColor
from config import ROOT, RESULTS_PATH, TEST_PATH, MODEL_FILES, MODEL_DIR, FEATURES, TARGET

NAVY=HexColor('#16324f'); TEAL=HexColor('#176e7a'); LIGHT=HexColor('#e9f2f5'); GRAY=HexColor('#576876')


def make_pdf(name, bits_id, github, app, screenshot, output):
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Title2',parent=styles['Title'],fontName='Helvetica-Bold',fontSize=22,leading=26,textColor=NAVY,spaceAfter=14))
    styles.add(ParagraphStyle(name='Sub2',parent=styles['Normal'],fontName='Helvetica',fontSize=10,leading=15,textColor=GRAY,spaceAfter=10))
    styles.add(ParagraphStyle(name='Head2',parent=styles['Heading2'],fontSize=12.5,leading=16,textColor=TEAL,spaceBefore=14,spaceAfter=6))
    styles.add(ParagraphStyle(name='Body2',parent=styles['BodyText'],fontSize=9.3,leading=13.5,spaceAfter=7))
    styles.add(ParagraphStyle(name='Small2',parent=styles['BodyText'],fontSize=8,leading=11,spaceAfter=5))
    styles.add(ParagraphStyle(name='Cell2',parent=styles['BodyText'],fontSize=7.3,leading=10))
    styles.add(ParagraphStyle(name='CellHeader2',parent=styles['BodyText'],fontSize=7.3,leading=10,textColor=colors.white,fontName='Helvetica-Bold'))
    styles.add(ParagraphStyle(name='Banner2',parent=styles['Normal'],fontSize=10,textColor=HexColor('#9b4c18'),fontName='Helvetica-Bold',spaceAfter=10))
    styles.add(ParagraphStyle(name='Center2',parent=styles['Body2'],alignment=TA_CENTER))
    pdf=SimpleDocTemplate(str(output),pagesize=A4,rightMargin=1.7*cm,leftMargin=1.7*cm,topMargin=1.65*cm,bottomMargin=1.6*cm,
                          title='DigitScope | Machine Learning Assignment 2')
    story=[]
    add=lambda s,style='Body2': story.append(Paragraph(s,styles[style]))
    heading=lambda s: add(s,'Head2')
    ready=all([name.strip(),bits_id.strip(),github.strip(),app.strip(),screenshot is not None and screenshot.exists()])

    add('Machine Learning - Assignment 2','Title2')
    add('DigitScope: Handwritten-Digit Classification | BITS WILP M.Tech AIML','Sub2')
    if not ready:
        add('TEMPLATE ONLY - ADD YOUR OWN LINKS AND BITS VIRTUAL LAB SCREENSHOT BEFORE SUBMISSION','Banner2')
    add(f'<b>Student:</b> {escape(name or "[YOUR NAME]")} &nbsp;&nbsp; <b>BITS ID:</b> {escape(bits_id or "[YOUR BITS ID]")}')
    add('<b>Course:</b> AIMLZG565 - Machine Learning')
    heading('1. GitHub Repository Link')
    if github:
        safe=escape(github,quote=True)
        add(f'<link href="{safe}" color="blue">{escape(github)}</link>')
    else:
        add('[ADD YOUR WORKING GITHUB REPOSITORY LINK]')
    heading('2. Live Streamlit App Link')
    if app:
        safe=escape(app,quote=True)
        add(f'<link href="{safe}" color="blue">{escape(app)}</link>')
    else:
        add('[ADD YOUR WORKING STREAMLIT COMMUNITY CLOUD LINK]')
    heading('3. BITS Virtual Lab / Prayogshala Screenshot')
    if screenshot and screenshot.exists():
        from PIL import Image as PILImage
        with PILImage.open(screenshot) as im:
            iw,ih=im.size
        maxw=16.0*cm; maxh=12.2*cm
        ratio=min(maxw/iw,maxh/ih)
        story.append(Image(str(screenshot),width=iw*ratio,height=ih*ratio))
        add('Screenshot supplied by student. Confirm it shows execution inside BITS Virtual Lab.','Small2')
    else:
        placeholder=Table([[Paragraph('<b>INSERT YOUR OWN ACTUAL BITS VIRTUAL LAB EXECUTION SCREENSHOT HERE</b>',styles['Center2'])]],
                          colWidths=[16.1*cm],rowHeights=[6.1*cm])
        placeholder.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),LIGHT),('BOX',(0,0),(-1,-1),1,HexColor('#afc5ce')),
                                         ('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),18),('RIGHTPADDING',(0,0),(-1,-1),18)]))
        story.append(placeholder)
    story.append(PageBreak())
    heading('4. README / Assignment Report')
    heading('a. Problem statement')
    add('Predict which of ten digits (0-9) is represented by a compact 8 by 8 handwritten-character grid. '
        'Using a common stratified holdout split, train Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes, '
        'and SVM. Evaluate all five with six consistent performance measures and deploy saved inference pipelines.')
    heading('b. Dataset description')
    add('Data source: <link href="https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits" color="blue">UCI Optical Recognition of Handwritten Digits</link> '
        '(Alpaydin and Kaynak, 1998; DOI 10.24432/C50P49). The working dataset is a 1,797-record copy of the '
        '<b>original UCI test subset</b> distributed in scikit-learn, not the full original collection.')
    add('Features: 64 numeric 8-by-8 pixel aggregate values, 0-16. Target: digit 0-9. Dataset rows: 1,797; '
        'missing values: 0; duplicates: 0. Stratified split: 1,347 training and 450 held-out test rows '
        '(test_size=0.25, random_state=73).')
    heading('Preprocessing and evaluation')
    add('All five classifiers use an identical pipeline: median-value imputation, StandardScaler and classifier, '
        'with preprocessing fitted on training rows only. AUC is macro-average one-vs-rest using probability estimates. '
        'Precision, Recall and F1 are macro-averaged over digits 0-9. MCC uses the multiclass formulation. '
        'Saved pipeline files are loaded by Streamlit without retraining.')
    heading('c. GitHub Repository Link')
    if github:
        safe=escape(github,quote=True)
        add(f'<link href="{safe}" color="blue">{escape(github)}</link>')
    else:
        add('[SAME GITHUB LINK AS SECTION 1]')
    heading('d. Model comparison: results on 450 held-out records')
    metrics=pd.read_csv(RESULTS_PATH)
    columns=['Model','Accuracy','AUC','Precision','Recall','F1','MCC']
    header=['ML Model Name','Accuracy','AUC','Precision','Recall','F1','MCC']
    table_rows=[[Paragraph(escape(v),styles['CellHeader2']) for v in header]]
    for _,r in metrics.iterrows():
        table_rows.append([Paragraph(escape(str(r['Model'])),styles['Cell2'])]+[
            Paragraph(f"{r[c]:.4f}",styles['Cell2']) for c in columns[1:]
        ])
    tbl=Table(table_rows,repeatRows=1,colWidths=[4.8*cm]+[1.85*cm]*6,hAlign='LEFT')
    tbl.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('TEXTCOLOR',(0,0),(-1,0),colors.white),
          ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,HexColor('#f0f5f7')]),
          ('GRID',(0,0),(-1,-1),0.35,HexColor('#c9d4da')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
          ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
    story.append(tbl)
    add('The table contains results produced by the provided, executed training script. Rerun it on your own VM '
        'after any changes and replace these numbers if they differ.','Small2')
    heading('Per-model observations')
    import joblib
    from sklearn.metrics import recall_score
    test_data = pd.read_csv(TEST_PATH)
    y_actual = test_data[TARGET]
    for name in ['Logistic Regression','Decision Tree','KNN','Gaussian Naive Bayes','Support Vector Machine']:
        r=metrics.loc[metrics['Model']==name].iloc[0]
        model=joblib.load(MODEL_DIR / MODEL_FILES[name])
        prediction=model.predict(test_data[FEATURES])
        rec=recall_score(y_actual,prediction,labels=list(range(10)),average=None,zero_division=0)
        low=int(rec.argmin()); high=int(rec.argmax())
        wrong=int((prediction != y_actual.to_numpy()).sum())
        add(f'<b>{name}:</b> {wrong} out of {len(test_data)} digits were classified incorrectly. '
            f'Weakest recall was on digit {low} ({rec[low]:.3f}); strongest was on digit {high} ({rec[high]:.3f}). '
            f'Macro F1 = {r["F1"]:.4f} and MCC = {r["MCC"]:.4f}.')
    add('<b>Overall winner:</b> SVM for this holdout split (Accuracy 0.9778, AUC 0.9996, F1 0.9778, '
        'MCC 0.9754). KNN ranks second in accuracy. These findings should be interpreted with the class-level '
        'confusion matrix, and need not extend to arbitrary new handwriting.')
    heading('e. Streamlit application and usage')
    add('The app provides labeled CSV upload or the bundled test file, five-model selector, Accuracy, AUC, '
        'Precision, Recall, F1, MCC, confusion matrix, per-digit classification report, prediction download, '
        'all-model comparison, and 8-by-8 digit preview. The test CSV must have the 64 pixel features and the '
        'digit target; missing or nonnumeric features trigger a validation error. Streamlit only calls predict '
        'and predict_proba on pre-trained models, never fit().')
    add('<b>Reproduce:</b> Install with <font face="Courier">python -m pip install -r requirements.txt</font>; '
        'train using <font face="Courier">python train.py</font>; run with '
        '<font face="Courier">python -m streamlit run app.py</font>. The repository includes the executed notebook, '
        'dataset, 450-row labeled test CSV, metrics CSV, saved trained pipelines and a README.')
    heading('Dataset attribution and scope')
    add('E. Alpaydin &amp; C. Kaynak (1998), Optical Recognition of Handwritten Digits, UCI Machine Learning Repository. '
        'DOI 10.24432/C50P49. Upstream license: CC BY 4.0. The analysis uses the 1,797-record scikit-learn subset.')

    def foot(canvas, doc):
        canvas.saveState(); w,h=A4
        canvas.setStrokeColor(HexColor('#c7d5dc'))
        canvas.line(1.7*cm,1.35*cm,w-1.7*cm,1.35*cm)
        canvas.setFont('Helvetica',8); canvas.setFillColor(GRAY)
        canvas.drawString(1.7*cm,1.0*cm,'DigitScope | ML Assignment 2')
        canvas.drawRightString(w-1.7*cm,1.0*cm,f'Page {doc.page}')
        canvas.restoreState()
    pdf.build(story,onFirstPage=foot,onLaterPages=foot)
    print('Report created:',output, 'Final-ready inputs provided:',ready)


def main():
    parser=ArgumentParser()
    parser.add_argument('--name',default='')
    parser.add_argument('--bits-id',default='')
    parser.add_argument('--github',default='')
    parser.add_argument('--app',default='')
    parser.add_argument('--screenshot',type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'docs'/'REPORT_TEMPLATE.pdf')
    args=parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    make_pdf(args.name,args.bits_id,args.github,args.app,args.screenshot,args.output)

if __name__=='__main__':
    main()
