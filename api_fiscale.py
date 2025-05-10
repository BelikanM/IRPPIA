
import os, json, re
import numpy as np
import faiss
import pandas as pd
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from fpdf import FPDF
import matplotlib.pyplot as plt
import seaborn as sns

app = FastAPI()

MODEL_PATH = "fiscal_model"
INDEX_PATH = "fiscal_index.faiss"

if os.path.exists(MODEL_PATH) and os.path.exists(INDEX_PATH):
    model = SentenceTransformer(MODEL_PATH)
    index = faiss.read_index(INDEX_PATH)
    if os.path.exists("questions.json") and os.path.exists("answers.json"):
        with open("questions.json", "r") as f:
            questions = json.load(f)
        with open("answers.json", "r") as f:
            answers = json.load(f)
    else:
        questions = []
        answers = []
else:
    model = None
    index = None
    questions = []
    answers = []

@app.post("/upload/")
async def upload_jsonl(file: UploadFile):
    content = await file.read()
    path = f"./{file.filename}"
    with open(path, "wb") as f:
        f.write(content)
    fine_tune_model(path)
    return {"status": "Fine-tuning terminé."}

class QuestionRequest(BaseModel):
    query: str
    parts: int = 1
    top_k: int = 3
    generate_pdf: bool = False

@app.post("/ask/")
def ask_question(req: QuestionRequest):
    return ask_fiscal_question(req.query, req.parts, req.top_k, req.generate_pdf)

def fine_tune_model(jsonl_path):
    global model, index, questions, answers
    with open(jsonl_path, "r", encoding="utf-8") as f:
        data = [json.loads(line.strip()) for line in f]

    questions = [item['input'] for item in data]
    answers = [item['output'] for item in data]

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(questions)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    model.save(MODEL_PATH)
    faiss.write_index(index, INDEX_PATH)
    with open("questions.json", "w") as f:
        json.dump(questions, f)
    with open("answers.json", "w") as f:
        json.dump(answers, f)

def ask_fiscal_question(query, parts=1, top_k=3, generate_pdf=False):
    q = query.lower()
    montant = extract_montant(q)
    if "irpp" in q and montant:
        total, tableau, explication = calcul_irpp_explique(montant, parts)
        return {
            "explication": explication,
            "tableau": tableau,
            "pdf_path": generer_rapport_pdf("Calcul IRPP", explication) if generate_pdf else None
        }
    if model and index:
        query_vec = model.encode([query])
        distances, indices = index.search(np.array(query_vec), top_k)
        results = []
        for idx in indices[0]:
            results.append({
                "question_similaire": questions[idx],
                "réponse": answers[idx]
            })
        return {"réponses": results}
    return {"message": "Modèle non entraîné. Veuillez téléverser un fichier JSONL."}

def extract_montant(text):
    chiffres = re.findall(r"\d{4,}", text.replace(" ", "").replace(",", ""))
    return int(chiffres[0]) if chiffres else None

def calcul_irpp_explique(revenu, parts=1, taux=0.15, abattement=300000):
    q = revenu / parts
    k = q * taux - abattement
    irpp = max(k, 0) * parts
    tableau = (["Étapes", "Valeurs"], [
        ["Revenu Net Global", f"{revenu:,} FCFA"],
        ["Nombre de Parts", parts],
        ["Quotient Familial", f"{q:,.2f} FCFA"],
        ["K (q * taux - abattement)", f"{k:,.2f} FCFA"],
        ["IRPP Final", f"{irpp:,.2f} FCFA"]
    ])
    explication = f"""
### Calcul de l'IRPP

1. Quotient familial : {revenu:,} / {parts} = {q:,.2f} FCFA
2. Calcul de K : ({q:,.2f} × {taux*100}%) - {abattement} = {k:,.2f} FCFA
3. IRPP avant parts : max({k:,.2f}, 0) = {max(k,0):,.2f} FCFA
4. IRPP final : {max(k,0):,.2f} × {parts} = {irpp:,.2f} FCFA
"""
    return round(irpp, 2), tableau, explication

def generer_rapport_pdf(titre, contenu, nom_fichier="rapport_fiscal.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, titre, ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", "", 12)
    for ligne in contenu.split("\n"):
        pdf.multi_cell(0, 10, ligne)
    chemin_pdf = f"./{nom_fichier}"
    pdf.output(chemin_pdf)
    return chemin_pdf
