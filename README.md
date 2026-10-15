# ClaimLens — Evidence Relationship Learning for Insurance Claims

ClaimLens is a research prototype that helps an insurance investigator review a claim by organizing evidence and highlighting relationships between reported facts and supporting material. It is **decision support**, not an automatic fraud verdict.

## What it does
- Create a claim with incident details and attach supporting files.
- Review evidence sources such as an accident report, repair invoice, vehicle photographs, and hospital records.
- Classify claim-detail/evidence relationships as **Supported**, **Contradicted**, or **Insufficient evidence**.
- Display connected evidence relationships and source-linked findings.
- Generate a review summary and recommended follow-up evidence.
- Train and evaluate a text-based baseline model on a clearly labelled synthetic benchmark, with metrics and experiment logs.

> Research limitation: the included dataset is synthetic for reproducibility and demonstration. Results do not establish real-world insurance effectiveness. Uploaded documents are not sent to an external service; this prototype uses metadata/rules for the demo workflow and does not claim production-grade OCR or visual damage recognition.

## Run locally

### Prerequisites
- Python 3.10–3.12 recommended
- Node.js 20+
- Docker Desktop (optional)

### 1. Start API
Windows PowerShell:
```powershell
cd backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
CMD activation:
```cmd
cd backend
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
API docs: http://127.0.0.1:8000/docs

### 2. Start web app (new terminal)
```powershell
cd frontend
npm install
npm run dev
```
Open the URL printed by Vite (usually http://localhost:5173).

### Docker
From the project root:
```bash
docker compose up --build
```
Open http://localhost:5173. The API is available at http://localhost:8000.

## Research pipeline
1. Generate a reproducible synthetic relationship dataset.
2. Split by examples with a fixed random seed.
3. Convert evidence-pair text into TF-IDF features.
4. Train Logistic Regression baseline and compare against a majority-class baseline.
5. Report macro-F1, accuracy, classification report, and confusion matrix.
6. Run claim evidence through transparent demo rules and show source-linked findings.

## Important
- A contradiction is not proof of fraud.
- Similar/reused images are not proof of fraud.
- “Insufficient evidence” is intentionally separate from “Contradicted”.
- Investigators must verify original documents and make final decisions.
