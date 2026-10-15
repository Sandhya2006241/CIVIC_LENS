"""Reproducible synthetic benchmark for evidence relationship classification."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import joblib

LABELS = ["Supported", "Contradicted", "Insufficient evidence"]
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_PATH = DATA_DIR / "relationship_model.joblib"
METRICS_PATH = DATA_DIR / "experiment_metrics.json"

def build_synthetic_dataset(seed: int = 42):
    rng = np.random.default_rng(seed)
    examples = []
    patterns = {
      "Supported": [
        ("Report says front bumper damage", "Photo shows visible front bumper damage"),
        ("Invoice lists front bumper replacement", "Repair invoice confirms front bumper replacement"),
        ("Accident report date is 12 April", "Hospital record records incident on 12 April"),
        ("Vehicle registration is KA 01 AB 1234", "Report and policy show KA 01 AB 1234"),
        ("Claim says left door is damaged", "Photograph clearly shows damage on left door"),
        ("Reported repair cost is 75000 rupees", "Invoice total is 75000 rupees"),
      ],
      "Contradicted": [
        ("Report says front bumper damage", "Invoice describes rear bumper replacement"),
        ("Claim incident date is 12 April", "Attached record gives incident date as 15 April"),
        ("Vehicle registration is KA 01 AB 1234", "Photograph plate reads KA 01 CD 9876"),
        ("Claim reports left door damage", "Photo clearly shows only right door damage"),
        ("Claim says repair cost is 75000 rupees", "Invoice total is 25000 rupees"),
        ("Report says vehicle was at Coimbatore", "Document places the vehicle in Chennai at that time"),
      ],
      "Insufficient evidence": [
        ("Claim reports rear quarter panel damage", "Uploaded photograph does not show the rear quarter panel"),
        ("Claim incident date is 12 April", "No dated independent record is available"),
        ("Claim says vehicle was repaired", "Invoice attachment is missing or unreadable"),
        ("Claim reports frame damage", "Image is blurred and damage cannot be assessed"),
        ("Reported repair cost is 75000 rupees", "Only a partial invoice page is supplied"),
        ("Claim says another vehicle was involved", "No witness statement or incident record is attached"),
      ]
    }
    for label, pairs in patterns.items():
        for i in range(160):
            a,b=pairs[i % len(pairs)]
            # small wording variation to create a useful but intentionally modest baseline
            prefixes=["Evidence comparison: ", "Claim detail: ", "Review pair: ", ""]
            suffixes=[".", " in the submitted evidence.", " based on the available record.", ""]
            text=f"{rng.choice(prefixes)}{a}; evidence states: {b}{rng.choice(suffixes)}"
            examples.append((text,label))
    rng.shuffle(examples)
    return [x[0] for x in examples], [x[1] for x in examples]

def train_and_evaluate():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    texts, labels = build_synthetic_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=5000, sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)
    cm = confusion_matrix(y_test, pred, labels=LABELS).tolist()
    metrics = {
      "dataset": "ClaimLens synthetic evidence-pair benchmark",
      "samples": len(texts), "train_samples": len(X_train), "test_samples": len(X_test),
      "seed": 42, "model": "TF-IDF (1-2 grams) + class-weighted Logistic Regression",
      "labels": LABELS,
      "accuracy": round(float(accuracy_score(y_test,pred)),4),
      "macro_f1": round(float(f1_score(y_test,pred,average="macro")),4),
      "baseline_accuracy": round(float(accuracy_score(y_test,baseline_pred)),4),
      "baseline_macro_f1": round(float(f1_score(y_test,baseline_pred,average="macro")),4),
      "confusion_matrix": cm,
      "classification_report": classification_report(y_test,pred,labels=LABELS,output_dict=True,zero_division=0),
      "caveat": "Synthetic benchmark only; not a real-world performance estimate."
    }
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    return model, metrics

def get_model_and_metrics():
    if MODEL_PATH.exists() and METRICS_PATH.exists():
        return joblib.load(MODEL_PATH), json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return train_and_evaluate()

def predict_relationship(claim_text: str, evidence_text: str):
    model, _ = get_model_and_metrics()
    joined = f"Claim detail: {claim_text}; evidence states: {evidence_text}"
    probabilities = model.predict_proba([joined])[0]
    classes = list(model.classes_)
    idx = int(np.argmax(probabilities))
    label = classes[idx]
    confidence = float(probabilities[idx])
    # A conservative abstention guard: weak confidence means we do not overstate the result.
    if confidence < 0.48:
        label = "Insufficient evidence"
    return {"label": label, "confidence": round(confidence, 3), "model": "TF-IDF + Logistic Regression baseline"}
