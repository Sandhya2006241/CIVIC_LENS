# ClaimLens research framing

## Research question
Does uncertainty-aware three-way evidence relationship classification, paired with traceable source relationships, distinguish supported claims, genuine conflicts, and missing evidence more usefully than a binary match/mismatch baseline?

## Labels
- **Supported:** available evidence supports the claim detail.
- **Contradicted:** reliable evidence conflicts with the same fact in a way that cannot readily be reconciled.
- **Insufficient evidence:** the source is missing, unreadable, ambiguous, or does not cover the fact.

## Common research gap to investigate
Insurance fraud literature already includes structured-data classifiers, multimodal claim models, damage-image matching, graph-based fraud analysis, and explainable models. Therefore ClaimLens should not claim those broad capabilities as novel. Its testable focus is the evidence-level distinction between contradiction and insufficient evidence, with source traceability and direct evaluation of explanations. This is a proposed research direction, not a claim that no prior work exists.

## Evaluation plan
- Text relationship classifier: accuracy, macro-F1, per-class precision/recall, confusion matrix.
- Baselines: majority class and binary supported/not-supported.
- Uncertainty: abstention coverage versus error rate.
- Evidence explanations: source attribution correctness and proportion of statements supported by cited sources.
- Leakage prevention: split by claim identity, not by duplicate text pairs.
- Error analysis: ambiguous dates, incomplete invoices, out-of-frame photographs, and conflicting components.

## Limitations
The packaged benchmark is synthetic. It is useful to exercise the pipeline and demonstrate metrics, but not a substitute for an authorized real-world dataset. The image shown in the UI is illustrative stock photography; no model in this starter package diagnoses damage from the photograph. Real deployments require privacy controls, authorized data, audited extraction, calibration, human review, and legal/compliance assessment.

## Starter references for literature review
1. “Auto Insurance Fraud Detection with Multimodal Learning,” Data Intelligence, 2023. https://direct.mit.edu/dint/article/5/2/388/114793/Auto-Insurance-Fraud-Detection-with-Multimodal
2. L. Maiano et al., “A deep-learning–based antifraud system for car-insurance claims,” Expert Systems with Applications, 2023. https://www.sciencedirect.com/science/article/pii/S0957417423011466
3. “Automobile insurance fraud detection using data mining: a systematic literature review,” Intelligent Systems with Applications, 2024. https://www.sciencedirect.com/science/article/pii/S2667305324000164
4. “The value of cross-data set analysis for automobile insurance fraud detection,” 2022. https://www.sciencedirect.com/science/article/pii/S0275531922001556
5. “Data misrepresentation detection for insurance underwriting fraud prevention,” Decision Support Systems, 2022. https://www.sciencedirect.com/science/article/pii/S0167923622000690
6. “Automobile Insurance Fraud Detection Based on PSO-XGBoost Model and Interpretable Machine Learning Method,” Insurance: Mathematics and Economics, 2025. https://www.sciencedirect.com/science/article/abs/pii/S0167668724001112
7. “The Interpretable Artificial Neural Network in Vehicle Insurance Claim Fraud Detection Based on SHAP,” 2024. https://ojs.sgsci.org/journals/jitp/article/view/367
8. “Uncovering Insurance Fraud Conspiracy with Network Learning,” arXiv, 2020. https://arxiv.org/abs/2002.12789
9. “Inductive inference of gradient-boosted decision trees on graphs for insurance fraud detection,” arXiv, 2025. https://arxiv.org/abs/2510.05676
10. “Methodological challenges in explainable AI for fraud detection: a systematic literature review,” Artificial Intelligence Review, 2026. https://link.springer.com/article/10.1007/s10462-026-11516-7

Verify bibliographic metadata and read primary papers before citing in the final IEEE report.
