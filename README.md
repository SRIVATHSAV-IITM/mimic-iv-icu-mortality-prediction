# Explainable Deep Learning for ICU Mortality Prediction using MIMIC-IV v3.1

> A research-grade deep learning framework for **ICU mortality prediction** using the **latest MIMIC-IV v3.1** dataset with **Explainable AI (SHAP/LIME)** for transparent clinical decision support.

---

## 🚀 Overview

This project predicts **patient mortality risk in ICU settings** using multivariate clinical time-series and static patient features from **MIMIC-IV v3.1**.

The model combines:

* **Temporal signals** from ICU vitals/labs
* **Static demographics + admission features**
* **Deep learning based mortality classifier**
* **XAI explanations using SHAP/LIME**

The goal is to improve **clinical trust, interpretability, and early risk detection**.

---

## ✨ Key Features

* 🏥 ICU mortality prediction
* 📊 Latest **MIMIC-IV v3.1** benchmark dataset
* 🧠 Deep learning model for time-series + tabular fusion
* 🔍 Explainable AI using SHAP/LIME
* 📈 Training and evaluation pipeline
* 📉 Patient-level explanation visualizations
* 📝 Research-paper friendly structure

---

## 📂 Project Structure

```text
ICU-XAI-MortalityNet/
│── data/
│   ├── timeseries.npy
│   ├── static.csv
│   └── labels.csv
│
│── src/
│   ├── model.py
│   ├── train.py
│   └── explain.py
│
└── README.md
```

---

## 🧪 Dataset

* **Dataset:** MIMIC-IV v3.1
* **Source:** PhysioNet critical care benchmark
* **Data Type:**

  * ICU time-series vitals
  * lab measurements
  * demographics
  * mortality labels

> Experiments use de-identified ICU patient records from Beth Israel Deaconess Medical Center.

---

## ⚙️ Installation

```bash
pip install torch numpy pandas scikit-learn shap lime matplotlib
```

---

## ▶️ Run Training

```bash
python src/train.py
```

---

## 🔍 Run Explainability

```bash
python src/explain.py
```

This generates:

* SHAP feature importance
* local patient explanations
* clinical factor attribution

---

## 📊 Expected Results

Typical evaluation metrics:

* Accuracy
* Precision
* Recall
* F1-score
* AUROC

XAI helps identify important predictors such as:

* heart rate
* blood pressure
* SpO2
* creatinine
* age
* ICU length of stay

---

## 🧠 Research Contribution

This project contributes toward:

* interpretable healthcare AI
* trustworthy ICU risk modeling
* explainable clinical decision support
* deep learning in critical care

---

## 📌 GitHub Topics

`mimic-iv` `mimic-iv-v3-1` `icu-mortality` `explainable-ai` `xai` `deep-learning` `healthcare-ai`

---

## Maintainer

**SRIVATHSAV-IITM**

Repository import and maintenance by SRIVATHSAV-IITM.

Original MIT copyright notice is retained in `LICENSE`.
