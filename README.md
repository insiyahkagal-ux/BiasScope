# BiasScope

BiasScope is a reproducible machine learning fairness project that measures geographic disparities in model performance using Bangladesh DHS 2022 data.

---

## Objective

To evaluate whether a machine learning model performs equally well across different geographic regions.

---

## Dataset

- Source: Bangladesh DHS 2022
- Samples: 30,078
- Features:
  - Age (v012)
  - Urban/Rural status (v025)
- Target:
  - Education level (v106)
- Group variable:
  - Region (v024)

---

## Method

- Random Forest Classifier
- Logistic Regression
- Train-test split (70/30)
- One-hot encoding for categorical features
- Balanced class weighting (Random Forest)

---

## Key Metrics

- Overall accuracy
- Region-wise accuracy
- Region-wise error rate
- Fairness Gap = max(region error) − min(region error)

---

## Main Result

The model shows systematic variation in error rates across geographic regions, indicating uneven predictive performance.

Observed fairness gap: ~0.10

Regions with higher error rates consistently differ from lower-error regions, showing geographic disparity in model behavior.

---

## How to Run

```bash
python biascope.py