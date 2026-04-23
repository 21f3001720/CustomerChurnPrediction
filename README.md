# Bayesian Customer Churn Model

**Adding uncertainty quantification to churn prediction using PyMC and MCMC sampling.**

Most churn models give you a score. This one gives you a score *and* tells you how much to trust it — a distinction that matters when deciding how to allocate retention spend.

---

## The core idea

A standard logistic regression outputs a single probability per customer. A Bayesian logistic regression outputs a *distribution* over that probability — a full posterior that captures what the model knows and, crucially, what it doesn't.

This means you can segment customers not just by predicted risk, but by confidence:

| Segment | Action |
|---|---|
| High risk, high confidence | Automated retention offer |
| High risk, uncertain | Human review before acting |
| Low risk | No action needed |

**Result: 400 customers flagged as both high-risk (>50% churn probability) and high-uncertainty — prioritised for human review before any retention spend.**

---

## Dataset

IBM Telco Customer Churn — 7,032 customers, 20 features, 26.6% churn rate.  
Source: [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## Results

### Posterior coefficients (94% HDI)

All HDI intervals exclude zero — every feature has a credible directional effect on churn.

| Feature | Posterior mean | Direction |
|---|---|---|
| tenure | -1.25 | Longer tenure → lower churn ✅ |
| Contract: month-to-month | +0.46 | No lock-in → higher churn ✅ |
| TotalCharges | +0.43 | Higher spend → higher churn |
| MonthlyCharges | +0.38 | Higher bill → more likely to leave |
| Contract: two-year | -0.38 | Locked in → lower churn ✅ |
| Fiber optic internet | +0.36 | Competitive market → higher churn |
| SeniorCitizen | +0.15 | Small but credible effect |

### Model comparison

| Metric | Frequentist LR | Bayesian LR |
|---|---|---|
| AUC | 0.8178 | 0.8189 |
| Brier score | 0.1462 | 0.1458 |
| Per-prediction uncertainty | ✗ | ✓ |
| Calibrated credible intervals | ✗ | ✓ |

AUC is essentially identical — the Bayesian advantage is not accuracy, it's knowing *when not to trust the prediction*.

### Convergence diagnostics

| Check | Result | Threshold |
|---|---|---|
| R-hat (all params) | 1.000 | < 1.01 |
| ESS bulk (min) | 1,960 | > 400 |
| ESS tail (min) | 2,193 | > 400 |
| Divergences | 0 | 0 |

---

## Repo structure

```
bayesian_churn/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── notebooks/
│   ├── 01_eda.py               # Churn rates by segment, feature correlations
│   ├── 02_bayesian_model.py    # Prior elicitation, PyMC model, MCMC sampling
│   ├── 03_inference.py         # Convergence diagnostics, posterior plots, uncertainty
│   └── 04_comparison.py        # Bayesian vs frequentist on AUC, Brier, calibration
├── outputs/
│   ├── trace.nc                # ArviZ InferenceData (saved posterior)
│   └── customer_uncertainty.csv # Per-customer risk + uncertainty scores
└── assets/
    ├── posterior_forest.png    # Coefficient plot with 94% HDI
    ├── trace_plots.png         # Chain mixing diagnostics
    ├── posterior_analysis.png  # Risk × uncertainty segmentation
    └── comparison.png          # Frequentist vs Bayesian comparison
```

---

## Visualisations

### Posterior coefficients
![Posterior forest plot](assets/posterior_forest.png)

### Risk × uncertainty segmentation
![Posterior analysis](assets/posterior_analysis.png)

### Frequentist vs Bayesian
![Comparison](assets/comparison.png)

### Convergence (trace plots)
![Trace plots](assets/trace_plots.png)

---

## Setup

```bash
pip install pymc arviz scikit-learn pandas matplotlib pytensor
python notebooks/02_bayesian_model.py   # ~4 min on CPU
python notebooks/03_inference.py
python notebooks/04_comparison.py
```

Tested with PyMC 5.28, ArviZ 0.21, Python 3.12.

---

## Why Bayesian?

Three scenarios where this approach has a concrete advantage over sklearn:

1. **Small data** — priors regularise naturally without cross-validated hyperparameters
2. **Prior knowledge** — domain expertise (e.g., "month-to-month contracts churn more") encodes directly into the model as priors, not just as feature engineering
3. **Decision-making under uncertainty** — when a churn intervention costs money, knowing the model is uncertain about a customer is itself actionable information

---

## Potential extensions

- **Hierarchical model**: partial pooling across customer segments (region, product tier) — priors on priors
- **Belief-state MDP**: Bayesian churn uncertainty maps naturally to a POMDP where the optimal retention policy depends on resolving belief uncertainty first — a natural RL extension
- **Online updating**: as new data arrives, the posterior becomes the next prior — no retraining from scratch
