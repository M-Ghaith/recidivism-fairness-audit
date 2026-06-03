# COMPAS Recidivism Fairness Audit

An end-to-end fairness audit of ML models trained on the COMPAS recidivism dataset, measuring and mitigating racial and gender bias in criminal justice predictions.

## Features

- **Exploratory Data Analysis** — distribution checks, correlation analysis, and demographic breakdowns by race and gender
- **ML Model Development** — nested cross-validation, GridSearchCV tuning, and evaluation with AUC, precision, recall, and F1
- **Fairness Analysis** — statistical parity and equalized odds metrics via Fairlearn; bias mitigation using resampling, reweighting, and exponentiated gradients
- **Model Interpretability** — LIME explanations, feature importance, and epistemic uncertainty estimation

## Tech Stack

- **Python 3.x**
- **scikit-learn** — model training, cross-validation, hyperparameter tuning
- **Fairlearn** — fairness metrics (`MetricFrame`, demographic parity, equalized odds) and bias mitigation
- **LIME** — local model explainability
- **pandas / NumPy** — data manipulation
- **Jupyter Notebook** — interactive analysis

## Quick Start

```bash
# Clone the repository
git clone https://github.com/m-ghaith/recidivism-fairness-audit.git
cd recidivism-fairness-audit

# Install dependencies
pip install jupyter scikit-learn fairlearn lime pandas numpy matplotlib

# Launch Jupyter
jupyter notebook
```

Open the notebooks in this order:
1. `EDA_analysis.ipynb`
2. `ML_models.ipynb`
3. `resampling_analysis.ipynb` or `resampling_race_analysis.ipynb`
4. `Model_fairness_analysis.ipynb` or `Model_fairness_race_analysis.ipynb`

## Future Improvements

- [ ] Add a `requirements.txt` for reproducible dependency installation
- [ ] Package fairness utilities (`fairness_metrics.py`, `metricFrame_vis.py`) with unit tests
- [ ] Extend bias mitigation to post-processing techniques (e.g., threshold optimization)
