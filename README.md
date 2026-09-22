# Predicting Smartphone Addiction — Kaggle Playground Series S6E8

Binary classification pipeline predicting smartphone addiction probability
from behavioral features (screen time, social media/gaming usage, sleep,
notifications). Evaluated on ROC-AUC.

**Result:** Public AUC 0.96657 | Private AUC 0.96631 | **Top 0.38%** leaderboard

## Approach

1. **EDA** (`notebooks/eda.ipynb`) — diagnosed the missing-data mechanism
   (target-rate comparison + co-occurrence analysis) and confirmed
   missingness is unrelated to the target (near-MCAR/MAR), supporting a
   no-imputation strategy for tree-based models.
2. **Feature engineering** (`src/preprocess.py`) — 10 features derived
   from correlation structure among screen-time variables (ratios,
   differences, aggregations). A second round of interaction (product)
   features was tested and discarded after it reduced OOF AUC.
3. **Hyperparameter tuning** (`src/tune.py`) — Optuna (TPE sampler),
   50 trials, improved LightGBM AUC by +0.002 over default parameters.
4. **Training & ensembling** (`src/train.py`) — 5-fold Stratified CV with
   out-of-fold (OOF) evaluation throughout. Final model blends a
   3-seed-averaged, tuned LightGBM with a default-parameter XGBoost,
   weighted via OOF-optimized grid search (0.60 / 0.40).

## Key findings

- Missingness across all 12 features (4–19% per column) carries no
  predictive signal — confirmed via target-rate and co-occurrence analysis.
- Gain-based feature importance revealed a spurious high-importance signal
  in `age` (likely overfitting to noise), which was resolved indirectly
  once stronger engineered features were introduced.
- Hyperparameter tuning with a limited trial budget (~20 trials) failed to
  beat default parameters for CatBoost and XGBoost — only the well-tuned
  LightGBM run (50 trials) showed clear improvement, highlighting that
  tuning ROI depends heavily on search budget.
- A 3-model ensemble (adding CatBoost) improved OOF AUC by only +0.00002
  over the 2-model blend — discarded in favor of the simpler pipeline.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python src/tune.py    # optional: re-run hyperparameter search
python src/train.py   # train models and produce submission.csv
```

Expects `data/train.csv` and `data/test.csv` (Kaggle competition files, not included).
