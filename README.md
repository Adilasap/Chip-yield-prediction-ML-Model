# Die-Yield Prediction ML Model

A machine-learning workflow for predicting semiconductor chip/die failures from parametric measurements, wafer spatial context, and block-level readings.

## Project overview

This notebook implements and compares two approaches on the supplied dataset:

- **Model A — Die-level + spatial:** Uses parametric measurements, die coordinates, pre-test status, and spatial context. A LightGBM classifier produces failure probabilities and predictions. It does not use block-derived features.
- **Model B — Die-level + spatial + block-level:** Adds noise-normalized block-signal evidence and combines it with die-level and spatial evidence using regularized logistic fusion.

The notebook includes preprocessing, feature engineering, wafer-level validation, threshold selection, evaluation plots, model explanations, and separate prediction submissions.

## Dataset

The notebook expects these files in its data directory:

- `train.csv`
- `test.csv`
- `validation.csv`

**Dataset files are not included in this repository.** Add your authorized dataset locally in the expected location before running the notebook. Do not commit private, confidential, or competition-restricted data.

The input data includes die identifiers/coordinates, parametric measurements, pre-test labels, and (for Model B) block readings. The notebook is designed to avoid using wafer IDs as predictive features and to evaluate with wafer-held-out folds.

## Run

1. Open `Die_Yield_Complete_Workflow.ipynb` in Jupyter or Google Colab.
2. Place the required CSV files where the notebook's `ROOT` / `DATA` configuration expects them.
3. Run the notebook cells from top to bottom.

The first code cell installs pinned versions of the required Python packages. Review the package versions and your environment before running it.

## Main methods

- Parametric feature processing and shared failure-signal projection
- Wafer spatial features and pre-test neighborhood densities
- LightGBM for Model A
- Noise-covariance-normalized block scanning and regularized evidence fusion for Model B
- Wafer-level cross-validation and training-derived decision thresholds
- Precision-recall, calibration, confusion-matrix, spatial, and feature-contribution analyses

## Outputs

The notebook is written to produce trained model files, model comparison metrics, explanations, visualizations, and separate submission files for Model A and Model B.

## Important limitations

The notebook documents that the supplied test set was inspected during earlier development and that the supplied validation inputs duplicate test inputs. Therefore, reported results should be treated as reused-benchmark results, not independent confirmation of real-world performance. Validate on new, independent wafers before making production claims. The block scan also assumes circular adjacency in the supplied synthetic block layout; this assumption may not fit real devices.

## Technologies

Python, NumPy, pandas, SciPy, scikit-learn, LightGBM, joblib, Matplotlib.
