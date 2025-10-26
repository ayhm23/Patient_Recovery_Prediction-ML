# Patient Recovery Prediction - ML

Predicting patient recovery outcomes using machine learning.

## Project overview

This repository contains a reproducible machine-learning pipeline to predict patient recovery outcomes from clinical and demographic data. The codebase includes data ingestion, exploratory data analysis (EDA), preprocessing, model training, evaluation, and scripts to save and load trained models for inference.

Key objectives:
- Analyze patient features and a recovery outcome target.
- Perform EDA and visualizations to understand data patterns.
- Preprocess and engineer features for machine learning.
- Train and compare multiple supervised learning models.
- Select the best model, evaluate it with appropriate metrics, and save the final artefact.
- Provide reproducible code and documentation.

## Repository structure

```
├── data/
│   ├── raw/           # original dataset(s)
│   └── processed/     # cleaned and engineered datasets
├── src/               # scripts for EDA, preprocessing, modelling
│   ├── 0_data_loading.py
│   ├── 1_eda.py
│   ├── 2_preprocessing.py
│   ├── 3_model_training.py
│   └── 4_model_evaluation.py
├── models/            # saved/trained model artefacts (pickle / joblib)
├── visualizations/    # plots and charts from EDA and results
├── output/            # final reports, logs, metrics
├── requirements.txt   # Python dependencies
└── README.md           # project overview (this file)
```

## Dataset

Describe the dataset used in the project. Replace the placeholders below with the actual dataset information.

- Source: (e.g., hospital database, public dataset — add link or citation)
- Features: e.g., age, sex, diagnosis, treatment type, comorbidities, lab results
- Target: recovery outcome (binary: Recovered / Not Recovered, or multiclass)
- Size: number of rows and columns
- Known issues: class imbalance, missing data patterns, outliers, or data leakage risks

Tip: If you want, I can help fill these fields using your dataset or a project report.

## Exploratory data analysis (EDA) & preprocessing

Typical EDA steps:
- Summaries (counts, means, medians), distributions and missing-value analysis
- Visualizations: histograms, boxplots, correlation heatmaps, class-balance plots
- Check for class imbalance and possible confounders

Typical preprocessing steps:
- Missing-value handling (imputation strategies or removal)
- Encoding categorical variables (one-hot / ordinal / target encoding as appropriate)
- Feature scaling (StandardScaler / MinMaxScaler) where required
- Feature engineering (interactions, aggregations, derived features)
- Train/test split and cross-validation strategy (e.g., stratified k-fold)

Ensure the same preprocessing pipeline is applied at training and inference time (use sklearn Pipelines or custom transformers + joblib/pickle).

## Model development

Recommended workflow:
- Candidate algorithms: Logistic Regression, Random Forest, XGBoost/LightGBM, SVM, etc.
- Hyperparameter tuning: GridSearchCV, RandomizedSearchCV, or Optuna for more advanced searches
- Evaluation metrics: accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix, and class-specific metrics (particularly for imbalanced problems)
- Model selection: choose a model balancing performance and clinical interpretability

Save the final model together with the fitted preprocessing pipeline (e.g., a single joblib file containing a Pipeline).

## Example usage

1. Clone the repository:
   git clone https://github.com/ayhm23/Patient_Recovery_Prediction-ML.git
   cd Patient_Recovery_Prediction-ML

2. Install dependencies:
   pip install -r requirements.txt

3. Prepare data:
   - Place raw data into data/raw/
   - Run data-loading and preprocessing scripts:
     python src/0_data_loading.py
     python src/1_eda.py
     python src/2_preprocessing.py

4. Train models:
   python src/3_model_training.py

5. Evaluate models:
   python src/4_model_evaluation.py

6. The trained model artifact will be saved in models/, and visualizations / reports will be in visualizations/ and output/.

7. For inference: load the saved pipeline and call predict / predict_proba on new, preprocessed samples.

## Reproducibility & environment

- Python version: 3.8+ recommended (specify exact version used)
- Key libraries: pandas, numpy, scikit-learn, matplotlib/seaborn, xgboost/lightgbm (if used), joblib or pickle
- For reproducible results: set random seeds (e.g., numpy and model random_state)
- Consider using venv / conda or Docker to capture the exact environment

## Results & interpretation

When documenting final results:
- Report chosen model and key metrics (e.g., "Random Forest — ROC-AUC: 0.87, recall (Recovered): 0.82")
- Present a confusion matrix and class-specific metrics
- Provide feature importance or SHAP explanations for model interpretability
- Summarize clinical implications and limitations (sample size, bias, generalizability)
- Suggest next steps and future work

## Contributing

Contributions are welcome. Suggested workflow:
- Open an issue to discuss major changes
- Create feature branches from main
- Add tests and update documentation for new code
- Open a pull request describing the change

## License & acknowledgements

- License: add your chosen license (e.g., MIT)
- Acknowledgements: credit dataset sources, collaborators, frameworks, or advisors

---

Last updated: 2025-10-26

If you’d like, I can:
- Fill in dataset-specific details and actual model results,
- Convert this into a release-ready README with badges and sample outputs,
- Or update the README directly in your repository (I can create a PR if you want).