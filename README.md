# Rare Failure Hunter: Predictive Maintenance Pipeline

## Overview
This project is an end-to-end Machine Learning pipeline designed to predict **machine failures** in industrial settings. The core challenge addressed is the **extreme class imbalance** (only ~3.4% of machines fail), which renders traditional accuracy metrics useless.

The solution implements a robust **Ensemble Strategy** combining:
1.  **CatBoost Classifier**: Leveraging native categorical feature handling.
2.  **XGBoost + SMOTE**: Using Synthetic Minority Over-sampling Technique (SMOTE) to synthetically balance training data.

By averaging the probabilities of these two distinct models and applying **Dynamic Threshold Tuning**, the pipeline achieves high recall while maintaining precision, minimizing expensive false negatives in a maintenance context.

---

## Data Source
The dataset is the **AI4I 2020 Predictive Maintenance Dataset**, originally donated to the UCI Machine Learning Repository.

*   **Original Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
*   **Kaggle Mirror**: [Stephan Matzka - AI4I 2020](https://www.kaggle.com/stephanmatzka/predictive-maintenance-dataset-ai4i-2020)

**Note**: The project includes an automated loader that downloads the dataset directly from Kaggle using the `opendatasets` library.

---

## Project Structure
The codebase is refactored from a monolithic notebook into a modular Python package to simulate a production-ready environment.

```text
machine_failure/
├── src/
│   ├── data_loader.py    # Auto-downloads data from Kaggle & cleans it
│   ├── feature_eng.py    # Physics-based feature engineering (Power, Strain, etc.)
│   ├── processing.py     # Stratified Train/Test splitting & Preprocessing
│   ├── eda.py            # Exploratory Data Analysis (generates plots)
│   ├── model.py          # CatBoost training & optimization logic
│   └── model_xgb.py      # XGBoost + SMOTE pipeline & Ensemble logic
├── main.py               # Orchestrator script
├── requirements.txt      # Dependency management
└── README.md             # Documentation
```

---

## Technical Flow

### 1. Data Ingestion & EDA
*   **Automated Download**: Checks for local data; if missing, authenticates with Kaggle API and downloads the CSV.
*   **Cleaning**: Renames columns to snake_case (e.g., `Rotational speed [rpm]` -> `rotational_speed_rpm`) and drops non-predictive IDs (`UDI`, `Product ID`).
*   **EDA**: Generates Target Distribution and Correlation Matrix plots to `reports/`.

### 2. Feature Engineering
Domain knowledge is applied to create physics-based synthetic features:
*   **Power (W)**: `Torque * Rotational Speed`
*   **Temperature Difference (K)**: `Process Temp - Air Temp`
*   **Strain / Wear Stress**: `Torque * Tool Wear`

### 3. Model Training (Dual-Pipeline)
The pipeline splits into two parallel paths to maximize performance:

*   **Path A (CatBoost)**: 
    *   Uses native support for categorical variables (`Type`).
    *   Optimized with `auto_class_weights='Balanced'`.
    *   **Auto-Thresholding**: Instead of the default 0.5, it calculates the optimal probability threshold to maximize F1-Score.

*   **Path B (XGBoost + SMOTE)**:
    *   **One-Hot Encoding**: Converts categorical variables for XGBoost compatibility.
    *   **SMOTE (Synthetic Minority Over-sampling)**: Generates synthetic examples of "Failures" in the training set only, preventing the model from being biased toward the majority class.
    *   **Auto-Thresholding**: Independently tunes its decision boundary.

### 4. Ensemble & Evaluation
The final prediction uses a **Weighted Soft Voting** mechanism:
1.  Averages the probability outputs from CatBoost and XGBoost.
2.  Performs a final pass of **Precision-Recall Curve optimization** to find the absolute best threshold for the combined probabilities.
3.  Outputs three distinct reports: CatBoost Standalone, XGBoost Standalone, and Ensemble.

---

## Installation & Setup

### Steps
1.  **Clone the Repository**
2.  **Create a Virtual Environment** (Recommended):
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```
3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the Pipeline**:
    ```bash
    python main.py
    ```

---

