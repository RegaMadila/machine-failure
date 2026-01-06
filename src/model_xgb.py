from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import numpy as np

def train_xgb_smote(X_train, y_train, X_test, y_test):
    print("\n--- Starting XGBoost + SMOTE Pipeline ---")
    
    # 1. SMOTE Oversampling
    print(f"Original Train Shape: {X_train.shape}")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"Resampled (SMOTE) Train Shape: {X_train_smote.shape}")
    
    # 2. XGBoost Training
    model = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    )
    
    print("Training XGBoost...")
    model.fit(X_train_smote, y_train_smote)
    print("XGBoost training complete.")
    
    return model

def evaluate_models_comparison(cat_model, xgb_model, X_test_cat, X_test_xgb, y_test):
    print("\n" + "="*40)
    print("       ENSEMBLE EVALUATION REPORT       ")
    print("="*40 + "\n")
    
    # 1. CatBoost Probabilities
    p_cat = cat_model.predict_proba(X_test_cat)[:, 1]
    
    # 2. XGBoost Probabilities
    p_xgb = xgb_model.predict_proba(X_test_xgb)[:, 1]
    
    # --- EVALUATE XGBOOST ---
    print("\n" + "="*40)
    print("       XGBOOST + SMOTE EVALUATION       ")
    print("="*40 + "\n")
   
    prec_xgb, rec_xgb, thresh_xgb = precision_recall_curve(y_test, p_xgb)
    f1_xgb = 2 * (prec_xgb * rec_xgb) / (prec_xgb + rec_xgb)
    opt_thresh_xgb = thresh_xgb[np.argmax(f1_xgb)]
    
    y_pred_xgb_opt = (p_xgb >= opt_thresh_xgb).astype(int)
    print(f"XGBoost Optimal Threshold: {opt_thresh_xgb:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred_xgb_opt))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, p_xgb):.4f}")

    # --- EVALUATE ENSEMBLE ---
    print("\n" + "="*40)
    print("       ENSEMBLE EVALUATION REPORT       ")
    print("="*40 + "\n")
    
    # 3. Ensemble (Average)
    p_ensemble = (p_cat + p_xgb) / 2
    
    # Threshold Tuning for Ensemble
    precisions, recalls, thresholds = precision_recall_curve(y_test, p_ensemble)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"Optimal Threshold Found: {optimal_threshold:.4f}")
    y_pred_ensemble = (p_ensemble >= optimal_threshold).astype(int)
    
    print("Ensemble (CatBoost + XGBoost) Report:")
    print(classification_report(y_test, y_pred_ensemble))
    
    auc = roc_auc_score(y_test, p_ensemble)
    print(f"Ensemble ROC-AUC Score: {auc:.4f}")
    
    return p_ensemble
