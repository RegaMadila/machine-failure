from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def train_model(X_train, y_train, X_test, y_test, cat_features_indices):

    model = CatBoostClassifier(
        iterations=1500,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3,
        cat_features=cat_features_indices,
        auto_class_weights='Balanced',
        verbose=200,
        early_stopping_rounds=50,
        eval_metric='F1'
    )
    
    print("Starting training...")
    model.fit(X_train, y_train, eval_set=(X_test, y_test))
    print("Training complete.")
    return model

def evaluate_model(model, X_test, y_test):
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("\n" + "="*40)
    print("       MODEL EVALUATION REPORT       ")
    print("="*40 + "\n")
    
    print("Classification Report:")

    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"CatBoost Optimal Threshold: {optimal_threshold:.4f}")
    y_pred_opt = (y_prob >= optimal_threshold).astype(int)
    print(classification_report(y_test, y_pred_opt))
    
    auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC Score: {auc:.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    return y_pred, y_prob
