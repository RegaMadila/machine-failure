import os
import sys
from src.data_loader import load_data, clean_data
from src.feature_eng import add_physics_features, encode_features
from src.processing import prepare_data, get_categorical_indices
from src.model import train_model, evaluate_model
from src.model_xgb import train_xgb_smote, evaluate_models_comparison
from src.eda import perform_eda

# Constants / Configuration
DATA_PATH = 'ai4i2020.csv'
MODEL_SAVE_PATH = 'machine_failure_model.cbm'

def main():
    print("Machine Failure Prediction Pipeline Starting...")
    
    # 1. Load Data
    try:
        df = load_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Clean Data
    df = clean_data(df)
    
    # 2.1 EDA
    # perform_eda(df)
    
    # 3. Feature Engineering
    df = add_physics_features(df)
    
    # CatBoost
    print("\n[Mode 1] Preparing data for CatBoost...")
    X_train_cat, X_test_cat, y_train, y_test = prepare_data(df)
    cat_features = get_categorical_indices(X_train_cat)
    cat_model = train_model(X_train_cat, y_train, X_test_cat, y_test, cat_features)
    evaluate_model(cat_model, X_test_cat, y_test)
    cat_model.save_model(MODEL_SAVE_PATH)
    
    # XGBoost + SMOTE
    print("\n Preparing data for XGBoost...")
    df_encoded = encode_features(df)
    X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = prepare_data(df_encoded)
    
    # Build XGBoost Model
    xgb_model = train_xgb_smote(X_train_xgb, y_train_xgb, X_test_xgb, y_test_xgb)
    
    # Ensemble Comparison
    evaluate_models_comparison(cat_model, xgb_model, X_test_cat, X_test_xgb, y_test)

if __name__ == "__main__":
    main()
