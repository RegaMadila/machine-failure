import pandas as pd

def add_physics_features(df):
    df_eng = df.copy()
    
    # Power = Torque * Rotational Speed
    if 'torque_nm' in df_eng.columns and 'rotational_speed_rpm' in df_eng.columns:
        df_eng['power'] = df_eng['torque_nm'] * df_eng['rotational_speed_rpm']
        
    # Temperature Difference = Process Temp - Air Temp
    if 'process_temperature_k' in df_eng.columns and 'air_temperature_k' in df_eng.columns:
        df_eng['temp_diff'] = df_eng['process_temperature_k'] - df_eng['air_temperature_k']
        
    # Wear Stress = Tool Wear * Torque
    if 'tool_wear_min' in df_eng.columns and 'torque_nm' in df_eng.columns:
        df_eng['wear_stress'] = df_eng['tool_wear_min'] * df_eng['torque_nm']
        
    print("Feature engineering complete. Added: power, temp_diff, wear_stress")
    return df_eng

def encode_features(df):
    df_encoded = pd.get_dummies(df, drop_first=True)
    
    df_encoded.columns = [
        col.replace('[', '').replace(']', '').replace(' ', '_').replace('K', '_k').replace('Nm', '_nm').replace('rpm', '_rpm').replace('min', '_min')
        for col in df_encoded.columns
    ]
    return df_encoded
