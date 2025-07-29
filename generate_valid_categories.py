import pandas as pd
import joblib

df = pd.read_csv("co2_emission_combined.csv") 

valid_categories = {
    'MAKER': df['MAKER'].str.upper().unique().tolist(),
    'MODEL': df['MODEL'].str.upper().unique().tolist(),
    'VEHICLECLASS': df['VEHICLECLASS'].str.upper().unique().tolist(),
    'TRANSMISSION': df['TRANSMISSION'].str.upper().unique().tolist(),
    'FUEL': df['FUEL'].str.upper().unique().tolist()
}

joblib.dump(valid_categories, 'valid_categories.pkl')
