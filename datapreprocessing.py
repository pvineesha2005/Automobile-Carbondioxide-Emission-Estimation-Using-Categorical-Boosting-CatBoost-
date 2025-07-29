import pandas as pd

# Load datasets
df1 = pd.read_csv("dataset1.csv")  
df2 = pd.read_csv("dataset2.csv",sep="\t") 
if "YEAR" in df1.columns:
    df1 = df1.drop(columns=["YEAR"])

#Rename dataset2 columns to match dataset1 format 
df2 = df2.rename(columns={
    "Make": "MAKER",
    "Model": "MODEL",
    "Vehicle Class": "VEHICLECLASS",
    "Engine Size(L)": "ENGINESIZE",
    "Cylinders": "CYLINDERS",
    "Transmission": "TRANSMISSION",
    "Fuel Type": "FUEL",
    "Fuel Consumption Comb (L/100 km)": "FUELCONSUMPTION",
    "CO2 Emissions(g/km)": "COEMISSIONS"
})

# --- final columns in the datasets
final_cols = [
    "MAKER", "MODEL", "VEHICLECLASS", "ENGINESIZE",
    "CYLINDERS", "TRANSMISSION", "FUEL", "FUELCONSUMPTION", "COEMISSIONS"
]

# checking for any missing columns
missing = [col for col in final_cols if col not in df2.columns]
if missing:
    print(f"df2 is missing these columns: {missing}")
    exit()

# --- Standardize column format 
df1.columns = df1.columns.str.replace(r"\s+", "", regex=True).str.upper()
for df in [df1, df2]:
    df.columns = df.columns.str.replace(r"\s+", "", regex=True).str.upper()
    df["MODEL"] = df["MODEL"].str.upper()  
    df["MAKER"] = df["MAKER"].str.upper()  
    df["VEHICLECLASS"] = df["VEHICLECLASS"].str.upper()
    df["TRANSMISSION"] = df["TRANSMISSION"].str.upper()
    df["FUEL"] = df["FUEL"].str.upper()


#  Select only the needed columns
df1 = df1[final_cols]
df2 = df2[final_cols]

# Combining datasets
combined_df = pd.concat([df1, df2], ignore_index=True)

# Cleaning combined data
combined_df.dropna(inplace=True)
combined_df.drop_duplicates(inplace=True)

# Filter out invalid values ---
combined_df = combined_df[
    (combined_df["ENGINESIZE"] > 0) &
    (combined_df["FUELCONSUMPTION"] > 0) &
    (combined_df["CYLINDERS"] > 0)
]

combined_df.to_csv("co2_emission_combined.csv", index=False) # Saving final dataset
print("Combined and cleaned dataset saved as 'co2_emission_combined.csv'")
