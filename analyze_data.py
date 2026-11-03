import pandas as pd

# Load our payment dataset
data = pd.read_csv("data/transactions.csv")

# Show the first 5 transactions
print(data.head())

# Show basic information about the dataset
print("\nDataset Information:")
(data.info())

# Show how many payments were recovered
print("\nRecovery Status:")
print(data["recovery_status"].value_counts())