import pandas as pd

data = pd.read_excel("diabetic_data.xlsx")

unique_values_1 = sorted(list(data['rosiglitazone'].unique()))

unique_values_2 = sorted(list(data['insulin'].unique()))
# To see the unique values
print(unique_values_1)

print(unique_values_2)
