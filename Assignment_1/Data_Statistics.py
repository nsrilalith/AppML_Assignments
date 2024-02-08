import pandas as pd
from StatsReport import StatsReport


excel_data = "diabetic_data.xlsx"
diabetes_df = pd.read_excel(excel_data)

# print(diabetes_df.shape)

labels = diabetes_df.columns
report = StatsReport()

# print(diabetes_df['age'].dtype)

for i in labels:
    thisCol = diabetes_df[i]
    report.addCol(i, thisCol)

print(report.to_String())

