import pandas as pd
from StatsReport import StatsReport

excel_data = "/Users/srilalithnampally/Classes/AppML_Assignments/Assignment_3/BattingSalariesData.xlsx"
diabetes_df = pd.read_excel(excel_data)

# print(diabetes_df.shape)

labels = diabetes_df.columns
report = StatsReport()

# print(diabetes_df['age'].dtype)

for i in labels:
    thisCol = diabetes_df[i]
    report.addCol(i, thisCol)

print(report.to_String())

toExcel_file = "/Users/srilalithnampally/Classes/AppML_Assignments/Assignment_3/data_report_salaries.xlsx"
print(f'\nSuccessfully generated a data statistics report\nTranscribing data into {toExcel_file} ....')

report.write_to_file(toExcel_file)
