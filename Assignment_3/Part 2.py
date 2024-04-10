import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

# Load the dataset
file_path = "BattingSalariesData.xlsx"
df = pd.read_excel(file_path)

# One-Hot Encoding for categorical variables 'lgID' and 'teamID' 
encoder = OneHotEncoder()
encoded_data = encoder.fit_transform(df[['lgID', 'teamID']])

# Create a new DataFrame with the encoded columns
encoded_df = pd.DataFrame(encoded_data.toarray(), columns=encoder.get_feature_names_out(['lgID', 'teamID']))

# Combine encoded columns with the original DataFrame
df_combined = pd.concat([df, encoded_df], axis=1)
df_combined.drop(['lgID', 'teamID'], axis=1, inplace=True)

# Remove rows with NaN values in 'Salary'
df_combined.dropna(subset=['Salary'], inplace=True)

# Initialize lists to store results
years = []
dataset_sizes = []
mse_values = []
r2_values = []

# Print header for the outputs
print(f"{'Year':<6} {'Size':<6} {'MSE':<20} {'R2':<6}")

# Loop through each year
for year in df_combined['yearID'].unique():
    # Filter the data for the given year
    yearly_data = df_combined[df_combined['yearID'] == year]

    # Define the predictors and target variable
    X = yearly_data.drop(['Salary', 'playerID', 'yearPlayer', 'yearID'], axis=1)
    y = yearly_data['Salary']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=22222)

    # Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Evaluation
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Store the results
    years.append(year)
    dataset_sizes.append(len(yearly_data))
    mse_values.append(mse)
    r2_values.append(r2)
    
    # Print results for each year
    print(f"{year:<6} {len(yearly_data):<6} {mse:<20} {r2:<6}")

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(years, mse_values, marker='o')
plt.title('MSE vs Year for Salary Prediction')
plt.xlabel('Year')
plt.ylabel('Mean Squared Error')
plt.grid(True)
plt.show()
