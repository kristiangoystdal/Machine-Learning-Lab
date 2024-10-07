import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
csv_file_path = 'results.csv'
data = pd.read_csv(csv_file_path)

# Plot a specific column
column_name = 'Ridge R^2'  # Replace with your actual column name
data[column_name].plot(kind='line')

# Find the index of the highest value
max_index = data[column_name].idxmax()
max_value = data[column_name].max()

# Add a dashed line and label at the highest value
plt.axvline(x=max_index, color='r', linestyle='--', label=f'Max Value: {max_value:.2f}')
plt.legend()

# Add labels and title
plt.xlabel('Index')
plt.ylabel(column_name)
plt.title(f'Plot of {column_name}')

# Show the plot
plt.show()