import pandas as pd
import os

# Create the data
data = {
    "Name": ["Alice", "Bob", "Charlie", "anshu"],
    "Age": [25, 30, 35, 21],
    "City": ["New York", "Los Angeles", "Chicago", "RKL"]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Get current working directory
current_dir = os.getcwd()
print("Current directory:", current_dir)

# Create the 'data' folder if it doesn't exist
data_folder = os.path.join(current_dir, "data")
os.makedirs(data_folder, exist_ok=True)  # This will create it if missing

# Define the full path inside the 'data' folder
csv_file_path = os.path.join(data_folder, "people_data.csv")

# Save the DataFrame to CSV
df.to_csv(csv_file_path, index=False)

print("CSV file created at:", csv_file_path)
