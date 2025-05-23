import pandas as pd
import os

# Create the data
data = {
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [25, 30, 35],
    "City": ["New York", "Los Angeles", "Chicago"]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Get current working directory
current_dir = os.getcwd()
print("Saving to directory:", current_dir)

# Define the full path for the CSV file
csv_file_path = os.path.join(current_dir, "people_data.csv")

# Save the DataFrame to CSV
df.to_csv(csv_file_path, index=False)

print("CSV file created:", csv_file_path)
