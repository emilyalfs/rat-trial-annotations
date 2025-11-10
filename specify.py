# convert general object interaction into specific object interaction
#step 1
#update the column names and class names in XXX_2_model.csv
#ranme the columns "frame_id" as "frame" and "label" as "pred"
#change "none" to "None", "object_interaction" to "object" and save it as "XXX_2_model_updated.csv"

import pandas as pd


# Load the CSV files

csv_file = "path to XXX_2_model.csv"
df1 = pd.read_csv(csv_file)

# Rename columns
df1 = df1.rename(columns={"frame_id": "frame", "label": "pred"})

# Replace None/NaN with the string "None"
df1 = df1.where(pd.notna(df1), "None")

# Replace "none" with "None" in all string columns
df1 = df1.applymap(lambda x: x.replace("none", "None") if isinstance(x, str) else x)

# Replace "object_interactin" with "object" in all string columns
df1 = df1.applymap(lambda x: x.replace("object_interaction", "object") if isinstance(x, str) else x)

# Save the updated DataFrame
df1.to_csv("path to XXX_2_model_updated.csv", index=False)

print("Updated file saved as 'individual_results_with_frame.csv'")

#step 2
#add column names to XXX_1_keypoints.csv and save it as XXX_1_keypoints_header.csv

# Read the CSV with no headers
df = pd.read_csv("path to XXX_1_keypoints.csv", header=None)

# Add column names
df.columns = ["frame", "nose_x", "nose_y", "tail_x", "tail_y"]

# Save it back to CSV
df.to_csv("path to XXX_1_keypoints_header.csv", index=False)


#step 3

#merge XXX_2_model_updated.csv and XXX_1_keypoints_header.csv into XXX_1_merged.csv based on the column "frame"

csv_1 = "path to XXX_2_model_updated.csv"
csv_2 = "/path to XX_1_keypoints_header.csv"

df1 = pd.read_csv(csv_1)  # First CSV
df2 = pd.read_csv(csv_2)  # Second CSV

# Merge based on the common column (replace 'common_column' with the actual column name)
merged_df = pd.merge(df1, df2, on="frame", how="inner")  # 'inner' keeps only matching rows

merged_df = merged_df.fillna("None")

# Save the merged result

# df = df.rename(columns={'aggregated_class': 'predicted_class})
merged_df.to_csv("path to XXX_1_merged.csv", index=False)
print("Merge successful! Saved as merged_output.csv.")


#step 4



