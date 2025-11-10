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


#step 4 for 2 object videos

#get the middle point of bounding boxes of each object
#and save them in l (lower object) and u(upper object)


import json

#load the json file
with open('path to annotated frame of XXX.mp4 video (json file)') as f:
    data = json.load(f)


#find the coordinates of 'lower_ob' and 'upper_ob'
shapes = {s['label'].lower(): s['points'] for s in data['shapes']}
lower = shapes.get('lower_ob')
upper = shapes.get('upper_ob')

#find the middle point of each bounding box

l = [(lower[0][0]+lower[1][0])/2, (lower[0][1]+lower[1][1])/2] if lower else None
u = [(upper[0][0]+upper[1][0])/2, (upper[0][1]+upper[1][1])/2] if upper else None

print(f"lower_mid: {l}\nupper_mid: {u}")

#step 5 for 2 object videos

#determine specific object interaction based on nose and tail coordinates
 
 
import pandas as pd
import csv
import math

#read the csv
df = pd.read_csv("path to XXX_1_merged.csv")
df = df.where(pd.notna(df), "None")
df = df.sort_values(by='frame')

# pred = df.values.tolist()

# for i in range(len(pred)):
#     pred[i][0] = int(pred[i][0])
# print(pred)

#loop through every frame and conver "object" class to "lower-ob" or "upper-ob"

nose_object = []
ob = ["lower_ob", "upper_ob"]
for i in range(len(pred)):
    if(pred[i][1] == "object"):
        print("enter")
        # dist_u = math.sqrt((u[0]-pred[i][1])**2+(u[1]-pred[i][2])**2)
        # dist_l = math.sqrt((l[0]-pred[i][1])**2+(l[1]-pred[i][2])**2)
        dist_u = math.sqrt((u[0]-float(pred[i][3]))**2+(u[1]-float(pred[i][4]))**2)
        dist_l = math.sqrt((l[0]-pred[i][3])**2+(l[1]-pred[i][4])**2)
        # print("l=",dist_l)
        # print("u=",dist_u)
        dist = [dist_l, dist_u]
        ob = ["lower_ob", "upper_ob"]
        minimum = dist.index(min(dist))
        # print(minimum)
        # print(ob[minimum])
        d = {}
        # d.update({"frame":pred[i][0]})
        # d.update({"nose_x":pred[i][5]})
        # d.update({"nose_y":pred[i][6]})
        # d.update({"tail_x":pred[i][7]})
        # d.update({"tail_y":pred[i][8]})
        # d.update({"pred_nose_tail":ob[minimum]})
        # d.update({"pred_cls": pred[i][2]})
        # d.update({"gt_3cls":pred[i][1]})
        # # d.update({"pred_emily":pred[i][10]})
        # nose_object.append(d)
        d.update({"frame":pred[i][0]})
        d.update({"nose_x":pred[i][3]})
        d.update({"nose_y":pred[i][4]})
        d.update({"tail_x":pred[i][5]})
        d.update({"tail_y":pred[i][6]})
        d.update({"pred_nose_tail":ob[minimum]})
        d.update({"pred_sanaz": pred[i][1]})
        # d.update({"gt_3cls":pred[i][7]})
        # d.update({"gt":pred[i][8]})
        # d.update({"pred_emily":pred[i][10]})
        nose_object.append(d)
    else:
        d = {}
        d.update({"frame":pred[i][0]})
        d.update({"nose_x":pred[i][3]})
        d.update({"nose_y":pred[i][4]})
        d.update({"tail_x":pred[i][5]})
        d.update({"tail_y":pred[i][6]})
        d.update({"pred_nose_tail":pred[i][1]})
        d.update({"pred_sanaz": pred[i][1]})
        # d.update({"gt_3cls":pred[i][7]})
        # d.update({"gt":pred[i][8]})
        # d.update({"gt_3cls":pred[i][1]})
        # d.update({"pred_emily":pred[i][10]})
        nose_object.append(d)
        
            


# fields = ['frame', 'nose_x', 'nose_y', 'tail_x', 'tail_y', 'pred_cls', 'gt_3cls', 'pred_nose_tail']
fields = ['frame', 'nose_x', 'nose_y', 'tail_x', 'tail_y', 'pred_cls_model', 'pred_nose_tail']

filename = "path to XXX_1_post.py"
with open(filename, 'w') as csvfile:
  writer = csv.DictWriter(csvfile, fieldnames=fields)
  writer.writeheader()
  writer.writerows(nose_object)    



#step 4 for 5 object videos

#get the middle point of bounding boxes of each object
#and save them in l (lower object) and u(upper object)

import json

#load the json file
with open('path to annotated frame of XXX.mp4 video (json file)') as f:
    data = json.load(f)

# Create a dict mapping labels to midpoints
objs = {
    s['label']: [
        (s['points'][0][0] + s['points'][1][0]) / 2,
        (s['points'][0][1] + s['points'][1][1]) / 2
    ]
    for s in data['shapes']
}

#get the midpoints into different lists
ob1 = objs.get("ob1", [])
ob2 = objs.get("ob2", [])
ob3 = objs.get("ob3", [])
ob4 = objs.get("ob4", [])
ob5 = objs.get("ob5", [])

print(ob1)
print(ob2)
print(ob3)
print(ob4)
print(ob5)



#step 5 for 5 object videos


#5-obj determine the which object based on nose and tail coordinates with nose_tail filled in for a single file
 
import pandas as pd
import csv
import math

# ob1 = [ob1_x, ob1_y]
# ob2 = [ob2_x, ob2_y]
# ob3 = [ob3_x, ob3_y]
# ob4 = [ob4_x, ob4_y]
# ob5 = [ob5_x, ob5_y]


df = pd.read_csv("path to XXX_1_merged.csv")
df = df.where(pd.notna(df), "None")
df = df.sort_values(by='frame')
pred = df.values.tolist()

# for i in range(len(pred)):
#     pred[i][0] = int(pred[i][0])
# print(pred)

nose_object = []

# od1 = 0
# od2 = 0
# od3 = 0
# od4 = 0
# od5 = 0

for i in range(len(pred)):
    if(pred[i][1] == "object"):
        dist_1 = math.sqrt((ob1[0]-pred[i][3])**2+(ob1[1]-pred[i][4])**2)
        dist_2 = math.sqrt((ob2[0]-pred[i][3])**2+(ob2[1]-pred[i][4])**2)
        dist_3 = math.sqrt((ob3[0]-pred[i][3])**2+(ob3[1]-pred[i][4])**2)
        dist_4 = math.sqrt((ob4[0]-pred[i][3])**2+(ob4[1]-pred[i][4])**2)
        dist_5 = math.sqrt((ob5[0]-pred[i][3])**2+(ob5[1]-pred[i][4])**2)
        # print("l=",dist_l)
        # print("u=",dist_u)
        dist = [dist_1, dist_2, dist_3, dist_4, dist_5]
        ob = ["ob1", "ob2", "ob3", "ob4", "ob5"]
        minimum = dist.index(min(dist))
        # print(minimum)
        # print(ob[minimum])
        d = {}
        # d.update({"frame":pred[i][0]})
        # d.update({"nose_x":pred[i][5]})
        # d.update({"nose_y":pred[i][6]})
        # d.update({"tail_x":pred[i][7]})
        # d.update({"tail_y":pred[i][8]})
        # d.update({"pred_nose_tail":ob[minimum]})
        # d.update({"pred_cls": pred[i][2]})
        # d.update({"gt_3cls":pred[i][1]})
        # # d.update({"pred_emily":pred[i][10]})
        # nose_object.append(d)
        d.update({"frame":pred[i][0]})
        d.update({"nose_x":pred[i][3]})
        d.update({"nose_y":pred[i][4]})
        d.update({"tail_x":pred[i][5]})
        d.update({"tail_y":pred[i][6]})
        d.update({"pred_nose_tail":ob[minimum]})
        d.update({"pred_sanaz": pred[i][1]})
        # d.update({"gt_3cls":pred[i][7]})
        # d.update({"gt":pred[i][8]})
        # d.update({"pred_emily":pred[i][10]})
        nose_object.append(d)
    else:
        d = {}
        d.update({"frame":pred[i][0]})
        d.update({"nose_x":pred[i][3]})
        d.update({"nose_y":pred[i][4]})
        d.update({"tail_x":pred[i][5]})
        d.update({"tail_y":pred[i][6]})
        d.update({"pred_nose_tail":pred[i][1]})
        d.update({"pred_sanaz": pred[i][1]})
        # d.update({"gt_3cls":pred[i][7]})
        # d.update({"gt":pred[i][8]})
        # d.update({"pred_emily":pred[i][10]})
        nose_object.append(d)
        
            


# fields = ['frame', 'nose_x', 'nose_y', 'tail_x', 'tail_y', 'pred_cls', 'gt_3cls', 'pred_nose_tail']
fields = ['frame', 'nose_x', 'nose_y', 'tail_x', 'tail_y', 'pred_cls_model', 'pred_nose_tail']

filename = "/home/a/aliva/venvs/csrnet/rat/final_paper_valid_analysis/individual_m/B6.04_B3_object_interaction.csv"
with open(filename, 'w') as csvfile:
  writer = csv.DictWriter(csvfile, fieldnames=fields)
  writer.writeheader()
  writer.writerows(nose_object)    





