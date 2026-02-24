import os
import cv2
import shutil
import sys
import json

paf = sys.argv[1]
project_name = paf.split("/")[-2]

files = os.listdir(paf)
files = [i for i in files if ".mp4" in i]

video_data = {}

for vid in files:
    cap = cv2.VideoCapture(paf+vid)
    _, frame = cap.read()
    short = vid[:-4]
    video_data[short] = {"fps":cap.get(cv2.CAP_PROP_FPS), "width": cap.get(cv2.CAP_PROP_FRAME_WIDTH), "height":cap.get(cv2.CAP_PROP_FRAME_HEIGHT), "frame_count":cap.get(cv2.CAP_PROP_FRAME_COUNT)}
    cap.release()
    cv2.imwrite(f"./{project_name}/{short}.png",frame)
with open(f"./{project_name}/video_data.json",'w') as writee:
    json.dump(video_data,writee,indent=4)
print(f"Extracted first frames from {len(files)} videos") 
