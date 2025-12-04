import os
import cv2
import shutil
import sys

paf = sys.argv[1]
project_name = sys.argv[2]

files = os.listdir(paf)
files = [i for i in files if ".mp4" in i]

for vid in files:
    cap = cv2.VideoCapture(paf+vid)
    _, frame = cap.read()
    short = vid[:-4]
    cap.release()
    cv2.imwrite(f"./{project_name}/first_frames/{short}.png",frame)
shutil.make_archive("./{project_name}/first_frames",'zip',f"./{project_name}/first_frames")
print(f"Extracted first frames from {len(files)} videos") 
