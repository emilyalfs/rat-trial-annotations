import os
import cv2
import shutil

paf = "/homes/bhavana/rat-scoring/batch/videos/NO_TMS/"
project_name = "projectX"

files = os.listdir(paf)
files = [i for i in files if ".mp4" in i]

for vid in files:
    print(vid)
    cap = cv2.VideoCapture(paf+vid)
    _, frame = cap.read()
    short = vid[:-4]
    cap.release()
    cv2.imwrite(f"./{project_name}/first_frames/{short}.png",frame)
shutil.make_archive("./{project_name}/first_frames",'zip',f"./{project_name}/first_frames")

