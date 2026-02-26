import os
import cv2
import sys
import json

paf = sys.argv[1] # full path to the videos directory

files = os.listdir(paf)
files = [i for i in files if ".mp4" in i] # get just the mp4s (can include other video types)

video_data = {} # json data created to capture video details (fps, width, height, frame count)

for vid in files:
    short = vid[:-4] # removes .mp4 

    cap = cv2.VideoCapture(paf+vid)
    _, frame = cap.read() # open the video just get first frame

    # save video data
    video_data[short] = {"fps":cap.get(cv2.CAP_PROP_FPS), 
                         "width": cap.get(cv2.CAP_PROP_FRAME_WIDTH), 
                         "height":cap.get(cv2.CAP_PROP_FRAME_HEIGHT), 
                         "frame_count":cap.get(cv2.CAP_PROP_FRAME_COUNT)}
    cap.release() # done with the video
    cv2.imwrite(f"{paf}{short}.png",frame) # save the first frame for annotaion of objects

# save the video data
with open(f"{paf}video_data.json",'w') as writee:
    json.dump(video_data,writee,indent=4)

print(f"Extracted first frames from {len(files)} videos") 