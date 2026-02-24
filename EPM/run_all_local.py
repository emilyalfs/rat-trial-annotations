#/usr/bin/env python3
"""
Streamed inference with temporal aggregation (no intermediate frames on disk).
Outputs:
  - Per model size: raw JSON (all probs) + CSV (top label per frame)
  - Per (method, window, size): aggregated JSON + CSV
"""

import cv2
import csv
import sys

import numpy as np
import torch
from ultralytics import YOLO
import os
import json
import math
from datetime import datetime


from shapely.geometry import Point, Polygon
import shapely


def load_keyp_model(device: str):
    model = YOLO("/home/plakkelab/Desktop/models/keypoint-model.pt")
    # move to device if possible (YOLO supports .to)
    if device.startswith('cuda') and torch.cuda.is_available():
        model.to(device)
    return model

# ------------------------- Run Keypoint Model -------------------------

@torch.inference_mode()
def stream_inference(video_path: str,  device: str, proj:str,x_cutoff:int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")

    # Load once
    keyp_model = load_keyp_model(device)

    short_name = (video_path.split("/")[-1])[:-4]

    out_file = f"./{proj}/{short_name}_raw_keypoints.csv"
    writee = open(out_file,'w')
    writee.write("frame_id,nx,ny,tx,ty\n")

    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = frame[0:1020,0:x_cutoff]
        frame_idx += 1
        keyp_results = keyp_model(frame,verbose=False)
        keyp_result = keyp_results[0]
        if len(keyp_result.keypoints.data)>0:
            keyps = list(keyp_result.keypoints.data[0])
            writee.write(f"{frame_idx},{float(keyps[0][0])},{float(keyps[0][1])},{float(keyps[1][0])},{float(keyps[1][1])}\n")
        else:
            writee.write(f"{frame_idx},0,0,0,0\n")



    writee.close()
    cap.release()

# --------------- helper to replace missing nose and/or tail ------------

def find_close(ind, orig, need):
    checker = 1
    to_fill = orig[ind].copy()

    while True:
        if ind-checker < 0:
            before = [0,0,0,0,0]
        else:
            before = orig[ind-checker]
        if ind+checker >= len(orig):
            after = [0,0,0,0,0]
        else:
            after = orig[ind+checker]
        if need == "head":
            if before[1] != 0 and before[2] != 0:
                to_fill[1] = before[1]
                to_fill[2] = before[2]
            elif after[1] != 0 and after[2] != 0:
                to_fill[1] = after[1]
                to_fill[2] = after[2]
        elif need == "tail":
            if before[3] != 0 and before[4] != 0:
                to_fill[3] = before[3]
                to_fill[4] = before[4]
            elif after[3] != 0 and after[4] != 0:
                to_fill[3] = after[3]
                to_fill[4] = after[4]
        elif need == "both":
            if to_fill[1] == 0 and to_fill[2] == 0:
                if before[1] != 0 and before[2] != 0:
                    to_fill[1] = before[1]
                    to_fill[2] = before[2]
                elif after[1] != 0 and after[2] != 0:
                    to_fill[1] = after[1]
                    to_fill[2] = after[2]

            if to_fill[3] == 0 and to_fill[4] == 0:
                if before[3] != 0 and before[4] != 0:
                    to_fill[3] = before[3]
                    to_fill[4] = before[4]
                elif after[3] != 0 and after[4] != 0:
                    to_fill[3] = after[3]
                    to_fill[4] = after[4]


        if to_fill[1] != 0 and to_fill[2] != 0 and to_fill[3] != 0 and to_fill[4] != 0:
            return to_fill
        checker += 1

# ----- fill in missing nose/tails intermediate step 1-----------

def fill_in_missing(video_name:str, proj:str):
    short_name = (video_name.split("/")[-1])[:-4]
    data_file = f"./{proj}/{short_name}_raw_keypoints.csv"
    f = open(data_file)
    data = f.read()
    f.close()

    data = data.split("\n")[1:-1]
    data = [i.split(",") for i in data]

    data = [[float(i[0]),float(i[1]),float(i[2]),float(i[3]),float(i[4]) ]for i in data]
    updata = []
    count = 0
    for i in range(len(data)):
        head = True
        tail = True
        both = True
        if data[i][1] == 0 or data[i][2] == 0:
            head = False
        if data[i][3] == 0 or data[i][4] == 0:
            tail = False

        if not head and not tail:
            both = False
        
        if not both:
            closest = find_close(i, data, "both")
            updata.append(closest)
            count += 1
        elif not head:
            closest = find_close(i, data, "head")
            updata.append(closest)
            count += 1
        elif not tail:
            closest = find_close(i, data, "tail")
            updata.append(closest)
            count += 1
        else:
            updata.append(data[i])

    new_count = 0
    for i in updata:
        if i[1] == 0 or i[2] == 0 or i[3] == 0 or i[4] == 0:
            new_count += 1
    data_file = f"./{proj}/{short_name}_1_keypoints.csv"
    with open(data_file,"w") as writ:
        writ.write("frame_id,nx,ny,tx,ty\n")
        for i in updata:
            writ.write(f"{int(i[0])},{i[1]},{i[2]},{i[3]},{i[4]}\n")

def get_intersections(x1,y1,x2,y2): 
    #Top left
    #try the y=0
    tl =  x1-y1*(x1-x2)/(y1-y2)
    if tl < 0:
        tl = y1 - x1*(y1-y2)/(x1-x2)
        tlpt = (0,int(tl))
    else:
        tlpt = (int(tl),0)

    #Bottom right
    #try the y=1080
    br = x1 + (1080-y1)*(x1-x2)/(y1-y2)
    if br > 1920:
        br = y1 - (1920-x1)*(y1-y2)/(x1-x2)
        brpt = (1920,int(br))
    else:
        brpt = (int(br),1080)
        
    #Top right
    #try y=0
    tr = x1+(0-y2)*(x2-x1)/(y1-y2)
    if tr > 1920:
        tr = y1 - (1920-x1)*(y2-y1)/(x1-x2)
        trpt = (1920,int(tr))
    else:
        trpt = (int(tr),0)
    
    #bottom left
    #try y=1080
    bl = x1+(1080-y2)*(x2-x1)/(y1-y2)
    if bl < 0:
        bl = y1 - (x1)*(y2-y1)/(x1-x2)
        blpt = (0,int(bl))
    else:
        blpt = (int(bl),1080)

    return tlpt, brpt, trpt, blpt

def my_within(point,region):
    result = shapely.within(point,region)
    intersect = region.intersects(point)
    if result or intersect:
        return 1
    else:
        return 0

def condense_results(paf,meta_data,bout_length=15):
    files = os.listdir(paf)
    files = [i for i in files if "_5_aggregated.csv" in i]

    comprehensive = paf+"comprehensive.csv"
    with open(comprehensive,'w') as writ:
        writ.write("video name, N (in secs), E (in secs), S (in secs), W (in secs), O (in secs)\n")

    bouts_file = paf+"bout-report.csv"
    with open(bouts_file,'w') as writ:
        writ.write("video name, N bouts, E bouts, S bouts, W bouts, O bouts\n") 

    regions = ['N','E','S','W',"O","X"]
    video_data = {}
    bout_count = {}


    for file in files:
        for i in regions:
            video_data[i] = 0
            bout_count[i] = 0
        reder = open(paf+file,'r')
        data = reder.read()
        reder.close()

        data = data.strip()
        data = data.split("\n")
        data = [i.split(",") for i in data]

        bout = 0
        sequen = ""
        for i in data:
            curr = i[1]
            video_data[curr] += 1
            if curr == sequen:
                bout += 1
            else:
                if bout >= bout_length:
                    bout_count[curr] += 1
                bout = 0
                sequen = curr

        short = file[:-17]
        fps = meta_data[short]["fps"]
        with open(comprehensive,'a') as writ:
            writ.write(f"{short},{int(video_data['N']/fps)},{int(video_data['E']/fps)},{int(video_data['S']/fps)},{int(video_data['W']/fps)},{int(video_data['O']/fps)}\n")

        with open(bouts_file,'a') as writ:
            writ.write(f"{short},{bout_count['N']},{bout_count['E']},{bout_count['S']},{bout_count['W']},{bout_count['O']}\n")

    

# ------------------------- CLI -------------------------

def main():
    # input will be python run_all_local.py path/to/videos/ 
    paf = sys.argv[1]
    proj = paf.split("/")[-2]
    # device fallback
    dev = "cuda"
    if dev.startswith('cuda') and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU.")
        dev = 'cpu'
    abs_start = datetime.now()
    regions = ['N','E','S','W',"O","X"]

    files = os.listdir(paf)
    files = [i for i in files if ".png" in i] # get all our png files

    with open(f"{paf}/video_data.json",'r') as reader:
        meta_data = json.load(reader)
    
    for fil in files:
        short_name = fil[:-4]
        start = datetime.now()
        x_cutoff = 1920
        print(f"working on {short_name}, start time: {start.strftime('%H:%M:%S')}")
        print(f"\tLoading the JSON file")
        with open(paf+short_name+".json",'r') as reader:
            data = json.load(reader)
        if len(data["shapes"]) > 1: 
            for i in data["shapes"]:
                if i["label"] == "border":
                    x_cutoff = int(float(i["points"][0][0])) # check this implementation 
            print(f"\tWill be cropping this video to {x_cutoff}x1080")
        
        print("\tStarting keypoint model")
        video = paf+ short_name + ".mp4"
        if not os.path.exists(paf + short_name + "_raw_keypoints.csv"):
            stream_inference(video, dev, proj,x_cutoff)
        if not os.path.exists(paf + short_name + "_1_keypoints.csv"):
            fill_in_missing(video,proj)
        print(f"\tKeypoint Model Done")
    	

        for i in data["shapes"]:
            if i["label"] == "center":
                xa, ya = i["points"][0]
                xb, yb = i["points"][1]
                # just in case annotators didn't do top left then bottom right
                x1, y1 = min(xa,xb), min(ya,yb)
                x2, y2 = max(xa,xb), max(ya,yb)
        
        (tlx,tly),(brx,bry),(trx,tri),(blx,bly) = get_intersections(x1,y1,x2,y2)
        x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)

	    # set up our quadrants
        north_sect = Polygon([(x1,y1),(x2,y1),(trx,tri),(tlx,tly)])
        east_sect = Polygon([(x2,y1),(x2,y2),(brx,bry),(x_cutoff,1080),(x_cutoff,0),(trx,tri)])
        south_sect = Polygon([(x1,y2),(x2,y2),(brx,bry),(blx,bly)])
        west_sect = Polygon([(x1,y1),(x1,y2),(blx,bly),(0,1080),(0,0),(tlx,tly)]) 
        center_sect = Polygon([(x1,y1),(x2,y1),(x2,y2),(x1,y2)])

        nt_file = open(paf + short_name + "_1_keypoints.csv",'r')
        data = nt_file.read()
        nt_file.close()
        data = data.strip()
        data = data.split("\n")
        data = [i.split(",") for i in data[1:]]

        reg_file = open(paf+short_name+"_5_aggregated.csv",'w')
        for row in range(len(data)):
            _,nx,ny,tx,ty = data[row]
            nose_point = Point(nx,ny)
            tail_point = Point(tx,ty)

            nose_res = [my_within(nose_point,north_sect),my_within(nose_point,east_sect),my_within(nose_point,south_sect),my_within(nose_point,west_sect),my_within(nose_point,center_sect)]
            tail_res = [my_within(tail_point,north_sect),my_within(tail_point,east_sect),my_within(tail_point,south_sect),my_within(tail_point,west_sect),my_within(tail_point,center_sect)]

            full = [nose_res[i] or tail_res[i] for i in range(len(nose_res))]
            sumr = sum(full)

            if sumr == 0:
                region = "X"
            elif sumr > 1:
                region = "O"
            else:
                idxr = full.index(1)
                region = regions[idxr]

            if region == "X":
                print(f"{short_name} at frame {row}")

            reg_file.write(f"{row},{region}\n")
        reg_file.close()
        end = datetime.now()
        change = end - start
        print(f"Video {short_name} completed and took {str(change)}")

    print("All keypoints and regions calculated!")
    print("Now condensing results...")
    # Now put all the results together
    condense_results(paf,meta_data,bout_length=15)
    end = datetime.now()
    change = end - abs_start
    print(f"Processing complete for {proj}. Total time: {str(change)}")


if __name__ == '__main__':
    main() # generate region data
