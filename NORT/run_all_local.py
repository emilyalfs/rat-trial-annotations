#/usr/bin/env python3
"""
Streamed inference with temporal aggregation (no intermediate frames on disk).
Outputs:
  - Per model size: raw JSON (all probs) + CSV (top label per frame)
  - Per (method, window, size): aggregated JSON + CSV
"""

import cv2
import sys

import numpy as np
import torch
from ultralytics import YOLO
import os
import json
import math
from datetime import datetime


def load_behv_model(device: str):
    model = YOLO("../models/behavior-model.pt") 
    # move to device if possible (YOLO supports .to)
    if device.startswith('cuda') and torch.cuda.is_available():
        model.to(device)
    return model

def load_keyp_model(device: str):
    model = YOLO("../models/keypoint-model.pt") 
    # move to device if possible (YOLO supports .to)
    if device.startswith('cuda') and torch.cuda.is_available():
        model.to(device)
    return model


# ------------------------- Run Keypoint and Behavior Models -------------------------
@torch.inference_mode()
def stream_inference(video_path: str,  device: str, batch_size: int, proj:str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")

    # Load once
    behv_model = load_behv_model(device)
    keyp_model = load_keyp_model(device)
    

    # pull class names from model metadata via a dummy call OR from model.model.names (cls task)
    # safer: run one dummy on a tiny black frame to acquire names from Results
    dummy = np.zeros((8, 8, 3), dtype=np.uint8)
    res = behv_model([dummy], verbose=False)[0]
    if isinstance(res.names, dict):
        names = [res.names[k] for k in sorted(res.names)]
    else:
        names = list(res.names)

    short_name = (video_path.split("/")[-1])[:-4] # name of video without .mp4

    # setup keypoint writer
    writee = open(f"./{proj}/{short_name}_raw_keypoints.csv",'w')
    writee.write("frame_id,nx,ny,tx,ty\n")

    # setup behavior writer 
    behv_writee = open(f"./{proj}/{short_name}_2_model.csv",'w')
    behv_writee.write('frame_id,label\n')

    batch_frames, batch_idxs = [], []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        batch_frames.append(frame)
        batch_idxs.append(frame_idx)
        frame_idx += 1

        # Process full batches
        if len(batch_frames) == batch_size:
            
            # run behavior model
            results = behv_model(batch_frames, verbose=False)
            for i, res in enumerate(results):
                idx = batch_idxs[i] # frame number
                top = int(res.probs.top1) # best class match
                pred_class = names[top] if 0 <= top < len(names) else str(top) # class name 
                behv_writee.write(f"{idx},{pred_class.lower()}\n") # write to file

            # run keypoint model
            results = keyp_model(batch_frames, verbose=False)
            for i, res in enumerate(results):
                idx = batch_idxs[i] # frame number
                if len(res.keypoints.data)>0: # if the rat was found in frame write to file
                    keyps = list(res.keypoints.data[0])
                    writee.write(f"{idx},{float(keyps[0][0])},{float(keyps[0][1])},{float(keyps[1][0])},{float(keyps[1][1])}\n")
                else: # otherwise, write a row of zeros (zeros will be replaced in future step)
                    writee.write(f"{idx},0,0,0,0\n") 

            # reset batch variables
            batch_frames.clear()
            batch_idxs.clear()

    # Flush leftovers
    if batch_frames:

        # run behavior model
        results = behv_model(batch_frames, verbose=False)
        for i, res in enumerate(results):
            idx = batch_idxs[i]
            top = int(res.probs.top1)
            pred_class = names[top] if 0 <= top < len(names) else str(top)
            behv_writee.write(f"{idx},{pred_class.lower()}\n")
	    
        # run keypoint model 
        results = keyp_model(batch_frames, verbose=False)
        for i, res in enumerate(results):
            idx = batch_idxs[i]
            if len(res.keypoints.data)>0:
                keyps = list(res.keypoints.data[0])
                writee.write(f"{idx},{float(keyps[0][0])},{float(keyps[0][1])},{float(keyps[1][0])},{float(keyps[1][1])}\n")
            else:
                writee.write(f"{idx},0,0,0,0\n")
    writee.close()
    behv_writee.close()
    cap.release()

# --------------- helper to replace missing nose and/or tail ------------
def find_close(ind, orig, need):
    # ind: frame index that was missing an annotaion
    # orig: original keypoint annotaion
    # need: "head" "tail" or "both"

    checker = 1 # start by checking one before and after the index to be replaced
    to_fill = orig[ind].copy() #

    while True:

        if ind-checker < 0: # we hav gone too far backward in our check 
            before = [0,0,0,0,0] 
        else:
            before = orig[ind-checker] # get the keypiont annotaion 'checker' before our missing annotaion

        if ind+checker >= len(orig): # we have gone too far forward in our check
            after = [0,0,0,0,0]
        else:
            after = orig[ind+checker] # get the keypoint annotation 'checker' after our missing annotation

        if need == "head": # if we are looking just for a 'head' annotation 
            if before[1] != 0 and before[2] != 0: # if the past annotation was not 0's, then use it
                to_fill[1] = before[1]
                to_fill[2] = before[2]
            elif after[1] != 0 and after[2] != 0: # if the future annotation was not 0's then use it
                to_fill[1] = after[1]
                to_fill[2] = after[2]
        elif need == "tail": # if we are looking for just a 'tail' annotation (identical to head logic)
            if before[3] != 0 and before[4] != 0: 
                to_fill[3] = before[3]
                to_fill[4] = before[4]
            elif after[3] != 0 and after[4] != 0:
                to_fill[3] = after[3]
                to_fill[4] = after[4]
        elif need == "both": # we need both 'head' and 'tail' annotaions (identical logic and now checking both at once)
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

        # if we found an annotation, then return it
        if to_fill[1] != 0 and to_fill[2] != 0 and to_fill[3] != 0 and to_fill[4] != 0:
            return to_fill
        
        checker += 1 # if not, we must check futher


# ----- fill in missing nose/tails intermediate step 1-----------
def fill_in_missing(video_name:str, proj:str):
    short_name = (video_name.split("/")[-1])[:-4] # the video name with out .mp4

    # load the keypoint annotations
    data_file = f"./{proj}/{short_name}_raw_keypoints.csv"
    f = open(data_file)
    data = f.read()
    f.close()

    data = data.split("\n")[1:-1]
    data = [i.split(",") for i in data]

    data = [[float(i[0]),float(i[1]),float(i[2]),float(i[3]),float(i[4])] for i in data]
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


# ----------- Convert Generic Obj interaction into specific step 3-------------

def specify(video_path, proj, num_objs):

    short_name = (video_path.split("/")[-1])[:-4]
    
    nt_file = f"./{proj}/{short_name}_1_keypoints.csv"
    nt_read = open(nt_file,'r')
    nt_data = nt_read.read()
    nt_read.close()
    
    nt_data = nt_data.strip()
    nt_data = nt_data.split("\n")[1:]
    nt_data = [i.split(",") for i in nt_data]
    nt_data = [[float(i[0]),float(i[1]),float(i[2]),float(i[3]),float(i[4]) ]for i in nt_data]

    beh_file = f"./{proj}/{short_name}_2_model.csv"
    beh_read = open(beh_file,'r')
    beh_data = beh_read.read()
    beh_read.close()

    beh_data = beh_data.strip()
    beh_data = beh_data.split("\n")[1:]
    beh_data = [i.split(",") for i in beh_data]
    beh_data = [[float(i[0]),i[1]] for i in beh_data]

    if num_objs != 0:
        pos_file = open(f"./{proj}/{short_name}.json",'r')
        pos_data = json.load(pos_file)
        pos_file.close()

        obs = pos_data["shapes"]
    
    for i in range(len(beh_data)):
        _, beh = beh_data[i]
        if "object" in beh.lower() and num_objs > 0:
            min_dist, shp = 5000, "none"
            for s in obs:
                x_pt, y_pt = s["points"][0]
                nose_dist = ((x_pt-nt_data[i][1])**2+(y_pt-nt_data[i][2])**2)**0.5
                if min_dist > nose_dist:
                    min_dist = nose_dist
                    shp = s["label"]
            beh_data[i][1] = shp
        elif "object" in beh.lower() and num_objs == 0:
            beh_data[i][1] = "none"

    with open(f"./{proj}/{short_name}_3_post.csv","w") as wrt:
        wrt.write("frame_id,label\n")
        for i in beh_data:
            wrt.write(f"{int(i[0])},{i[1]}\n")
    

# ------ for scoring the 'other' behaviors step 4 ---------

def score_freeze(nose_tails,avg_nt_dist,fps):
    motion_factor = 0.02 # v1= 0.03
    nf_idle = int(fps)         # Num frames to consider at rest (set to 1 second)
    activity = []

    in_idle = False
    istart = 1
    for i in range( 1, len(nose_tails) ):

        if i - istart >= nf_idle:        # Test if prev range greater than nf_idle
            asum = [0.0, 0.0, 0.0, 0.0] # Calculate average positions and variance from
            for j in range( istart, i+1 ):
                asum[0] += nose_tails[j][1]/(i-istart+1)
                asum[1] += nose_tails[j][2]/(i-istart+1)
                asum[2] += nose_tails[j][3]/(i-istart+1)
                asum[3] += nose_tails[j][4]/(i-istart+1)
            nsum = 0.0
            tsum = 0.0
            for j in range( istart, i+1 ): # Calculate average movement around averages
                nx = nose_tails[j][1]
                ny = nose_tails[j][2]
                nsum += math.sqrt( (nx-asum[0])**2+(ny-asum[1])**2 )/(i-istart+1)
                tx = nose_tails[j][3]
                ty = nose_tails[j][4]
                tsum += math.sqrt( (tx-asum[2])**2+(ty-asum[3])**2 )/(i-istart+1)

            moving_now = max(nsum,tsum) > motion_factor*avg_nt_dist

            if in_idle and moving_now:   # Was in idle, now moving, log activity range
                activity.append( [ istart, i-1] )
                in_idle = False
                istart = i
            #elif in_idle and not moving:  # Still at rest, do nothing
            elif not in_idle and not moving_now:  # Was not at rest, now is at rest
                in_idle = True
            elif not in_idle and moving_now:
                istart = i - nf_idle

    return activity

def score_rest(nose_tails,avg_nt_dist,fps):
    motion_factor = 0.02 # v1= 0.03
    nf_idle = int(fps)         # Num frames to consider at rest (set to 1 second)
    activity = []

    in_idle = False
    istart = 1
    for i in range( 1, len(nose_tails) ):

        if i - istart >= nf_idle:        # Test if prev range greater than nf_idle
            asum = [0.0, 0.0, 0.0, 0.0] # Calculate average positions and variance from
            for j in range( istart, i+1 ):
                asum[0] += nose_tails[j][1]/(i-istart+1)
                asum[1] += nose_tails[j][2]/(i-istart+1)
                asum[2] += nose_tails[j][3]/(i-istart+1)
                asum[3] += nose_tails[j][4]/(i-istart+1)
            tsum = 0.0
            for j in range( istart, i+1 ): # Calculate average movement around averages
                tx = nose_tails[j][3]
                ty = nose_tails[j][4]
                tsum += math.sqrt( (tx-asum[2])**2+(ty-asum[3])**2 )/(i-istart+1)

            moving_now = tsum  > motion_factor*avg_nt_dist

            if in_idle and moving_now:   # Was in idle, now moving, log activity range
                activity.append( [ istart, i-1] )
                in_idle = False
                istart = i
            #elif in_idle and not moving:  # Still at rest, do nothing
            elif not in_idle and not moving_now:  # Was not at rest, now is at rest
                in_idle = True
            elif not in_idle and moving_now:
                istart = i - nf_idle
    return activity

def binned_distance_calc(project,video_short,nose_tail_pos,pixels_per_cm,binframes):
    with open(f"./{project}/{video_short}-distances.csv",'w', newline='') as fout:
        ibin = 0
        dsum = 0.0
        nframes = len(nose_tail_pos)
        for i in range(1,nframes):
            px, py = nose_tail_pos[i-1][3], nose_tail_pos[i-1][4]
            cx, cy = nose_tail_pos[i][3], nose_tail_pos[i][4]
            dist = ((cx-px)**2+(cy-py)**2)**0.5
            dsum += dist/pixels_per_cm
            if (i+1) % binframes == 0 or i == nframes-1:
                fout.write(f"{ibin},{dsum}\n")
                dsum = 0.0
                ibin += 1
        
def others(video_path, proj, num_obs, fps, bin_sec_size=30):

    binframes = bin_sec_size * fps


    if num_obs == 2:
        pixels_per_cm = 13.95
    elif num_obs == 0 or num_obs == 5:
        pixels_per_cm = 10.5
    else:
        print("ERROR!!! setup not recognized, set pix/cm value")

    short_name = (video_path.split("/")[-1])[:-4]

    nt_file = f"./{proj}/{short_name}_1_keypoints.csv"
    nt_read = open(nt_file,'r')
    nt_data = nt_read.read()
    nt_read.close()

    nt_data = nt_data.split("\n")[1:-1]
    nt_data = [i.split(",") for i in nt_data]
    nt_data = [[float(i[0]),float(i[1]),float(i[2]),float(i[3]),float(i[4]) ]for i in nt_data]
    nframes = len(nt_data)

    avg_nt_dist = sum([math.sqrt((i[1]-i[3])**2+(i[2]-i[4])**2) for i in nt_data])/nframes

    merged_res = []
    for i in range(nframes):
        merged_res.append([str(i),"none"])

    rest = score_rest(nt_data,avg_nt_dist,fps)
    freeze = score_freeze(nt_data,avg_nt_dist,fps)

    for r in rest:
        start, stop = r[0], r[1]
        for foo in range(start,stop):
            merged_res[foo] = [str(foo),"resting"]

    for f in freeze:
        start, stop = f[0], f[1]
        for foo in range(start,stop):
            merged_res[foo] = [str(foo),"freezing"]

    with open(f"./{proj}/{short_name}_4_other.csv", 'w', newline='') as fout:
        fout.write("frame_id,activity")
        for i in merged_res:
            fout.write(",".join(i))
            fout.write("\n")

    binned_distance_calc(proj,short_name,nt_data,pixels_per_cm,binframes)


# ----- merge 3_post and 4_other ---------

def overwrite_consecutive(data, threshold, replacement="none"):
    """
    frozen and resting require consecutive 1 second

    Args:
        data (list): The input list.
        threshold (int): The minimum number of consecutive occurrences.

    Returns:
        list: A new list with overwritten elements.
    """
    result = data[:]  # Create a copy to avoid modifying the original list
    count = 1
    start_index = 0

    for i in range(1, len(data)):
        if data[i] == data[i - 1] and (data[i]=="frozen" or data[i]=="resting"): # only care about rest/froze
            count += 1
        else:
            if count < threshold and (data[i-1]=="frozen" or data[i-1]=="resting"): # case where rest/froze ended but there weren't threshold or more
                for j in range(start_index, i):
                    result[j] = replacement
            count = 1
            start_index = i

    # Check for the last sequence
    if count < threshold:
        for j in range(start_index, len(data)):
            result[j] = replacement
    
    return result

def aggregate(proj,meta_data):
    
    results_dir = f"./{proj}/"
    files = os.listdir(results_dir)
    files = [ i for i in files if "_3_post.csv" in i]
    # Go through each XXXX_3_post.csv to merge it with the XXXX_4_other.csv into XXXX_5_aggregated.py
    for fil in files:
        short = fil[:-11] # gets the prefix as len(_3_post.csv) is 11
        fps = meta_data[short]["fps"]
        # load XXXX_3_post
        objet = open(results_dir + fil)
        object_data = objet.read()
        objet.close()

        # will be frame, ...., pred
        object_data = object_data.split("\n")[1:]
        object_data = [i.split(",") for i in object_data]
        object_data = [i for i in object_data if len(i)>1]


        # load XXXX_4_other
        freeze = open(results_dir + short + "_4_other.csv")
        freeze_data = freeze.read()
        freeze.close()

        # will be frame, pred
        freeze_data = freeze_data.split("\n")[1:]
        freeze_data = [i.split(",") for i in freeze_data]
        freeze_data = [i for i in freeze_data if len(i)>1]


        merge_data = []

        num_frames = min(len(object_data),len(freeze_data)) # in case there is a discrepency in # of frames
        for i in range(num_frames):
            fr_frame, fr_pred = freeze_data[i][0], freeze_data[i][-1]
            obj_frame, obj_pred = object_data[i][0], object_data[i][-1]

            if obj_pred.lower() == "none": # if _3_post has none, then default to _4_other
                merge_data.append(fr_pred.lower())
            else:
                merge_data.append(obj_pred.lower())


        # since rest/froze requires 1 sec sustained, make sure it is preserved
        merg_result = overwrite_consecutive(merge_data,fps,"none")

        # will be frame, pred
        with open(results_dir + short + "_5_aggregated.csv","w") as wrti:
            wrti.write("frame num, label\n")
            for i in range(len(merge_data)):
                wrti.write(f"{i}, {merg_result[i]}\n")


# ---- condense all video files together -----

def get_proj_classes(proj):
    other_path = f"./{proj}/"
    files = os.listdir(other_path)
    files = [i for i in files if "aggregated.csv" in i]
    unique_classes = set()
    for fil in files: 
        with open(f"{other_path}{fil}",'r') as reder:
            for line in reder:
                line = line.strip().split(",")
                clas = line[-1].strip().lower()
                unique_classes.add(clas)
    unique_classes.remove('label')
    return list(unique_classes)

def condense_dists(proj):
    file_path = f"./{proj}/"
    files = os.listdir(file_path)
    files = [i for i in files if "distances.csv" in i]

    all_results = []
    for f in files:
        short = f.split("-")[0]
        pre, post = short.split("_")
        reader = open(file_path + f)
        data = reader.read()
        reader.close()

        data = data.split("\n")[1:-1]
        data = [i.split(",") for i in data]
        summer = 0.0
        for res in data:
            summer += float(res[1])
            all_results.append([short,res[0],res[1]])
        all_results.append([short,"total",summer])

    tot_writer = open(file_path + "final-condensed-tail-dist-cm-per-30-sec.csv",'w')
    tot_writer.write(f"Individual,Day,Bin (30 seconds),Distance (in CM)\n")

    for d in all_results:
        pre, post = d[0].split("_")
        tot_writer.write(f"{pre},{post},{d[1]},{d[2]}\n")
    tot_writer.close()

def condense_bevs(proj,meta_data):
    other_path = f"./{proj}/"
    files = os.listdir(other_path)
    files = [i for i in files if "aggregated.csv" in i]
    labels = get_proj_classes(proj)

    data_dict = {}
    all_results = {}
    for f in files:
        short = f[:-17]
        data_dict[short] = {}
        for p in labels:
            data_dict[short][p] = []

        reader = open(other_path + f)
        data = reader.read()
        reader.close()

        data = data.strip().split("\n")[1:]
        data = [i.split(",") for i in data]

        for i in data:
            idx,name =  i[0],i[1]
            name = name.strip()
            name = name.lower()
            if name in data_dict[short]:
                data_dict[short][name].append(int(idx))
        fps = meta_data[short]["fps"]
        all_results[short] = {}
        for p in data_dict[short]:
            all_results[short][p] = len(data_dict[short][p])/fps

    write_comb = open(f"{other_path}final-condensed-behaviors-in-secs.csv",'w')

    write_comb.write(f"Individual,Day")
    for i in labels:
        write_comb.write(f",{i}")
    write_comb.write("\n")

    for d in data_dict:
        pre, post = d.split("_")
        r = all_results[d]
        write_comb.write(f"{pre},{post}")
        for i in labels:
            write_comb.write(f",{r[i]}")
        write_comb.write("\n")

    write_comb.close()



# ------------------------- CLI -------------------------

def main():

    # input will be python run_all_local.py path/to/videos/ 2
    paf = sys.argv[1]
    num_objs = int(sys.argv[2])

    proj = paf.split("/")[-2]
    # device fallback
    dev = "cuda"
    if dev.startswith('cuda') and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU.")
        dev = 'cpu'
    abs_start = datetime.now()
    with open(f"./{paf}video_data.json",'r') as reader:
        meta_data = json.load(reader)
    files = os.listdir(paf)
    files = [i for i in files if "mp4" in i]
    for v_name in files: 
        start = datetime.now()
        print(f"working on {v_name}, start time: {start.strftime('%H:%M:%S')}")
        video = paf+ v_name
        short_name = ((video).split("/")[-1])[:-4]
        fps = meta_data[short_name]["fps"]
        if not os.path.exists(paf + short_name + "_raw_keypoints.csv") or not os.path.exists(paf + short_name + "_2_model.csv"):
            stream_inference(video, dev, 1, proj)
        print(f"\tBehvior Model Done")
        if not os.path.exists(paf + short_name + "_1_keypoints.csv"):
            fill_in_missing(video,proj)
        print(f"\tKeypoint Model Done")
        if not os.path.exists(paf+short_name+"_3_post.csv"):
            specify(video,proj,num_objs)
        print(f"\tObject Specification Done")
        if not os.path.exists(paf+short_name+"_4_other.csv"):
            others(video,proj,num_objs,fps)
        print(f"\tOther Behviors Done")
        end = datetime.now()
        change = end - start
        print(f"Video {v_name} completed and took {str(change)}")
    print("All videos done individually, now condensing")
    # at this point, all videos in the dir have files 1-4.
    # now condense those files to the nice output
    aggregate(proj,meta_data) #generate file 5
    condense_bevs(proj,meta_data)
    condense_dists(proj)
    end = datetime.now()
    change = end - abs_start
    print(f"Processing complete for {proj}. Total time: {str(change)}")
    
if __name__ == '__main__':
    main()# generate behaviors (standing, object interaction, none)
