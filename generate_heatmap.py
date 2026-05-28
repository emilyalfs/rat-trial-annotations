import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sys
import json
from collections import Counter

def custom_plot(plus,minus,bin_size,details):

    

    dict_of_plus_counts = Counter(plus)
    dict_of_minus_counts = Counter(minus)
    
    all_tracker_array = np.zeros((details["img_height"],details["img_width"]))
    plus_tracker_array = np.zeros((details["img_height"],details["img_width"]))
    minus_tracker_array = np.zeros((details["img_height"],details["img_width"]))

    for (x,y) in dict_of_plus_counts.keys():
        for i in range(bin_size):
            for j in range(bin_size):
                plus_tracker_array[int(y+i),int(x+j)] = dict_of_plus_counts[(x,y)]
                all_tracker_array[int(y+i),int(x+j)] = dict_of_plus_counts[(x,y)]
                
    for (x,y) in dict_of_minus_counts.keys():
        for i in range(bin_size):
            for j in range(bin_size):
                minus_tracker_array[int(y+i),int(x+j)] = dict_of_minus_counts[(x,y)]
                all_tracker_array[int(y+i),int(x+j)] -= dict_of_minus_counts[(x,y)]                

    

    fig, ax = plt.subplots()
    im = ax.imshow(all_tracker_array, cmap='PiYG', norm="linear",vmin=-10,vmax=10)
    plt.ylim(details["y_upper_limit"],details["y_lower_limit"])
    plt.xlim(details["x_lower_limit"],details["x_upper_limit"])
    plt.title("Heatmap")
    fig.colorbar(im, ax=ax)
    plt.savefig(f"{details['path']}all.svg", dpi=300)
    
    viridis = mpl.colormaps['viridis'].resampled(256)
    my_viridis = viridis(np.linspace(0.2,1,256)) #this is where you change the range of colors in your color palette. Chnage by adjusting the first two numbers
    cmp_viridis = mpl.colors.ListedColormap(my_viridis)

    fig, ax = plt.subplots()
    im = ax.imshow(plus_tracker_array, cmap=cmp_viridis, norm="log")#,vmin=details["scale_min"],vmax=details["scale_max"]) jet
    plt.ylim(details["y_upper_limit"],details["y_lower_limit"])
    plt.xlim(details["x_lower_limit"],details["x_upper_limit"])
    plt.title("Heatmap")
    fig.colorbar(im, ax=ax) 
    plt.savefig(f"{details['path']}plus.svg", dpi=300)

    fig, ax = plt.subplots()
    im = ax.imshow(minus_tracker_array, cmap=cmp_viridis, norm="log")#,vmin=details["scale_min"],vmax=details["scale_max"])
    plt.ylim(details["y_upper_limit"],details["y_lower_limit"])
    plt.xlim(details["x_lower_limit"],details["x_upper_limit"])
    plt.title("Heatmap")
    fig.colorbar(im, ax=ax)
    plt.savefig(f"{details['path']}minus.svg", dpi=300)

def main():
    path = sys.argv[1] # path to the project (where the keypoints.csv's are)
    #json_instructions = sys.argv[2]

    my_px_size = 3
    standard_fps = 60

    with open(f"{path}heat_layer.csv",'r',encoding='utf-8-sig') as reader:
        heat_data = reader.read()
    heat_data = (heat_data.strip()).split("\n")
    heat_data = [i.split(',') for i in heat_data]
    
    # get the meta data to accomodate for mismatched fps
    try:
        with open(f"{path}video_data.json",'r') as reader:
            video_data = json.load(reader)
    except:
        print("WARNING! video_data.json was not found!\n\tThis is okay if you are CERTAIN that your fps is uniform. ")  
        video_data = {}  
    details = {
        "img_width":1920,
        "img_height":1080,
        "y_lower_limit":0, # for NORT 100
        "y_upper_limit":1080, # for NORT 700
        "x_lower_limit":0, # for NORT 400
        "x_upper_limit":1920, # for NORT 1100
        "scale_min":0,
        "scale_max":1,
        "path": path,
        } 

    plus_noses = []
    minus_noses = []
    
    
    # setting up an 'anchor' for the heatmaps to be adjusted for better overlay
    found_anchor = False
    possible_names = [i[0] for i in heat_data]
    for poss in possible_names:
        poss_path = f"{path}{poss}.json"
        if os.path.exists(poss_path):
            print(f"Using anchor points from {poss_path}")
            found_anchor = True
            with open(poss_path, 'r') as reader:
                temp_pts_data = json.load(reader)
            anchor_shapes = temp_pts_data["shapes"]
            anchor_points = {}
            num_ref_pts = 0
            for elem in anchor_shapes:
                if elem["shape_type"] == "point":
                    anchor_points[elem["label"]] = elem["points"][0]
                    num_ref_pts += 1
            break

    for (name,which) in heat_data:
        try:
            keyps = f"{path}{name}_1_keypoints.csv"
            foo = open(keyps)
        except:
            keyps = f"{path}{name}_1_keypoint.csv"
            foo = open(keyps)    
        
        avg_x_shift, avg_y_shift = 0, 0    
        if found_anchor:
            try:
                with open(f"{path}{name}.json",'r') as reader:
                    ref_json = json.load(reader)
                these_shapes = ref_json["shapes"]

                for pt in these_shapes:
                    if pt["shape_type"] == "point":
                        this_label = pt["label"]
                        chg_x = anchor_points[this_label][0] - pt["points"][0][0] #anchor x - this x
                        chg_y = anchor_points[this_label][1] - pt["points"][0][1] #anchor y - this y
                        avg_x_shift += chg_x/num_ref_pts
                        avg_y_shift += chg_y/num_ref_pts
                        
            except:
                print(f"WARNING: an original anchor file was found but {name} did not have a json file")
                print(f"\tThis is okay if your arena setups are consistent (meaning objects are in same places)")
                       
        data = foo.read()
        foo.close()
        data = data.strip()
        data = data.split("\n")[1:]
        if video_data == {}:
            this_fps = 60
        else:
            this_fps = video_data[name]["fps"]
        flate_rate = standard_fps/this_fps
        
        for row in data:
            if which == "+":
                for i in range(int(flate_rate)):
                    plus_noses.append(
                    (int(((float(row.split(',')[1])+avg_x_shift)//my_px_size)*my_px_size),
                    int(((float(row.split(',')[2])+avg_y_shift)//my_px_size)*my_px_size)))
            elif which == "-":
                for i in range(int(flate_rate)):
                    minus_noses.append(
                    (int(((float(row.split(',')[1])+avg_x_shift)//my_px_size)*my_px_size),
                    int(((float(row.split(',')[2])+avg_y_shift)//my_px_size)*my_px_size)))
                    
    custom_plot(plus_noses,minus_noses,my_px_size,details)

if __name__ == '__main__':
    main() 
