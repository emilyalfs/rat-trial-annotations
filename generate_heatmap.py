import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sys
import json
from collections import Counter

def custom_plot(data_list,bin_size,details):

    fig, ax = plt.subplots()

    dict_of_counts = Counter(data_list)
    tracker_array = np.zeros((details["img_height"],details["img_width"]))

    for (x,y) in dict_of_counts.keys():
        for i in range(bin_size):
            for j in range(bin_size):
                tracker_array[int(y+i),int(x+j)] = dict_of_counts[(x,y)]

    im = ax.imshow(tracker_array, cmap='jet', norm="log")

    plt.ylim(details["y_upper_limit"],details["y_lower_limit"])
    plt.xlim(details["x_lower_limit"],details["x_upper_limit"])
    plt.title("Heatmap")
    plt.show()

def main():
    path = sys.argv[1] # path to the project (where the keypoint.csv's are)

    # get all the keypoint files 
    files = os.listdir(path)
    files = [i for i in files if "_1_keypoint.csv" in i]
    files.sort()

    my_order = []#js_data["selection"]
    my_px_size = 5#js_data["box_size"]
    
    # if the order was not input, then default to sorted 
    if len(my_order) == 0:
        short_files = ["_".join(i.split("_")[:2]) for i in files]
        short_files.sort()
        my_order = short_files

    details = {
        "img_width":1920,
        "img_height":1080,
        "y_lower_limit":350,
        "y_upper_limit":1080,
        "x_lower_limit":400,
        "x_upper_limit":1400}
    
    noses = []
    for f in files:
        short = "_".join(f.split("_")[:2])
        if short in my_order:
            full = path + f
            foo = open(full)
            data = foo.read()
            foo.close()
            data = data.strip()
            data = data.split("\n")[1:]
            for row in data:
                noses.append((int((float(row.split(',')[1])//my_px_size)*my_px_size),int((float(row.split(',')[2])//my_px_size)*my_px_size)))
    custom_plot(noses,my_px_size,details)

if __name__ == '__main__':
    main() 
