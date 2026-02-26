import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sys
import json


# Given a set of numbers, 3,4,5,6,7,12,13,14,15, will return the ranges [(3,7),(12,15)]
def consolidate_to_ranges(numbers):
    # check that there were numbers passed
    if not numbers:
        return []

    # check that they are sorted and start with the smallest number
    numbers.sort()
    ranges = []
    start = numbers[0]
    end = numbers[0]

    # loop through the numbers check that the next number is one more than the current
    for i in range(1, len(numbers)):
        if numbers[i] == end + 1: # still an increase by one
            end = numbers[i] 
        else: # not an increase by one
            if start == end: # if there was a solitary number, don't count it
                pass
            else: # append the range 
                ranges.append((start,end))
            # start again
            start = numbers[i]
            end = numbers[i]

    # make sure we capture the last number of the list of numbers
    if start == end: # an instance of a solitary number
        pass
    else: # append the range
        ranges.append((start,end))
    return ranges


# generate the plot 
# results - dictionary of {v_name1: {behav1:[(r1,r2),(r3,r4)],behav2:[(r5,r6)],...},v_name2:{...}}
# info_map - dictionary of behviors with their colors and pretty names
# format_map - dictionary of plot settings
# nframes - number of frames to display on the plot
# time_seconds - number of seconds to display on the plot
def survey(results, info_map, format_map, nframes,time_seconds,plot_start,fps,minute_tick):
    fig = plt.figure( figsize=( format_map["img_width"], format_map["img_height"] )  )
    plt.clf()    # Clear past stuff (must go after plt.figure() )
    ax = plt.gca()   # Get current axis  (must go after plt.clf() )
    ax.set_xlim( plot_start*fps, nframes ) # axis for frame numbers
    ax.tick_params( axis='x')
    ax.set_xlabel("Frame Number", fontweight='bold',fontsize=format_map["legend_font_size"])


    ylabs = format_map["video_order"]
    ax.tick_params( axis='y', which='both', direction='out' )
    ax.set_ylim(min(format_map["y_ticks"])-format_map["y_offset"],max(format_map["y_ticks"])+format_map["y_offset"])
    plt.yticks(format_map["y_ticks"] , ylabs,fontweight='bold', fontsize=format_map["video_font_size"],color='k' ) 


    ax2 = ax.twiny()
    ax2.set_xlabel( 'Time in Minutes',fontweight='bold',color='k',fontsize=format_map["legend_font_size"]) # axis for time in minutes
    ax2.tick_params( axis='x', colors='k' )
    ax2.set_xticks(np.arange( plot_start/ 60.0,time_seconds // 60.0 + 1 + minute_tick , minute_tick))
    ax2.set_xlim( plot_start/ 60.0, time_seconds / 60.0 ) # 60 seconds in one minute

    # in charge of filling in the ethogram bars
    ct = 0
    for i in format_map["video_order"]:
        for p in results[i]:
            y_placement = format_map["y_ticks"][ct]
            ax.plot( [0,nframes], [y_placement,y_placement], color='black', linewidth=1 )
            for r in results[i][p]:
                ax.plot( [r[0],r[1]], [y_placement,y_placement], color=info_map[p]["color"], linewidth=4 )
        ct += 1

    # in charge of the legend making 
    ct = 0
    for i in info_map:
        plt.figtext(0.15 + 0.15*ct, 0.004, info_map[i]["pretty_name"], color=info_map[i]["color"], fontweight='bold', fontsize=format_map["legend_font_size"] )
        ct += 1
    plt.savefig(format_map["img_save_name"])


def main():
    path = sys.argv[1] # path to the project (where the aggregated.csv's are)
    json_file = sys.argv[2] # file that has the setup information 
    with open(json_file,'r') as reader:
        js_data = json.load(reader)
    kind = js_data["file_type"]
    
    # get all the aggregated files 
    files = os.listdir(path)
    files = [i for i in files if kind in i]
    files.sort()
    
    activities = js_data["activities"] # will be the activities from aggregated.csv
    colors = js_data["colors"] # colors for the bars
    pretty_names = js_data["pretty_names"] # nice names for the legend
    my_order = js_data["file_order"] # if you wanted the files in a certain order
    n_seconds = int(js_data["plot_stop_second"]) # number of seconds to display on the chart
    lower_bound = int(js_data["lower_bound_seconds"]) # cut off point if video was shorter than all the rest
    plot_start = int(js_data["plot_start_second"]) # start plot at this seconds timestamp
    minute_tick = float(js_data["minute_increment"]) # increment to break down minutes 1 = 1 min, 0.5 is half min

    # if the order was not input, then default to sorted 
    if len(my_order) == 0:
        short_files = ["_".join(i.split("_")[:2]) for i in files]
        short_files.sort()
        my_order = short_files

    format_map = js_data["format_map"]
    format_map["video_order"] = my_order # if there is a fixed order, get it now

    format_map["y_offset"] = 0.1 # controls vertical spacing of the bars

    # sets up the behavior, color, and pretty name dictionary
    label_map = {}
    for i in range(len(activities)):
        label_map[activities[i]] = {"color":colors[i], "pretty_name":pretty_names[i]}
        
    # get the meta data to accomodate for mismatched fps
    with open(f"{path}video_data.json",'r') as reader:
        meta_data = json.load(reader)
    
    # we anticipate all videos to be 60fps, so we will normalize to this
    normalized_fps = 60

    data_dict = {}
    for f in files:
        short = "_".join(f.split("_")[:2])
        if short in my_order or len(my_order)==0:
            fps = int(meta_data[short]["fps"]) # check this videos fps
            frame_count = int(meta_data[short]["frame_count"])

            # start a videos dictionary
            data_dict[short] = {}
            full = path + f
            foo = open(full)
            data = foo.read()
            foo.close()
            data = data.strip()
            data = data.split("\n")

            # if we need to cut some frames off of the end 
            # (IE want 300 sec trials but there is a 302 sec, will cut off 2 sec)
            if len(data) > n_seconds*fps:
                data = data[1:n_seconds*fps]
            else:
                data = data[1:-1]

            if frame_count//fps >= lower_bound: # make sure we have the minimum desired length
                # this is the guard for mismatched fps
                if normalized_fps >= fps:
                    flate_rate = int(normalized_fps / fps) # how much we need to inflate the data by
                    flated_data = []
                    count = 0

                    # for each label in the original data, enter it flate_rate times
                    # with 30 fps -> 60 fps becomes 
                    # [[0,a],[1,b],[2,c],[3,d],[4,e]] -> [[0,a],[1,a],[2,b],[3,b],[4,c],[5,c]...]
                    for i in data:
                        _, label = int(i.split(",")[0]), i.split(",")[1]
                        for _ in range(flate_rate):
                            flated_data.append([count,label])
                            count += 1
                    data = flated_data
                
                # go through the data to convert it to:
                # {"behv1":[x1,x2,x3,x4,....],"behv2":[x5,x6,x7,x8,...]...}
                for i in data:
                    idx, name = i[0], i[1]
                    name = name.strip()
                    # to allow for grouping of behaviors 
                    for acty in activities:
                        if name in acty:
                            name = acty
                            break
                            
                    
                    if name in activities: # allows for ommision of activities 
                        if name in data_dict[short]:
                            data_dict[short][name].append(int(idx))
                        else:
                            data_dict[short][name] = [int(idx)]
                
                # consolidate the lists to ranges
                for k in data_dict[short]:
                    cons = consolidate_to_ranges(data_dict[short][k])
                    data_dict[short][k] = cons
            else:
                my_order.remove(short) # need to remove the bad ones so it doesn't try to plot them later
                print(f"ERR! {short} less than {lower_bound}! It was {len(data)//fps} seconds")

    my_y_placements = [x/10 for x in range(len(my_order))]     # makes the trials evenly split
    format_map["y_ticks"] = my_y_placements
    survey(data_dict,label_map,format_map,n_seconds*normalized_fps,n_seconds,plot_start,normalized_fps,minute_tick)

if __name__ == '__main__':
    main() 
