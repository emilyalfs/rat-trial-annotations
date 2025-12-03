# combine the specific object interactions with the sustained behaviors
import sys
import os 

def overwrite_consecutive(data, threshold=60, replacement="none"):
    """
    frozen and resting require consecutive 60 frames

    Args:
        data (list): The input list.
        threshold (int, optional): The minimum number of consecutive occurrences. Defaults to 60
        replacement (any, optional): The value to replace elements with. Defaults to none.

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
            if count < threshold and (data[i-1]=="frozen" or data[i-1]=="resting"): # case where rest/froze ended but there weren't 60 or more
                for j in range(start_index, i):
                    result[j] = replacement
            count = 1
            start_index = i

    # Check for the last sequence
    if count < threshold:
        for j in range(start_index, len(data)):
            result[j] = replacement
    
    return result


def main():
    
    proj = sys.argv[1]

    results_dir = f"./{proj}/results/"
    files = os.listdir(results_dir)
    files = [ i for i in files if "_3_post.csv" in i]
    # Go through each XXXX_3_post.csv to merge it with the XXXX_4_other.csv into XXXX_5_aggregated.py
    for fil in files:
        short = fil[:-11] # gets the prefix as len(_3_post.csv) is 11
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


        # since rest/froze requires 60 cons. frames, make sure it is preserved
        merg_result = overwrite_consecutive(merge_data,60,"none")

        # will be frame, pred
        with open(results_dir + short + "_5_aggregated.csv","w") as wrti:
            wrti.write("frame num, label\n")
            for i in range(len(merge_data)):
                wrti.write(f"{i}, {merg_result[i]}\n")


    



if __name__ == '__main__':
    main()
