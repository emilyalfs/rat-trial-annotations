# combine the specific object interactions with the sustained behaviors
import sys
import os 

def get_proj_classes(proj):
    other_path = f"./{proj}/results/"
    files = os.listdir(other_path)
    files = [i for i in files if "aggregated.csv" in i]
    unique_classes = set()
    for fil in files: 
        with open(f"{other_path}{fil}",'r') as reder:
            for line in reder:
                line = line.strip().split(",")
                clas = line[-1].strip().lower()
                unique_classes.add(clas)

    return list(unique_classes)

def condense_dists(proj):
    file_path = f"./{proj}/results/"
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

    tot_writer = open(file_path + "final-condensed-butt-dist-cm-per-30-sec.csv",'w')
    tot_writer.write(f"Individual,Day,Bin (30 seconds),Distance (in CM)\n")

    for d in all_results:
        pre, post = d[0].split("_")
        tot_writer.write(f"{pre},{post},{d[1]},{d[2]}\n")
    tot_writer.close()


def condense_bevs(proj):
    other_path = f"./{proj}/results/"
    files = os.listdir(other_path)
    files = [i for i in files if "aggregated.csv" in i]
    labels = get_proj_classes(proj)

    data_dict = {}
    all_results = {}
    for f in files:
        short = f[:-17]

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

        all_results[short] = {}
        for p in data_dict[short]:
            all_results[short][p] = len(data_dict[short][p])/60

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
    condense_bevs(proj)
    condense_dists(proj)

if __name__ == '__main__':
    main()
