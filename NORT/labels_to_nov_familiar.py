import sys

def update(paf,data_row):
    short_name = data_row[0]
    lower_goes_to = data_row[1]
    upper_goes_to = data_row[2]
    agg_file = f"{paf}{short_name}_5_aggregated.csv"
    nov_file = f"{paf}{short_name}_6_novandfami.csv"
    with open(agg_file,'r') as reader:
        data = reader.read()
    data = data.split("\n")[1:-1]
    data = [i.split(",") for i in data]
    
    updated_data = []
    for r in range(len(data)):
        this_label = data[r][1]
        if "lower" in this_label:
            this_label = lower_goes_to
        elif "upper" in this_label:
            this_label = upper_goes_to
        updated_data.append([data[r][0],this_label])
    
    with open(nov_file,'w') as writer:
        writer.write("frame_id,label\n")
        for row in updated_data:
            writer.write(f"{row[0]},{row[1]}\n")
            
    
    
def main():
    paf = sys.argv[1]
    with open(f"{paf}familiar-novel-map.csv",'r') as reader:
        data = reader.read()
        data = data.strip()
    data = data.split("\n")[1:]
    data = [["_".join(i.split(",")[:2]),i.split(",")[2],i.split(",")[3]]for i in data] # this will give name_time, lower, upper
    for row in data:
        update(paf,row)
    
    
     
    
if __name__ == '__main__':
    main()

