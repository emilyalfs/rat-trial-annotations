import sys
import json

def main():
    video_path = sys.argv[1]
    proj = sys.argv[2]
    short_name = (video_path.split("/")[-1])[:-4]
    
    nt_file = f"./{proj}/results/{short_name}_1_keypoints.csv"
    nt_read = open(nt_file,'r')
    nt_data = nt_read.read()
    nt_read.close()
    
    nt_data = nt_data.strip()
    nt_data = nt_data.split("\n")[1:]
    nt_data = [i.split(",") for i in nt_data]
    nt_data = [[float(i[0]),float(i[1]),float(i[2]),float(i[3]),float(i[4]) ]for i in nt_data]
    nt_frames = len(nt_data)  

    beh_file = f"./{proj}/results/{short_name}_2_model.csv"
    beh_read = open(beh_file,'r')
    beh_data = beh_read.read()
    beh_read.close()

    beh_data = beh_data.strip()
    beh_data = beh_data.split("\n")[1:]
    beh_data = [i.split(",") for i in beh_data]
    beh_data = [[float(i[0]),i[1]] for i in beh_data]
    beh_frames = len(nt_data)

    pos_file = open(f"./{proj}/first_frames/{short_name}.json",'r')
    pos_data = json.load(pos_file)
    pos_file.close()

    obs = pos_data["shapes"]
    
    for i in range(len(beh_data)):
        _, beh = beh_data[i]
        if "object" in beh.lower():
            min_dist, shp = 5000, "none"
            for s in obs:
                x_pt, y_pt = s["points"][0]
                nose_dist = ((x_pt-nt_data[i][1])**2+(y_pt-nt_data[i][2])**2)**0.5
                if min_dist > nose_dist:
                    min_dist = nose_dist
                    shp = s["label"]
            beh_data[i][1] = shp

    with open(f"./{proj}/results/{short_name}_3_post.csv","w") as wrt:
        wrt.write("frame_id,label\n")
        for i in beh_data:
            wrt.write(f"{int(i[0])},{i[1]}\n")
    
    print(f"Specify processed {len(beh_data)} frames")

if __name__ == '__main__':
    main()
