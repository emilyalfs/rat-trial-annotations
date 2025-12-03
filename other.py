#generates the 'other' behaviors of resting and freezing
import sys
import math

def score_freeze(nose_tails,avg_nt_dist):
    motion_factor = 0.02 # v1= 0.03
    nf_idle = 60         # Num frames to consider at rest (set to 1 second)
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
                activity.append( [ istart, i-1 ] )
                in_idle = False
                istart = i
            #elif in_idle and not moving:  # Still at rest, do nothing
            elif not in_idle and not moving_now:  # Was not at rest, now is at rest
                in_idle = True
            elif not in_idle and moving_now:
                istart = i - nf_idle

    return activity

def score_rest(nose_tails,avg_nt_dist):
    motion_factor = 0.02 # v1= 0.03
    nf_idle = 60         # Num frames to consider at rest (set to 1 second)
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
                activity.append( [ istart, i-1 ] )
                in_idle = False
                istart = i
            #elif in_idle and not moving:  # Still at rest, do nothing
            elif not in_idle and not moving_now:  # Was not at rest, now is at rest
                in_idle = True
            elif not in_idle and moving_now:
                istart = i - nf_idle
    return activity

def main():
    video_path = sys.argv[1]
    proj = sys.argv[2]
    binframes = 1800 # 30sec bins with 60fps so 30*60 = 1800
    setup = "two"

    if setup == "two":
        pixels_per_cm = 13.95
    elif setup == "five":
        pixels_per_cm = 10.5
    else:
        print("ERROR!!! setup not recognized, set pix/cm value")

    short_name = (video_path.split("/")[-1])[:-4]

    nt_file = f"./{proj}/results/{short_name}_1_keypoints.csv"
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

    rest = score_rest(nt_data,avg_nt_dist)
    freeze = score_freeze(nt_data,avg_nt_dist)

    for r in rest:
        start, stop = r[0], r[1]
        for foo in range(start,stop):
            merged_res[foo] = [str(foo),"resting"]

    for f in freeze:
        start, stop = f[0], f[1]
        for foo in range(start,stop):
            merged_res[foo] = [str(foo),"freezing"]

    with open(f"./{proj}/results/{short_name}_4_other.csv", 'w', newline='') as fout:
        fout.write("frame_id,activity")
        for i in merged_res:
            fout.write(",".join(i))
            fout.write("\n")
    print(f"Other model processed {len(merged_res)} frames")

if __name__ == '__main__':
    main()
