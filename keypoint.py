import cv2 as cv
import os
import sys
from ultralytics import YOLO
import torch


def load_model(device: str):
    model = YOLO("./models/keypoint-model.pt")
    # move to device if possible (YOLO supports .to)
    if device.startswith('cuda') and torch.cuda.is_available():
        model.to(device)
    return model


@torch.inference_mode()
def stream_inference(video_path: str,  device: str, proj:str):
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")

    # Load once
    model = load_model(device)

    short_name = (video_path.split("/")[-1])[:-4]

    out_file = f"./{proj}/results/{short_name}_1_keypoints.csv"
    if os.path.exists(out_file):
        os.remove(out_file)

    f_id = 0
    writee = open(out_file,'w')
    writee.write("frame_id, nx, ny, tx, ty\n")
    ret, frame = cap.read()
    while(ret):
        results = model(frame,verbose=False)
        result = results[0]
        if len(result.keypoints.data)>0:
            keyps = list(result.keypoints.data[0])
            for k in range(len(keyps)):
                keyps[k] = list(keyps[k])
                for j in range(len(keyps[k])):
                    keyps[k][j] = float(keyps[k][j])
            writee.write(f"{f_id},{keyps[0][0]},{keyps[0][1]},{keyps[1][0]},{keyps[1][1]}\n")
        else:
            writee.write(f"{f_id},0,0,0,0\n")
        f_id += 1
        ret, frame = cap.read()

    cap.release()
    cv.destroyAllWindows()
    writee.close()


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


def fill_in_missing(video_name:str, proj:str):
    short_name = (video_name.split("/")[-1])[:-4]
    data_file = f"./{proj}/results/{short_name}_1_keypoints.csv"
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

    with open(data_file,"w") as writ:
        writ.write("frame_id,nx,ny,tx,ty\n")
        for i in updata:
            writ.write(f"{int(i[0])},{i[1]},{i[2]},{i[3]},{i[4]}\n")
    print(f"Keypoint model processed {len(data)} frames")
def main():

    video_name = sys.argv[1]
    dev = sys.argv[2]
    proj = sys.argv[3]

    if dev.startswith('cuda') and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU.")
        dev = 'cpu'

    stream_inference(video_name, dev, proj)
    fill_in_missing(video_name, proj)
if __name__ == '__main__':
    main()
