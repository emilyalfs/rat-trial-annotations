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
def stream_inference(video_path: str,  device: str):
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")

    # Load once
    model = load_model(device)

    short_name = (video_path.split("/")[-1])[:-4]

    out_file = f"./results/{short_name}_entire.csv"
    if os.path.exists(out_file):
        os.remove(out_file)

    f_id = 0
    writee = open(out_file,'w')
    writee.write("frame num, nx, ny, tx, ty\n")
    ret, frame = cap.read()
    while(ret):
        results = model(frame,verbose=False)
        result = results[0]
        keyps = list(result.keypoints.data[0])
        for k in range(len(keyps)):
            keyps[k] = list(keyps[k])
            for j in range(len(keyps[k])):
                keyps[k][j] = float(keyps[k][j])
        writee.write(f"{f_id},{keyps[0][0]},{keyps[0][1]},{keyps[1][0]},{keyps[1][1]}\n")
        f_id += 1
        ret, frame = cap.read()

    cap.release()
    cv.destroyAllWindows()
    writee.close()



def main():

    video_name = sys.argv[1]
    dev = sys.argv[2]

    if dev.startswith('cuda') and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU.")
        dev = 'cpu'

    stream_inference(video_name, dev)

if __name__ == '__main__':
    main()
