# keypoint prediction file

import cv2 as cv
import os
import sys
from ultralytics import YOLO

video_name = sys.argv[1]
print(f"generating keypoits for: {video_name}")

model = YOLO("./model/best.pt")
cap = cv.VideoCapture(video_name)
short_name = (video_name.split("/")[-1])[:-4]

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

print(f"Done with {video_name} entire")
