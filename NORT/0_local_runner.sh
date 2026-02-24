#!/bin/bash -l
source ./environment/bin/activate

python extract_first_frames.py $1
python -m labelme
