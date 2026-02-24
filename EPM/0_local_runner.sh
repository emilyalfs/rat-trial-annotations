#!/bin/bash -l
source ~/envs/nort/bin/activate

python extract_first_frames.py $1
python -m labelme
