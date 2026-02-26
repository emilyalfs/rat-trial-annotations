1) Load videos into a new directory/folder; we use .mp4 format. Name the folder something unique (IE the date). For the purposes of this instructions, assume we have videos in ~/Desktop/NORT/EXAMPLE/

2) From the terminal
	- cd ~/Desktop/NORT
	- bash 0_local_runner.sh EXAMPLE/
		- You should see output: Extracted first frames from XXX videos
	
3) This will launch the "LabelMe" interface. You will need to label the objects in the videos. You only need to mark the centers!
*** if no objects are needed, you can just close LabelMe
	- Select "Open Dir"
	- Choose the directory of the current project
		- You should now see a still image from one of the videos
	- On the left bar, select Create Point
	- Your mouse pointer will now be a cross-hair. Use that to select the center of an object. 
	- A dialog box will open to let you name the object. We have used the convention of upper_ob and lower_ob. 
	- Select the next object and give it a label. Repeat for however many objects. 
	- In the top toolbar, select Next Image, and it will prompt you to save your annotation. You will need to hit "Save" two times.
	- Continue for all videos. Once you are done, you can close the LabelMe interface. MAKE SURE THE LAST ONE SAVES! You may need to select "Save" rather than "Next Image".
	- Close the LabelMe interface
	
4) Now we are set to run the annotation scripts! From the terminal
	- bash 1_local_runner.sh EXAMPLE/ 2
	- the '2' at the end is the number of objects for the trial (enter 0, 2, or 5)
 	- this generates files in the project directory for each video:
  		- xxx_raw_keypoint.csv: raw keypoints, including 0,0's for missing keypoints (frame_id, nose x, nose y, tail x, tail y)
    	- xxx_1_keypoint.csv: post processed keypoints, filled in 0,0's (frame_id, nose x, nose y, tail x, tail y)
     	- xxx_2_model.csv: raw output from behavior model, values: none, object interaction, standing (frame_id,label)
      	- xxx_3_post.csv: post processed behaviors, specifies which objects are being interacted with (frame_id,label)
      	- xxx_4_other.csv: auxillary behaviors, resting and freezing (frame_id,activity)
      	- xxx_5_aggregated.csv: all behaviors combined (frame_id,label)

5) Optional: For the purposes of outputting ethograms as 'Novel' and 'Familiar' you can run 'labels_to_nov_familiar.py'. This will generate xxx_6_novandfami.csv in the project directory for each video. 
