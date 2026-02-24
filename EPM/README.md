1) Load videos into a new directory/folder; we use .mp4 format. Name the folder something unique (IE the date). For the purposes of this instructions, assume we have videos in ~/Desktop/EPM/EXAMPLE/

2) From the terminal
	- cd ~/Desktop/EPM
	- bash 0_local_runner.sh EXAMPLE/
		- You should see output: Extracted first frames from XXX videos
	
3) This will launch the "LabelMe" interface. You will need to label the intersection square in the videos. If other rats are present in the video, make a 'border'. Our videos have miscellaneous rats in some of the videos on the right hand side. 
	- Select "Open Dir"
	- Choose the directory of the current project
		- You should now see a still image from one of the videos
	- On the left bar, select Create Rectangle.
	- Your mouse pointer will now be a cross-hair. To make the square, click where the top left corner is and then click where the bottom right corner is.  
	- A dialog box will open to let you name the object. Name this "center".
	- If another rat is in the video, follow this part as well! 
		- On the left bar, select Create Line.
		- Click a point at the top of the frame that exclude the extra rats
		- Then click a point at the bottom of the frame that is roughly in a straight line from your starting point.
		- Name this line "border".
	- In the top toolbar, select Next Image, and it will prompt you to save your annotation. You will need to hit "Save" two times.
	- Continue for all videos. Once you are done, you can close the LabelMe interface. MAKE SURE THE LAST ONE SAVES! You may need to select "Save" rather than "Next Image".
	- Close the LabelMe interface
	
4) Now we are set to run the annotation scripts! From the terminal
	- bash 1_local_runner.sh EXAMPLE/ 


