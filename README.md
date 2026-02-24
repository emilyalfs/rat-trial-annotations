# Installation 

Clone this repo to your computer. We recommend making a dedicated directory for this repo. From the terminal or command line: 
- mkdir rat-trial-annotations
- git clone git_repo_url rat-trial-annotations

We also recommend setting up a dedicated virtual environment for the project. We have strayed away from using Anaconda due to licensing changes. From the terminal or command line
- cd rat-trial-annotations
- python -m venv environment
- source environment/bin/activate
- pip -r install requirements.txt


Download pretrained models <a href="https://ksuemailprod-my.sharepoint.com/:u:/g/personal/emilyalfs_ksu_edu/ESFgbrjK3S5ItHuYDFNulWQBR7myM2ouWYNqXovdDCcLfg?e=fCGTFT"> here</a> in a folder called 'models' (see project files structure below). 

You are now set to run NORT or EPM annotations. Check the corresponding directory for further instructions. 

## Project Structure

<ul>
      <li>models
      <ul>
         <li>nose-tail-keypoint.pt</li>
         <li>behavior.pt</li>
      </ul>
   </li>
   <li>EPM
      <ul>
         <li> 0_local_runner.sh </li>
         <li> 1_local_runner.sh </li>
         <li> extract_first_frames.py</li>
         <li> run_all_local.py</li>
         <li>EXAMPLE
            <ul>
               <li>video_1.mp4</li>
               <li>video_2.mp4</li>
            </ul>
         </li>
      </ul>
   </li>
   <li>NORT
      <ul>
         <li> 0_local_runner.sh </li>
         <li> 1_local_runner.sh </li>
         <li> extract_first_frames.py</li>
         <li> run_all_local.py</li>
         <li> labels_to_nov_familiar.py </li>
         <li>EXAMPLE
            <ul>
               <li>video_1.mp4</li>
               <li>video_2.mp4</li>
            </ul>
         </li>
      </ul>
   </li>
</ul>

