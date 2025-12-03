# NORT

1) Generate nose/tail keypoints (optionally prior to this, fine tune keypoint model further)
   - keypoint.py generates file XXX_1_keypoints.csv
2) Generate frame by frame behavior
   - behavior.py generates file XXX_2_model.csv
3) Post process general object interaction into specific
   - specify.py generates file XXX_3_post.csv
4) Process nose/tail keypoints for sustained behaviors
   - other.py generates XXX_4_other.csv
5) Aggregate the model behaviors and the sustained behaviors
   - aggregate.py generates XXX_5_aggregate.csv
6) Post process all trials to condense results into meaningful output

Download pretrained models <a href="https://ksuemailprod-my.sharepoint.com/:u:/g/personal/emilyalfs_ksu_edu/ESFgbrjK3S5ItHuYDFNulWQBR7myM2ouWYNqXovdDCcLfg?e=fCGTFT"> here</a>

## Project Structure

<ul>
   <li>raw-videos
      <ul>
         <li>video1.mp4</li>
         <li>video2.mp4</li>
      </ul>
   </li>
   <li>models
      <ul>
         <li>nose-tail-keypoint.pt</li>
         <li>behavior.pt</li>
      </ul>
   </li>
   <li>aggregate.py</li>
   <li>behavior.py</li>
   <li>keypoint.py</li>
   <li>other.py</li>
   <li>specify.py</li>
   <li> project-directory 
      <ul>
      <li>first-frames
         <ul>
            <li>video1-0.json</li>
            <li>video1-0.png</li>
            <li>video2-0.json</li>
            <li>video2-0.png</li>
         </ul>
      </li>
      <li>results
         <ul>
            <li>video1_1_keypoints.csv</li>
            <li>video1_2_model.csv</li>
            <li>video1_3_post.csv</li>
            <li>video1_4_other.csv</li>
            <li>video1_5_aggregate.csv</li>
         </ul>
      </li>
      <li>out_files
         <ul>
            <li>video1_1.out</li>
            <li>video1_2.out</li>
            <li>video1_3.out</li>
            <li>video1_4.out</li>
            <li>video1_5.out</li>
         </ul>
      </li>
      </ul>
   </li>
</ul>

