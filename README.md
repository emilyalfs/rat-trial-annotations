# NORT

1) Generate nose/tail keypoints (optionally prior to this, fine tune keypoint model further)
   - keypoint.py generates file XXX_1_keypoints.csv
   - <a href="https://ksuemailprod-my.sharepoint.com/:u:/g/personal/emilyalfs_ksu_edu/EZn-qk8mN6xHsV6Pn6NT-lkBdotLaGfWNc2d_EJPj6-pDg?e=DTNpS6">Pretrained Nose and Tail Model</a>
2) Generate frame by frame behavior
   - behavior.py generates file XXX_2_model.csv
3) Post process general object interaction into specific
   - specify.py generates file XXX_3_post.csv
4) Process nose/tail keypoints for sustained behaviors
   - other.py generates XXX_4_other.csv
5) Aggregate the model behaviors and the sustained behaviors
   - aggregate.py generates XXX_5_aggregate.csv
6) Post process all trials to condense results into meaningful output
