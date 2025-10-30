# NORT

1) Generate nose/tail keypoints (optionally prior to this, fine tune keypoint model further)
   - generates file XXX_1_keypoints.csv
2) Generate frame by frame behavior
   - generates file XXX_2_model.csv
3) Post process general object interaction into specific
   - generates file XXX_3_post.csv
4) Process nose/tail keypoints for sustained behaviors
   - generates XXX_4_other.csv
5) Aggregate the model behaviors and the sustained behaviors
   - generates XXX_5_aggregate.csv
6) Post process all trials to condense results into meaningful output
