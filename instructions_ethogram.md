# 


# Properties 
- 'activities': the list of activities from your files that you want to display. You can omit activities if you don't want them to be displayed. The names here must match the case and spelling from the data files. 
- 'colors': the colors you want to represent each activity. This is a one-to-one relationship so the length of 'colors' must be the same as the length of 'activities'. The color in the first spot will be the color of the first activity, and so on. 
- 'pretty_names': the names of the activities as you would like them to appear on the graphic. Again, this is a one-to-one mapping with 'activities'. 
- 'file_order': the list of files you want to appear on the graphic. Leave it as [] if you want all of the files in the directory in alphabetical order. The first file in the list will be at the bottom of the chart and the last file in the list will be at the top of the chart. Enter them as "B6.14_pre3" 
- 'file_type': either "aggregated.csv" or "novandfami.csv" at this time. Aggregated is the default output of both the EPM and NORT trials. Novandfami is the post-processed novel object videos to have "Novel" and "Familiar" labels rather than the "upper_ob" and "lower_ob".
- 'lower_bound_seconds': the lower bound on length to include a video in the graphic. IE if you have 5 minute (300 sec) videos, you may want to set this value to '245' to eliminate trials that ended more than 15 seconds early.  Set this to '0' if you want to include videos regardless of length. 
- 'plot_start_second': the time where you want to start the graphic. 
- 'plot_stop_second': the time where you want to stop the graphic. 
- 'minute_increment': how you want the minute ticks to show up. IE '2' means every 2 minutes, '0.25' means every quarter of a minute. Can be beneficial when looking at smaller time ranges. 
- 'format_map': handles the formating of the graphic
    - 'img_width': graphic width in inches
    - 'img_height': graphic height in inches
    - 'video_font_size': size of the video name labels marking the y axis
    - 'legend_font_size': size of the font for the legend
    - 'img_save_name': name of the file you want to save the graphic. Will save in the same lcation as the generate_etho.py file