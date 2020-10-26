#!/bin/bash

# turn on python environment
adl

# Disable if the original bounding boxes have been extracted already
# python experiment_4_1_bounding_boxes.py

# Extract the heatmaps for each image
python experiemnt_4_1_compute_heatmaps.py

# Extract the bounding boxes from the heatmaps
python experiment_4_1.py

# Compute Intersection of union
python score.py