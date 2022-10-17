#!/bin/bash

# clone data and move the traces to a local datasets directory
git clone https://github.com/ML-paging-project/Robust-Learning-Augmented-Caching-An-Experimental-Study-Datasets.git temp_dir
mv temp_dir/datasets datasets

# cleanup
rm -r temp_dir
