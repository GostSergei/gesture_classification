#!/bin/bash

nohup ./script_origin.sh &
nohup ./script_tucker.sh &
nohup ./script_svd.sh &



# source bash_scripts/cv_work-svd-mpipe.sh
# source bash_scripts/cv_work-svd-ntu.sh 
# source bash_scripts/cv_work-svd-smpl.sh 

# source bash_scripts/cv_work-tucker-mpipe.sh
# source bash_scripts/cv_work-tucker-ntu.sh 
# source bash_scripts/cv_work-tucker-smpl.sh 


# nohup bash_scripts/cv_work-origin-mpipe.sh &
# nohup bash_scripts/cv_work-origin-ntu.sh &
# nohup bash_scripts/cv_work-origin-smpl.sh &





# python3 ../bullets/model_work_cv.py -s data/skgest-mpipe-center-folds -m models/model6.json -d results_cv -t origin 
# python3 ../bullets/model_work_cv.py -s data/skgest-mpipe-center-folds -m models/model6.json -d results_cv -t tucker 
# python3 ../bullets/model_work_cv.py -s data/skgest-mpipe-center-folds -m models/model6.json -d results_cv -t svd 


# python3 ../bullets/model_work_cv.py -s data/skgest-ntu-center-folds -m models/model6.json -d results_cv -t origin 
# python3 ../bullets/model_work_cv.py -s data/skgest-ntu-center-folds -m models/model6.json -d results_cv -t tucker 
# python3 ../bullets/model_work_cv.py -s data/skgest-ntu-center-folds -m models/model6.json -d results_cv -t svd

# python3 ../bullets/model_work_cv.py -s data/skgest-smpl-center-folds -m models/model6.json -d results_cv -t origin 
# python3 ../bullets/model_work_cv.py -s data/skgest-smpl-center-folds -m models/model6.json -d results_cv -t tucker 
# python3 ../bullets/model_work_cv.py -s data/skgest-smpl-center-folds -m models/model6.json -d results_cv -t svd


# source bash_scripts/cv_work-origin-mpipe.sh
# source bash_scripts/cv_work-tucker-mpipe.sh
# source bash_scripts/cv_work-svd-mpipe.sh

# source bash_scripts/cv_work-origin-ntu.sh
# source bash_scripts/cv_work-tucker-ntu.sh
# source bash_scripts/cv_work-svd-ntu.sh

# source bash_scripts/cv_work-origin-smpl.sh
# source bash_scripts/cv_work-tucker-smpl.sh
# source bash_scripts/cv_work-svd-smpl.sh