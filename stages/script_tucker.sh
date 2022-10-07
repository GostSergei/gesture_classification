#!/bin/bash
# python3 ../bullets/model_work.py -s data/skgest-mpipe-center_tucker_full.pkl -m models/model6.json -d ./
# python3 ../bullets/model_work.py -s data/skgest-mpipe-center_tucker_full.pkl -m models/model2.json -d ./
# python3 ../bullets/model_work.py -s data/skgest-mpipe-center_tucker_full.pkl -m models/model3.json -d ./ 
# python3 ../bullets/model_work.py -s data/skgest-mpipe-center_tucker_full.pkl -m models/model4.json -d ./ 
# python3 ../bullets/model_work.py -s data/skgest-mpipe-center_tucker_full.pkl -m models/model5.json -d ./ 
# python3 ../bullets/model_work.py -s data/skgest-mpipe-center_tucker_full.pkl -m models/model1.json -d ./  

# source  ./bash_scripts/cv_work-tucker-mpipe.sh
# source  ./bash_scripts/cv_work-tucker-ntu.sh
# source  ./bash_scripts/cv_work-tucker-smpl.sh


nohup ./bash_scripts/cv_work-tucker-mpipe.sh > cv_work-tucker_tn-mpipe.txt  2>&1 &
nohup ./bash_scripts/cv_work-tucker-ntu.sh > cv_work-tucker_tn-ntu.txt  2>&1 &
nohup ./bash_scripts/cv_work-tucker-smpl.sh > cv_work-tucker_tn-smpl.txt  2>&1 &
