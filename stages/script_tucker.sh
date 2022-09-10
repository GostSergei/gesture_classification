#!/bin/bash
# python3 ../bullets/model_work.py -s data/skgest-mpipe-center_tucker_full.pkl -m models/model6.json -d ./
# python3 ../bullets/model_work.py -s data/skgest-mpipe-center_tucker_full.pkl -m models/model2.json -d ./
# python3 ../bullets/model_work.py -s data/skgest-mpipe-center_tucker_full.pkl -m models/model3.json -d ./ 
# python3 ../bullets/model_work.py -s data/skgest-mpipe-center_tucker_full.pkl -m models/model4.json -d ./ 
# python3 ../bullets/model_work.py -s data/skgest-mpipe-center_tucker_full.pkl -m models/model5.json -d ./ 
# python3 ../bullets/model_work.py -s data/skgest-mpipe-center_tucker_full.pkl -m models/model1.json -d ./  

source  ./bash_scripts/cv_work-tucker-mpipe.sh
source  ./bash_scripts/cv_work-tucker-ntu.sh
source  ./bash_scripts/cv_work-tucker-smpl.sh