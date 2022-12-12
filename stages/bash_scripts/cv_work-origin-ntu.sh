#!/bin/bash
python3 -u ../bullets/model_work_cv_v2.py -s data/skgest-ntu-center-folds-ffill-c_s-origin -m models/model6.json -d results_cv 
python3 -u ../bullets/model_work_cv_v2.py -s data/skgest-ntu-center-folds-ffill-c_s-origin -m models/model1.json -d results_cv
python3 -u ../bullets/model_work_cv_v2.py -s data/skgest-ntu-center-folds-ffill-c_s-origin -m models/model2.json -d results_cv
python3 -u ../bullets/model_work_cv_v2.py -s data/skgest-ntu-center-folds-ffill-c_s-origin -m models/model3.json -d results_cv
python3 -u ../bullets/model_work_cv_v2.py -s data/skgest-ntu-center-folds-ffill-c_s-origin -m models/model4.json -d results_cv
python3 -u ../bullets/model_work_cv_v2.py -s data/skgest-ntu-center-folds-ffill-c_s-origin  -m models/model5.json -d results_cv

# python3 -u ../bullets/model_work_cv_v2.py -s data/skgest-ntu-center-folds-ffill-c_s-origin_4d  -m models/model5.json -d results_cv