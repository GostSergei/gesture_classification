#!/bin/bash
# python3 ../bullets/model_work_cv_v2.py -s data/skgest-smpl-center-folds -m models/model6.json -d results_cv -t tucker
# python3 ../bullets/model_work_cv_v2.py -s data/skgest-smpl-center-folds -m models/model1.json -d results_cv -t tucker
 python3 -u  ../bullets/model_work_cv_v2.py -s data/skgest-smpl-center-folds_tucker_full -m models/model2.json -d results_cv > tucker-smpl-model2.txt 
 python3 -u  ../bullets/model_work_cv_v2.py -s data/skgest-smpl-center-folds_tucker_full -m models/model3.json -d results_cv > tucker-smpl-model3.txt 
 python3 -u  ../bullets/model_work_cv_v2.py -s data/skgest-smpl-center-folds_tucker_full -m models/model4.json -d results_cv > tucker-smpl-model4.txt 
 python3 -u  ../bullets/model_work_cv_v2.py -s data/skgest-smpl-center-folds_tucker_full -m models/model5.json -d results_cv > tucker-smpl-model5.txt  