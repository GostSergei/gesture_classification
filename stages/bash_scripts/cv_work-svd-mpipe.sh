#!/bin/bash
# python3 ../bullets/model_work_cv_v2.py -s data/skgest-mpipe-center-folds_SVD_max -m models/model6.json -d results_cv -t svd
# python3 ../bullets/model_work_cv_v2.py -s data/skgest-mpipe-center-folds_SVD_max -m models/model1.json -d results_cv -t svd
# python3 ../bullets/model_work_cv_v2.py -s data/skgest-mpipe-center-folds_SVD_max -m models/model2.json -d results_cv -t svd
python3 -u  ../bullets/model_work_cv_v2.py -s data/skgest-mpipe-center-folds_SVD_max -m models/model3.json -d results_cv > svd-mpipe-model3.txt
python3 -u  ../bullets/model_work_cv_v2.py -s data/skgest-mpipe-center-folds_SVD_max -m models/model4.json -d results_cv > svd-mpipe-model4.txt
python3 -u  ../bullets/model_work_cv_v2.py -s data/skgest-mpipe-center-folds_SVD_max -m models/model5.json -d results_cv > svd-mpipe-model5.txt
