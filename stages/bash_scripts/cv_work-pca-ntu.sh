#!/bin/bash
python3 -u ../bullets/model_work_cv_v2.py -s data/skgest-ntu-center-folds-fill_0-raw-pca_max -m models/model6.json -d results_cv   
python3 -u ../bullets/model_work_cv_v2.py -s data/skgest-ntu-center-folds-fill_0-raw-pca_max -m models/model1.json -d results_cv   
python3 -u ../bullets/model_work_cv_v2.py -s data/skgest-ntu-center-folds-fill_0-raw-pca_max -m models/model2.json -d results_cv   
python3 -u ../bullets/model_work_cv_v2.py -s data/skgest-ntu-center-folds-fill_0-raw-pca_max -m models/model3.json -d results_cv
python3 -u ../bullets/model_work_cv_v2.py -s data/skgest-ntu-center-folds-fill_0-raw-pca_max -m models/model4.json -d results_cv  
python3 -u ../bullets/model_work_cv_v2.py -s data/skgest-ntu-center-folds-fill_0-raw-pca_max -m models/model5.json -d results_cv 