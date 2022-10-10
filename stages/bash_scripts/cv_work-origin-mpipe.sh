#!/bin/bash
python3 -u ../bullets/model_work_cv_v2.py -s data/skgest-mpipe-center-folds-ffill-raw-origin -m models/model6.json -d results_cv > origin-mpipe-model6.txt
python3 -u ../bullets/model_work_cv_v2.py -s data/skgest-mpipe-center-folds-ffill-raw-origin -m models/model1.json -d results_cv > origin-mpipe-model1.txt
python3 -u ../bullets/model_work_cv_v2.py -s data/skgest-mpipe-center-folds-ffill-raw-origin -m models/model2.json -d results_cv > origin-mpipe-model2.txt
python3 -u ../bullets/model_work_cv_v2.py -s data/skgest-mpipe-center-folds-ffill-raw-origin -m models/model3.json -d results_cv > origin-mpipe-model3.txt
python3 -u ../bullets/model_work_cv_v2.py -s data/skgest-mpipe-center-folds-ffill-raw-origin -m models/model4.json -d results_cv > origin-mpipe-model4.txt
python3 -u ../bullets/model_work_cv_v2.py -s data/skgest-mpipe-center-folds-ffill-raw-origin -m models/model5.json -d results_cv > origin-mpipe-model5.txt