#!/bin/bash
python3 ../bullets/model_work_cv.py -s data/skgest-smpl-center-folds -m models/model6.json -d results_cv -t origin
python3 ../bullets/model_work_cv.py -s data/skgest-smpl-center-folds -m models/model1.json -d results_cv -t origin
python3 ../bullets/model_work_cv.py -s data/skgest-smpl-center-folds -m models/model2.json -d results_cv -t origin
python3 ../bullets/model_work_cv.py -s data/skgest-smpl-center-folds -m models/model3.json -d results_cv -t origin
python3 ../bullets/model_work_cv.py -s data/skgest-smpl-center-folds -m models/model4.json -d results_cv -t origin
python3 ../bullets/model_work_cv.py -s data/skgest-smpl-center-folds -m models/model5.json -d results_cv -t origin
