#!/bin/bash
python3 -u ../bullets/model_work_cv_v2.py -s data/skgest-ntu-center-folds-ffill-raw-origin -m models/model6.json -d results_cv > origin-ntu-model6.txt
python3 -u ../bullets/model_work_cv_v2.py -s data/skgest-ntu-center-folds-ffill-raw-origin -m models/model1.json -d results_cv > origin-ntu-model1.txt
python3 -u ../bullets/model_work_cv_v2.py -s data/skgest-ntu-center-folds-ffill-raw-origin -m models/model2.json -d results_cv > origin-ntu-model2.txt
python3 -u ../bullets/model_work_cv_v2.py -s data/skgest-ntu-center-folds-ffill-raw-origin -m models/model3.json -d results_cv > origin-ntu-model3.txt
python3 -u ../bullets/model_work_cv_v2.py -s data/skgest-ntu-center-folds-ffill-raw-origin -m models/model4.json -d results_cv > origin-ntu-model4.txt
python3 -u ../bullets/model_work_cv_v2.py -s data/skgest-ntu-center-folds-ffill-raw-origin -m models/model5.json -d results_cv > origin-ntu-model5.txt