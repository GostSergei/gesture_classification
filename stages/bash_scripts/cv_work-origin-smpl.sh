#!/bin/bash
# python3 ../bullets/model_work_cv_v2.py -s data/skgest-smpl-center-folds -m models/model6.json -d results_cv -t origin
#  python3 ../bullets/model_work_cv_v2.py -s data/skgest-smpl-center-folds -m models/model1.json -d results_cv 2>&1 > origin-smpl-model1.txt &
#  python3 ../bullets/model_work_cv_v2.py -s data/skgest-smpl-center-folds -m models/model2.json -d results_cv -t origin
#  python3 ../bullets/model_work_cv_v2.py -s data/skgest-smpl-center-folds -m models/model3.json -d results_cv -t origin
#  python3 ../bullets/model_work_cv_v2.py -s data/skgest-smpl-center-folds -m models/model4.json -d results_cv -t origin
#  python3 ../bullets/model_work_cv_v2.py -s data/skgest-smpl-center-folds -m models/model5.json -d results_cv -t origin



python3 -u  ../bullets/model_work_cv_v2.py -s data/skgest-smpl-center-folds -m models/model1.json -d results_cv > origin-smpl-model1.txt
python3 -u  ../bullets/model_work_cv_v2.py -s data/skgest-smpl-center-folds -m models/model2.json -d results_cv > origin-smpl-model2.txt
python3 -u  ../bullets/model_work_cv_v2.py -s data/skgest-smpl-center-folds -m models/model3.json -d results_cv > origin-smpl-model3.txt
python3 -u  ../bullets/model_work_cv_v2.py -s data/skgest-smpl-center-folds -m models/model4.json -d results_cv > origin-smpl-model4.txt
python3 -u  ../bullets/model_work_cv_v2.py -s data/skgest-smpl-center-folds -m models/model5.json -d results_cv > origin-smpl-model5.txt
