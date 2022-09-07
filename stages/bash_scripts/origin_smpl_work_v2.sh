#!/bin/bash
python3 ../bullets/model_work_2.py -s data/skgest-smpl-center.npz -m models/model6.json -d ./results_2-smpl
python3 ../bullets/model_work_2.py -s data/skgest-smpl-center.npz -m models/model1.json -d ./results_2-smpl
python3 ../bullets/model_work_2.py -s data/skgest-smpl-center.npz -m models/model2.json -d ./results_2-smpl
python3 ../bullets/model_work_2.py -s data/skgest-smpl-center.npz -m models/model3.json -d ./results_2-smpl
python3 ../bullets/model_work_2.py -s data/skgest-smpl-center.npz -m models/model4.json -d ./results_2-smpl
python3 ../bullets/model_work_2.py -s data/skgest-smpl-center.npz -m models/model5.json -d ./results_2-smpl