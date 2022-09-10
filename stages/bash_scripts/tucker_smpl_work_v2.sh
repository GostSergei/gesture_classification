#!/bin/bash
# python3 ../bullets/model_work_2.py -s data/skgest-smpl-center_tucker_full.pkl -m models/model6.json -d ./results_2-smpl
# python3 ../bullets/model_work_2.py -s data/skgest-smpl-center_tucker_full.pkl -m models/model1.json -d ./results_2-smpl
# python3 ../bullets/model_work_2.py -s data/skgest-smpl-center_tucker_full.pkl -m models/model2.json -d ./results_2-smpl
# python3 ../bullets/model_work_2.py -s data/skgest-smpl-center_tucker_full.pkl -m models/model3.json -d ./results_2-smpl
python3 -u ../bullets/model_work_2.py -s data/skgest-smpl-center_tucker_full.pkl -m models/model4.json -d ./results_2-smpl   > work_2-tucker-smpl.txt &
python3 -u ../bullets/model_work_2.py -s data/skgest-smpl-center_tucker_full.pkl -m models/model5.json -d ./results_2-smpl   > work_2-tucker-smpl.txt &