# Scripts and notebookv(module) for testing tensor-based methods for gesture classification. 

See requroment.txt to get all libs

This module consisted of:
- python scripts to work with data ... gesture_classification/bullets/*
The main bulltest are model_work_2.py (the first experiment) and model_work_cv.py (the second experiment) 

model_work_2.py:
input: 
srs - sourse of the data, filepath
src_m - source of the ML model descriptin in model, filepaty. gesture_classification/stages/models/*
dst - output dir of the model (dirpath), output name will be formed automatically: .json - accuracy data, .pkl - doct of {   y_test: ... , preds: ...}, .jpg - confusion matrix