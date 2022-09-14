# Scripts and notebook(module) for testing tensor-based methods for gesture classification. 

See requroment.txt to get all libs

This module consisted of:
- python scripts to work with data ... gesture_classification/bullets/*
The main bullets are model_work_2.py (the first experiment) and model_work_cv_v2.py (the second experiment) as well as svd_decompose.py and tucker_decompose.py 

- ML model parameters description gesture_classification/stages/models/*

- notes: gesture_classification/stages/notes/*

- data sorece folder: gesture_classification/stages/data/*

- Different results folders: gesture_classification/stages/results...

- auxiliary bash scripts to launch the pythons sripts: gesture_classification/stages/bash_scripts/*

- other bash scripts to more easily launch python scripts via bash scripts
, etc.



### Python scripts to convert original data with Tucher decomposition and SVD (initial data is in .npz,  transformed - .pkl):
svd_decompose.py: SVD-based data transformation 
-s or --srs - a source of the data, file path
-d or --dst - output file_name (extension should be .pkl as pickle lib was used to save the output data), file name 
-r or --rank - decomposition rank

tucker_decompose.py: Tucker-based data transformation 
-s or --srs - a source of the data, file path
-d or --dst - output file_name (extension should be .pkl as pickle lib was used to save the output data), file name 
-r or --rank - decomposition rank




### Python scripts to test ML models
model_work_2.py: Conduction the first experiments
parameters: 
-s or --srs - a source of the data, file path.
-m or --src_m - a source of the ML model description, file path. gesture_classification/stages/models/*
-d or --dst - output dir of the model (dir path), output name will be formed automatically: .json - accuracy data, .pkl - dictionary of {   y_test: ... , preds: ...}, .jpg - confusion matrix


model_work_cv_v2.py:  Conduction cross-validation experiment
parameters: 
-s or --srs - a source of the data, dir path.
-m or --src_m - a source of the ML model description, file path. gesture_classification/stages/models/*
-d or --dst - output dir of the model (dir path), output name will be formed automatically: .json - accuracy data, .pkl - python dictionary of {   y_test: ... , preds: ...}

