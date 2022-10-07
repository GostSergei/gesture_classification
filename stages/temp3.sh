# Decomposition
# nohup transformation_scripts/transform_pca-mpipe.sh  >  pca-mpipe.txt  2>&1 &
# nohup transformation_scripts/transform_pca-ntu.sh    >  pca-ntu.txt    2>&1 &
# nohup transformation_scripts/transform_pca-smpl.sh   >  pca-smpl.txt   2>&1 &

nohup transformation_scripts/transform_tucker-mpipe.sh  >  tucker_tn-mpipe.txt  2>&1 &
nohup transformation_scripts/transform_tucker-ntu.sh   >  tucker_tn-ntu.txt    2>&1 &
nohup transformation_scripts/transform_tucker-smpl.sh  >  tucker_tn-smpl.txt   2>&1 &


# CV work
# nohup bash_scripts/cv_work-pca-mpipe.sh  >  cv_work-pca-mpipe.txt  2>&1 &
# nohup bash_scripts/cv_work-pca-ntu.sh   >  cv_work-pca-ntu.txt    2>&1 &
# nohup bash_scripts/cv_work-pca-smpl.sh   >  cv_work-pca-smpl.txt   2>&1 &

