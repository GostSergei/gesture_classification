# Decomposition
# nohup transformation_scripts/transform_pca-mpipe.sh  >  pca-mpipe.txt  2>&1 &
# nohup transformation_scripts/transform_pca-ntu.sh    >  pca-ntu.txt    2>&1 &
# nohup transformation_scripts/transform_pca-smpl.sh   >  pca-smpl.txt   2>&1 &


# nohup transformation_scripts/transform_reshape_4d-mpipe.sh  >  transform_reshape_4d-mpipe.txt  2>&1 &
# nohup transformation_scripts/transform_reshape_4d-ntu.sh   >  transform_reshape_4d-ntu.txt    2>&1 &
# nohup transformation_scripts/transform_reshape_4d-smpl.sh  >  transform_reshape_4d-smpl.txt   2>&1 &


nohup transformation_scripts/transform_tucker-mpipe.sh  >  tucker_tn_4D-mpipe.txt  2>&1 &
nohup transformation_scripts/transform_tucker-ntu.sh    >  tucker_tn_4D-ntu.txt    2>&1 &
nohup transformation_scripts/transform_tucker-smpl.sh   >  tucker_tn_4D-smpl.txt   2>&1 &

# nohup transformation_scripts/data_transfrom-coord_speed-mpipe.sh  >  c_s-mpipe.txt  2>&1 &
# nohup transformation_scripts/data_transfrom-coord_speed-ntu.sh    >  c_s-ntu.txt    2>&1 &
# nohup transformation_scripts/data_transfrom-coord_speed-smpl.sh   >  c_s-smpl.txt   2>&1 &





# CV work
# nohup bash_scripts/cv_work-pca-mpipe.sh  >  cv_work-pca-mpipe.txt  2>&1 &
# nohup bash_scripts/cv_work-pca-ntu.sh   >  cv_work-pca-ntu.txt    2>&1 &
# nohup bash_scripts/cv_work-pca-smpl.sh   >  cv_work-pca-smpl.txt   2>&1 &

