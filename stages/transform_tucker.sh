python3 ../bullets/tucker_decompose.py -s data/skgest-ntu-center.npz -d data/skgest-ntu-center_tucker_full.plk
python3 ../bullets/tucker_decompose.py -s data/skgest-smpl-center.npz -d data/skgest-smpl-center_tucker_full.plk

python3 ../bullets/svd_decompose.py -s data/skgest-ntu-center.npz -d data/skgest-ntu-center_SVD_max.plk
python3 ../bullets/svd_decompose.py -s data/skgest-smpl-center.npz -d data/skgest-smpl-center_SVD_max.plk