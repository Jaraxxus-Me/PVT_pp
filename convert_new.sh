export PYTHONPATH=/path/to/PVT++:$PYTHONPATH
# DTB70
python tools/rt_eva_new.py --raw_root Raw/DTB70 --tar_root results_eLAE/DTB70 --gtroot testing_dataset/DTB70
# UAVDT
python tools/rt_eva_new.py --raw_root Raw/UAVDT --tar_root results_eLAE/UAVDT --gtroot testing_dataset/UAVDT/anno
# UAV20L
python tools/rt_eva_new.py --raw_root Raw/UAV20L --tar_root results_eLAE/UAV20L --gtroot testing_dataset/UAV20L/anno
# UAV123
python tools/rt_eva_new.py --raw_root Raw/UAV123 --tar_root results_eLAE/UAV123 --gtroot testing_dataset/UAV123/anno