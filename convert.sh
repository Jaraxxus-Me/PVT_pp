export PYTHONPATH=/path/to/PVT++:$PYTHONPATH
# \sigma=0
python tools/rt_eva_pre.py --raw_root './results_rt_raw/DTB70/' --tar_root './results_rt/DTB70' --gtroot 'testing_dataset/DTB70'
python tools/rt_eva_pre.py --raw_root './results_rt_raw/UAVDT/' --tar_root './results_rt/UAVDT' --gtroot 'testing_dataset/UAVDT/anno'
python tools/rt_eva_pre.py --raw_root './results_rt_raw/UAV20L/' --tar_root './results_rt/UAV20L' --gtroot 'testing_dataset/UAV20L/anno'
python tools/rt_eva_pre.py --raw_root './results_rt_raw/UAV123/' --tar_root './results_rt/UAV123' --gtroot 'testing_dataset/UAV123/anno'