export PYTHONPATH=/path/to/PVT++:$PYTHONPATH
# PVT_pp2
RAW_ROOT='/ocean/projects/cis220061p/bli5/CVPR23/code/PVT_pp2/output_rt/test/tracking_results'
TAR_ROOT='/ocean/projects/cis220061p/bli5/CVPR23/code/PVT_pp2/rt_eva'
GT_ROOT='/ocean/projects/cis220061p/bli5/CVPR23/data'
# PVT_pp
RAW_ROOT='results_rt_raw'
TAR_ROOT='results_rt'
GT_ROOT='testing_dataset'
# \sigma=0
# python3 tools/rt_eva.py --raw_root "${RAW_ROOT}/DTB/" --tar_root "${TAR_ROOT}/DTB70/" --gtroot "${GT_ROOT}/DTB70"
python3 tools/rt_eva_pre.py --raw_root "${RAW_ROOT}/RealWorld/" --tar_root "${TAR_ROOT}/RealWorld/" --gtroot "${GT_ROOT}/real_world/anno"
# python tools/rt_eva_pre.py --raw_root './results_rt_raw/UAV20L/' --tar_root './results_rt/UAV20L' --gtroot 'testing_dataset/UAV20L/anno'
# python tools/rt_eva_pre.py --raw_root './results_rt_raw/UAV123/' --tar_root './results_rt/UAV123' --gtroot 'testing_dataset/UAV123/anno'