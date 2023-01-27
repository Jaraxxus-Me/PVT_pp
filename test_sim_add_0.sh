export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=/ocean/projects/cis220061p/bli5/CVPR23/code/PVT_pp:$PYTHONPATH
# Mob
# DTB70
python3 tools/test_rt_f_sim.py --config 'experiments/siamrpn_mobilev2_l234_dwxcorr/pre_mv_add_config.yaml' \
--snapshot "models/RPN_Mob_Add.pth" --dataset 'DTB70' --datasetroot 'testing_dataset/DTB70' --sim_info 'testing_dataset/sim_info/DTB70_SiamRPN++_Mob_sim.pkl'