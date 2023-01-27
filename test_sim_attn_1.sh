export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH=/ocean/projects/cis220061p/bli5/CVPR23/code/PVT_pp:$PYTHONPATH
# Mob
# DTB70
python3 tools/test_rt_f_sim.py --config 'experiments/siamrpn_mobilev2_l234_dwxcorr/pre_mv_attn_config.yaml' \
--snapshot "models/RPN_Mob_Attn.pth" --dataset 'UAVDT' --datasetroot 'testing_dataset/UAVDT' --sim_info 'testing_dataset/sim_info/UAVDT_SiamRPN++_Mob_sim.pkl'