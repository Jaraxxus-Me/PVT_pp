export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/ocean/projects/cis220061p/bli5/CVPR23/code/PVT_pp:$PYTHONPATH

# RPN_Mob
python ./tools/train.py --cfg 'experiments/siamrpn_mobilev2_l234_dwxcorr/pre_lb_config.yaml'
python ./tools/train.py --cfg 'experiments/siamrpn_mobilev2_l234_dwxcorr/pre_lbv_config.yaml'
python ./tools/train.py --cfg 'experiments/siamrpn_mobilev2_l234_dwxcorr/pre_mv_config.yaml'
