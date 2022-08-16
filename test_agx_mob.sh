export PYTHONPATH=/path/to/PVT++:$PYTHONPATH
# Mob
# DTB70
python onboard/test_rt_f.py --config 'experiments/siamrpn_mobilev2_l234_dwxcorr/pre_lb_config.yaml' \
--snapshot "my_models/RPN_Mob_M.pth" --dataset 'DTB70' --datasetroot 'testing_dataset/DTB70'
python onboard/test_rt_f.py --config 'experiments/siamrpn_mobilev2_l234_dwxcorr/pre_lbv_config.yaml' \
--snapshot "my_models/RPN_Mob_V.pth" --dataset 'DTB70' --datasetroot 'testing_dataset/DTB70'
python onboard/test_rt_f.py --config 'experiments/siamrpn_mobilev2_l234_dwxcorr/pre_mv_config.yaml' \
--snapshot "my_models/RPN_Mob_MV.pth" --dataset 'DTB70' --datasetroot 'testing_dataset/DTB70'
# # UAVDT
python onboard/test_rt_f.py --config 'experiments/siamrpn_mobilev2_l234_dwxcorr/pre_lb_config.yaml' \
--snapshot "my_models/RPN_Mob_M.pth" --dataset 'UAVDT' --datasetroot 'testing_dataset/UAVDT'
python onboard/test_rt_f.py --config 'experiments/siamrpn_mobilev2_l234_dwxcorr/pre_lbv_config.yaml' \
--snapshot "my_models/RPN_Mob_V.pth" --dataset 'UAVDT' --datasetroot 'testing_dataset/UAVDT'
python onboard/test_rt_f.py --config 'experiments/siamrpn_mobilev2_l234_dwxcorr/pre_mv_config.yaml' \
--snapshot "my_models/RPN_Mob_MV.pth" --dataset 'UAVDT' --datasetroot 'testing_dataset/UAVDT'
# # UAV20
python onboard/test_rt_f.py --config 'experiments/siamrpn_mobilev2_l234_dwxcorr/pre_lb_config.yaml' \
--snapshot "my_models/RPN_Mob_M.pth" --dataset 'UAV20' --datasetroot 'testing_dataset/UAV20L'
python onboard/test_rt_f.py --config 'experiments/siamrpn_mobilev2_l234_dwxcorr/pre_lbv_config.yaml' \
--snapshot "my_models/RPN_Mob_V.pth" --dataset 'UAV20' --datasetroot 'testing_dataset/UAV20L'
python onboard/test_rt_f.py --config 'experiments/siamrpn_mobilev2_l234_dwxcorr/pre_mv_config.yaml' \
--snapshot "my_models/RPN_Mob_MV.pth" --dataset 'UAV20' --datasetroot 'testing_dataset/UAV20L'
# # UAV123
python onboard/test_rt_f.py --config 'experiments/siamrpn_mobilev2_l234_dwxcorr/pre_lb_config.yaml' \
--snapshot "my_models/RPN_Mob_M.pth" --dataset 'UAV123' --datasetroot 'testing_dataset/UAV123'
python onboard/test_rt_f.py --config 'experiments/siamrpn_mobilev2_l234_dwxcorr/pre_lbv_config.yaml' \
--snapshot "my_models/RPN_Mob_V.pth" --dataset 'UAV123' --datasetroot 'testing_dataset/UAV123'
python onboard/test_rt_f.py --config 'experiments/siamrpn_mobilev2_l234_dwxcorr/pre_mv_config.yaml' \
--snapshot "my_models/RPN_Mob_MV.pth" --dataset 'UAV123' --datasetroot 'testing_dataset/UAV123'