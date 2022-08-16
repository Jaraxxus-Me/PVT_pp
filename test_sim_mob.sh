export PYTHONPATH=/path/to/PVT++:$PYTHONPATH
# Mob
# DTB70
python tools/test_rt_f_sim.py --config 'experiments/siamrpn_mobilev2_l234_dwxcorr/pre_lb_config.yaml' \
--snapshot "my_models/RPN_Mob_M.pth" --dataset 'DTB70' --datasetroot 'testing_dataset/DTB70' --sim_info 'testing_dataset/sim_info/DTB70_SiamRPN++_Mob_sim.pkl'
python tools/test_rt_f_sim.py --config 'experiments/siamrpn_mobilev2_l234_dwxcorr/pre_lbv_config.yaml' \
--snapshot "my_models/RPN_Mob_V.pth" --dataset 'DTB70' --datasetroot 'testing_dataset/DTB70' --sim_info 'testing_dataset/sim_info/DTB70_SiamRPN++_Mob_sim.pkl'
python tools/test_rt_f_sim.py --config 'experiments/siamrpn_mobilev2_l234_dwxcorr/pre_mv_config.yaml' \
--snapshot "my_models/RPN_Mob_MV.pth" --dataset 'DTB70' --datasetroot 'testing_dataset/DTB70' --sim_info 'testing_dataset/sim_info/DTB70_SiamRPN++_Mob_sim.pkl'
# # UAVDT
python tools/test_rt_f_sim.py --config 'experiments/siamrpn_mobilev2_l234_dwxcorr/pre_lb_config.yaml' \
--snapshot "my_models/RPN_Mob_M.pth" --dataset 'UAVDT' --datasetroot 'testing_dataset/UAVDT' --sim_info 'testing_dataset/sim_info/UAVDT_SiamRPN++_Mob_sim.pkl'
python tools/test_rt_f_sim.py --config 'experiments/siamrpn_mobilev2_l234_dwxcorr/pre_lbv_config.yaml' \
--snapshot "my_models/RPN_Mob_V.pth" --dataset 'UAVDT' --datasetroot 'testing_dataset/UAVDT' --sim_info 'testing_dataset/sim_info/UAVDT_SiamRPN++_Mob_sim.pkl'
python tools/test_rt_f_sim.py --config 'experiments/siamrpn_mobilev2_l234_dwxcorr/pre_mv_config.yaml' \
--snapshot "my_models/RPN_Mob_MV.pth" --dataset 'UAVDT' --datasetroot 'testing_dataset/UAVDT' --sim_info 'testing_dataset/sim_info/UAVDT_SiamRPN++_Mob_sim.pkl'
# # UAV20
python tools/test_rt_f_sim.py --config 'experiments/siamrpn_mobilev2_l234_dwxcorr/pre_lb_config.yaml' \
--snapshot "my_models/RPN_Mob_M.pth" --dataset 'UAV20' --datasetroot 'testing_dataset/UAV20L' --sim_info 'testing_dataset/sim_info/UAV20_SiamRPN++_Mob_sim.pkl'
python tools/test_rt_f_sim.py --config 'experiments/siamrpn_mobilev2_l234_dwxcorr/pre_lbv_config.yaml' \
--snapshot "my_models/RPN_Mob_V.pth" --dataset 'UAV20' --datasetroot 'testing_dataset/UAV20L' --sim_info 'testing_dataset/sim_info/UAV20_SiamRPN++_Mob_sim.pkl'
python tools/test_rt_f_sim.py --config 'experiments/siamrpn_mobilev2_l234_dwxcorr/pre_mv_config.yaml' \
--snapshot "my_models/RPN_Mob_MV.pth" --dataset 'UAV20' --datasetroot 'testing_dataset/UAV20L' --sim_info 'testing_dataset/sim_info/UAV20_SiamRPN++_Mob_sim.pkl'
# # UAV123
python tools/test_rt_f_sim.py --config 'experiments/siamrpn_mobilev2_l234_dwxcorr/pre_lb_config.yaml' \
--snapshot "my_models/RPN_Mob_M.pth" --dataset 'UAV123' --datasetroot 'testing_dataset/UAV123' --sim_info 'testing_dataset/sim_info/UAV123_SiamRPN++_Mob_sim.pkl'
python tools/test_rt_f_sim.py --config 'experiments/siamrpn_mobilev2_l234_dwxcorr/pre_lbv_config.yaml' \
--snapshot "my_models/RPN_Mob_V.pth" --dataset 'UAV123' --datasetroot 'testing_dataset/UAV123' --sim_info 'testing_dataset/sim_info/UAV123_SiamRPN++_Mob_sim.pkl'
python tools/test_rt_f_sim.py --config 'experiments/siamrpn_mobilev2_l234_dwxcorr/pre_mv_config.yaml' \
--snapshot "my_models/RPN_Mob_MV.pth" --dataset 'UAV123' --datasetroot 'testing_dataset/UAV123' --sim_info 'testing_dataset/sim_info/UAV123_SiamRPN++_Mob_sim.pkl'