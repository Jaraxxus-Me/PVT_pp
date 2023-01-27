import pickle
import os

pkl_path = 'Raw_Results_RPN_Mob/UAVDT/SiamRPN++_Mob'
tracker = 'SiamRPN++_Mob'
dataset = 'UAVDT'
tgt_path = 'testing_dataset/sim_info'

pkls = os.listdir(pkl_path)
info_dict = {}

for pkl in pkls:
    name = pkl[0:-4]
    with open(os.path.join(pkl_path, pkl), 'rb') as run_file:
        pkl_info = pickle.load(run_file)
    init_time = pkl_info['runtime'][0]
    running_time = sum(pkl_info['runtime'][1:])/len(pkl_info['runtime'][1:])
    info_dict[name] = {'init_time': init_time, 'running_time': running_time}

with open(os.path.join(tgt_path, '{}_{}_sim.pkl'.format(dataset, tracker)), "wb") as f_sim:
    pickle.dump(info_dict, f_sim)
