# PVT++: A Simple End-to-End Latency-Aware Visual Tracking Framework

### Bowen Li*, Ziyuan Huang*, Junjie Ye, Yiming Li, Sebastian Scherer, Hang Zhao, and Changhong Fu
### Our paper is accepted at ICCV 2023 !!




## Abstract

Visual object tracking is essential to intelligent robots. Most existing approaches have ignored the online latency that can cause severe performance degradation during real-world processing. Especially for unmanned aerial vehicles (UAVs), where robust tracking is more challenging and onboard computation is limited, the latency issue can be fatal. In this work, we present a simple framework for end-to-end latency-aware tracking, _i.e._, end-to-end predictive visual tracking (PVT++). Unlike existing solutions that naively append Kalman Filters after trackers, PVT++ can be jointly optimized, so that it takes not only motion information but can also leverage the rich visual knowledge in most pre-trained tracker models for robust prediction. Besides, to bridge the training-evaluation domain gap, we propose a relative motion factor, empowering PVT++ to generalize to the challenging and complex UAV tracking scenes. These careful designs have made the small-capacity lightweight PVT++ a widely effective solution. Additionally, this work presents an extended latency-aware evaluation benchmark for assessing an _any-speed_ tracker in the online setting. Empirical results on a robotic platform from the aerial perspective show that PVT++ can achieve significant performance gain on various trackers and exhibit higher accuracy than prior solutions, largely mitigating the degradation brought by latency. 


## Overview

We provide baseline results and trained models available for download in the [PVT++ Model Zoo](MODEL_ZOO.md).

**TODO.**
- [x] Code for PVT++
	- [x] Train
	- [x] Test
- [x] Code for [e-LAE](https://github.com/Jaraxxus-Me/e-LAE.git)
- [ ] All the official models
	- [x] SiamRPN++_Mob
	- [ ] SiamRPN++_Res
 	- [ ] SiamMask
- [ ] All the raw results for PVT++
	- [x] SiamRPN++_Mob
	- [ ] SiamRPN++_Res
 	- [ ] SiamMask
- [ ] All the vanilla tracker online results
     



## Installation

Please create a python environment including:

Python                  3.7.12

numpy                   1.19.2

CUDA compiler           CUDA 11.0

PyTorch                 1.7.0

Pillow                  8.3.1

torchvision             0.8.1

fvcore                  0.1.5

cv2                     4.5.4

colorama         0.4.4

tensorboardx             2.5.1

We are basically using [PySOT](https://github.com/STVIR/pysot) environments.


## Dataset Preparation

#### 1. Download test datasets

[DTB70](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/download/14338/14292), [UAVDT](https://openaccess.thecvf.com/content_ECCV_2018/papers/Dawei_Du_The_Unmanned_Aerial_ECCV_2018_paper.pdf), [UAV123,UAV20L](https://link.springer.com/chapter/10.1007/978-3-319-46448-0_27)

Put them into `testing_dataset` directory as:

```shell
testing_dataset/
	DTB70/
		Animal1/
		...
    UAVDT/
    	anno/
    	data_seq/
    UAV20L/
    	anno/
    	data_seq/
    ...
```

#### 2. Download training datasets

[VID](http://image-net.org/challenges/LSVRC/2017/), [LaSOT](https://paperswithcode.com/dataset/lasot), [GOT-10k](http://got-10k.aitestunion.com/downloads)

Put them into `training_dataset` directory as:

```shell
training_dataset/
	got10k/
		data/
			GOT-10k_Train_000001/
			...
		gen_json.py
		train.json
    lasot/
    	data/
    		airplane-1/
    		...
    	gen_json.py
    	gen_txt.py
    	train.json
    vid/
    	ILSVRC2015/
    		Annotations/
    		Data/
    		ImageSets/
    gen_json.py
    parse_vid.py
    train.json
```

#### 3. Generating train.json

```shell
cd training_dataset/got10k
python gen_json.py

cd training_dataset/lasot
python gen_txt.py
python gen_json.py

cd training_dataset/vid
python parse_vid.py
python gen_json.py
```

#### Note

You make check the dataset paths in `/PVT++/pysot/core/config.py` Line163-183



## Test models

#### 1. Add PVT++ to your PYTHONPATH

```bash
export PYTHONPATH=/path/to/PVT++:$PYTHONPATH
```

#### 2. Download PVT++ models
Download models in [PVT++ Model Zoo](MODEL_ZOO.md) and put the them in `my_models/`.

#### 3. Test models on Nvidian Jetson AGX Xavier (You may find [this tutorial](https://github.com/Jaraxxus-Me/AirDet_ROS.git) useful to set up env on AGX)

```shell
bash test_mob_agx.sh
```

#### 4. Test models on PC with the simulated latency

##### 4.1 Generate simulated latency

Download our [Raw_results](https://mega.nz/file/tFd02RxC#98PDk3XDhcXo9sZ-seKP5aklT0xC8rvbcUm77xu1Cmo), put it in PVT++ folder

```shell
python tools/gen_sim_info.py
```

You may need to specify the datasets in the file

The simulation pkl files will be in `testing_dataset/sim_info`

##### 4.2 Test with recorded latency

```shell
bash test_sim_mob.sh
```

You'll generate the raw results in `results_rt_raw/`



## Evaluation

#### 1. Convert .pkl files to .txt files

```shell
bash convert.sh # sigma = 0, predictive trackers, results in /results_rt_raw
# output results are in /results_rt
bash convert_new.sh # sigma = 0:0.02:1, original trackers, results in /Raw, we'll provide all the results soon
# output results are in /results_eLAE
```

### 2. Evaluation results

refer to [e-LAE](https://github.com/Jaraxxus-Me/e-LAE.git) code



##  Training :wrench:

Download *base tracking models* in [PVT++ Model Zoo](MODEL_ZOO.md) and put the them in `pretrained/`.

```shell
bash train.sh
```

The trained models will be in `/snapshot`

`LB5` refers to motion model, `lbv5` denotes visual predictor, `mv16` denotes joint model.

## Reference
If our work inspires your research, please cite us as:
```
@INPROCEEDINGS{Li2023iccv,       
	author={Li, Bowen and Huang, Ziyuan and Ye, Junjie and Li, Yiming and Scherer, Sebastian and Zhao, Hang and Fu, Changhong},   
	booktitle={Proceedings of the IEEE International Conference on Computer Vision (ICCV)}, 
	title={{PVT++: A Simple End-to-End Latency-Aware Visual Tracking Framework}},
	year={2023},
	volume={},
	number={},
	pages={1-18}
}
```
 


## Acknowledgement

Our work is motivated by [ECCV2020 "Towards Streaming Perception"](https://link.springer.com/chapter/10.1007/978-3-030-58536-5_28) and ["Predictive Visual Tracking"](https://arxiv.org/pdf/2103.04508.pdf), we express our gratitude to the authors. This library is developed upon [PySOT](https://github.com/STVIR/pysot), we sincerely thank the contributors and developers.
