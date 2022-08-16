# PVT++: A Simple End-to-End Latency-Aware Visual Tracking Framework

### CoRL 2022 Anonymous Submission 28




## Abstract

Visual object tracking is an essential capability of intelligent robots. Most existing approaches have ignored the online latency that can cause severe performance degradation during real-world processing. This work presents a simple framework for end-to-end latency-aware tracking, *i.e.*, end-to-end predictive visual tracking (PVT++). Our PVT++ is capable of turning most leading-edge trackers into predictive trackers by appending an online predictor. Unlike existing solutions that use model-based approaches, our framework is learnable, such that it can take not only motion information as input but it can also take advantage of visual cues or a combination of both. Moreover, since PVT++ is end-to-end optimizable, it can further boost the latency-aware tracking performance. Additionally, this work presents an extended latency-aware evaluation benchmark (e-LAE) for assessing an *any-speed* tracker in the online setting. Empirical results show that the motion-based PVT++ can obtain on par or better performance than existing approaches. Further incorporating visual information and joint training techniques, PVT++ can achieve up to **60%** performance gain on various trackers, essentially removing the degradation brought by their high latency onboard. 



## Overview

We provide baseline results and trained models available for download in the [PVT++ Model Zoo](MODEL_ZOO.md).

For submission, we provide models with SiamRPN++_Mob, all the rest models and results will be available upon acceptance.



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

We'll also provide the official docker image in the future for faster reproduction.



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

#### 3. Test models on Nvidian Jetson AGX Xavier

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
bash convert_new.sh # sigma = 0:0.02:1, original trackers, results in /Raw, we'll provide all the results upon acceptence
# output results are in /results_eLAE
```

### 2. Evaluation results

refer to `/e-LAE` code



##  Training :wrench:

Download *base tracking models* in [PVT++ Model Zoo](MODEL_ZOO.md) and put the them in `pretrained/`.

```shell
bash train.sh
```

The trained models will be in `/snapshot`

`LB5` refers to motion model, `lbv5` denotes visual predictor, `mv16` denotes joint model.

 


## Acknowledgement

This library is developed on [PySOT](https://github.com/STVIR/pysot), we sincerely thank the contributors and developers.