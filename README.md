# AdaptiGraph: Material-Adaptive Graph-Based Neural Dynamics for Robotic Manipulation

<span class="author-block">
<a target="_blank" href="https://kywind.github.io/">Kaifeng Zhang</a><sup>1*</sup>,
</span>
<span class="author-block">
<a target="_blank" href="https://www.linkedin.com/in/baoyu-li-b1646b220/">Baoyu Li</a><sup>1*</sup>,
</span>
<span class="author-block">
<a target="_blank" href="https://kkhauser.web.illinois.edu/">Kris Hauser</a><sup>1</sup>,
</span>
<span class="author-block">
<a target="_blank" href="https://yunzhuli.github.io/">Yunzhu Li</a><sup>1,2</sup>
</span>

<span class="author-block"><sup>1</sup>University of Illinois Urbana-Champaign,</span>
<span class="author-block"><sup>2</sup>Columbia University,</span>
<span class="author-block"><sup>*</sup>Equal contribution</span>

[Website](https://robopil.github.io/adaptigraph/) |
[Paper](https://robopil.github.io/adaptigraph/AdaptiGraph_RSS24.pdf) |
[ArXiv](https://arxiv.org/abs/2407.07889) |
[Data&Ckpt](https://drive.google.com/drive/folders/1HR39B7PXbYkM2w3XcA3feTCRCZH1q7UG?usp=sharing)

<img src="assets/teaser.png" alt="drawing" width="100%"/>

## 🛠️ Installation

We recommend installing the required packages in the following order to avoid potential version conflicts:

### Prerequisite

PyTorch with CUDA support are required. Our code is tested on python 3.9, torch 2.2.1, CUDA 12.1, and RTX 4090. Other torch and CUDA versions should suffice, but there might be conflicts when the cuda version used to compile torch is different from the cuda version detected in the system.

### Setup an environment

We recommend installing the packages in the following order:
```
# prepare python env
conda create -n python=3.9 adaptigraph
conda activate adaptigraph

# install PyTorch. We use 2.2.1+cu121 as an example:
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121

# install required packages for the dynamics model
pip install opencv-contrib-python scipy scikit-optimize cma pyyaml dgl matplotlib open3d threadpoolctl gdown ipdb pydantic moviepy
```

The following are optional installation steps for real planning experiments and simulation data generation:
```
# (optional) install packages for real experiments using xArm and Realsense cameras
git clone https://github.com/xArm-Developer/xArm-Python-SDK.git
cd xArm-Python-SDK
python setup.py install
cd ..
pip install pyrealsense2

# (optional) install GroundingDINO for real planning experiments
# Please make sure you have a CUDA environement and the environment variable CUDA_HOME is set
echo $CUDA_HOME  # check CUDA_HOME
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .
cd ..
mkdir -p src/planning/dump/weights
cd src/planning/dump/weights
gdown 1pc195KITtCCLteHkxD7mr1w-cG8XAvTs  # download DINO+SAM weights
gdown 1X-Et3-5TdSWuEUfq9gs_Ng-V-hypmLdB
gdown 1HR3O-rMv9qJoTrG6GNxi-4ZVREQzZ1Rf
gdown 1kN4trMTo5cavUqRSkYu0uzJq4mcL_ul_
cd -

# (optional) install additional packages for simulation data generation
pip install pymunk beautifulsoup4 pybullet gym
```

### Install PyFleX (optional)

Install PyFleX if you need to generate simulation data.

We are using a docker image to compile PyFleX, so before starting make sure you have the following packages:
- [docker-ce](https://docs.docker.com/engine/install/ubuntu/)
- [nvidia-docker](https://github.com/NVIDIA/nvidia-docker#quickstart)

Full installation:
```
pip install "pybind11[global]"
sudo docker pull xingyu/softgym
```
Run `bash install_pyflex.sh`. You may need to `source ~/.bashrc` to `import PyFleX`.

Or you can manually run
```
# compile pyflex in docker image
# re-compile if source code changed
# make sure ${PWD}/PyFleX is the pyflex root path when re-compiling
sudo docker run \
    -v ${PWD}/PyFleX:/workspace/PyFleX \
    -v ${CONDA_PREFIX}:/workspace/anaconda \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --gpus all \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -it xingyu/softgym:latest bash \
    -c "export PATH=/workspace/anaconda/bin:$PATH; cd /workspace/PyFleX; export PYFLEXROOT=/workspace/PyFleX; export PYTHONPATH=/workspace/PyFleX/bindings/build:$PYTHONPATH; export LD_LIBRARY_PATH=$PYFLEXROOT/external/SDL2-2.0.4/lib/x64:$LD_LIBRARY_PATH; cd bindings; mkdir build; cd build; /usr/bin/cmake ..; make -j"

# import to system paths. run these if you do not have these paths yet in ~/.bashrc
echo '# PyFleX' >> ~/.bashrc
echo "export PYFLEXROOT=${PWD}/PyFleX" >> ~/.bashrc
echo 'export PYTHONPATH=${PYFLEXROOT}/bindings/build:$PYTHONPATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=${PYFLEXROOT}/external/SDL2-2.0.4/lib/x64:$LD_LIBRARY_PATH' >> ~/.bashrc
echo '' >> ~/.bashrc
```

## 🖥️ Data

### Simulation data

We share a small set of simulation data for this project at [sim_data](https://drive.google.com/drive/folders/1HR39B7PXbYkM2w3XcA3feTCRCZH1q7UG?usp=sharing), which contains 100 episodes (98 for training and 2 for validation and visualization) for each material. Download and upzip the data, and put it into `./sim_data/`, the data format should be:

```
sim_data
├── rope
    ├── cameras
        ├── extrinsic.npy
        ├── intrinsic.npy
    ├── 000000
        ├── property_params.pkl
        ├── 00.h5
        ├── 01.h5
        ├── ...
    ├── ...
    ├── 000099
├── granular
├── cloth
```

The `.h5` files represent the trajectory for each pushing, which store the information as following:
```
{
    'info': 
    {
        'n_cams': number of cameras, 
        'timestamp': number of frames (T), 
        'n_particles': number of object particles
    },
    'action': action (action_dim, ) float32,
    'positions': raw object particle positions (T, N_obj, 3) float32,
    'eef_states': raw end effector states (T, N_eef, 14) float32,
    'observations': # Only effective for validation set
    {   'color': color_imgs {'cam_0': (T, H, W, 3), 'cam_1': ...},
        'depth': depth_imgs {'cam_0': (T, H, W), 'cam_1': ...},
    }
}
```

### Preprocess data

To preprocess the simulation data, run the following command:
```shell
cd src
python dynamics/preprocess/preprocess.py --config config/dynamics/rope.yaml
```
where `rope.yaml` can be replaced by other material yaml file. 

The preprocessed data should be saved into `./preprocess/`. The format of preprocess data is as follows:
```
preprocess
├── rope
    ├── frame_pairs
        ├── 000000_01.txt
        ├── 000000_02.txt
        ├── ...
        ├── 000099_05.txt
    ├── positions.pkl
    ├── phys_range.txt
    ├── metadata.txt
```
where the positions save preprocessed `object particle positions` and `eef particle posistions`.

### Data generation (Optional)

In our experiment, we generated 1000 episodes (900 for training and 100 for validation) for each material. Due to the raw data is huge, we provided a small dataset as above. We also provide the simulation data generation code, please install `PyFleX` as described in the `Installation` section before you implement it. 

To generate the simulation data, run the following command:
```shell
cd src
python sim/data_gen/data_gen.py --config config/data_gen/rope.yaml --save
```
where `rope.yaml` can be replaced by other material yaml file. 

## 🤖 Dynamics

### Rollout 

We provide the model checkpoints for each material at [model_ckpt](https://drive.google.com/drive/folders/1HR39B7PXbYkM2w3XcA3feTCRCZH1q7UG?usp=sharing). Please download it and put the checkpoint into `./log/{data_name}/checkpoints`. The dir format should be:
```
log
├── rope
    ├── checkpoints
        ├── model_100.pth
├── granular
├── cloth
```
Before rollout, please download the data and preprocess data as described above. 

To evaluate and visualize the performance, please run the following command:
```shell
cd src
python dynamics/rollout/rollout.py --config config/dynamics/rope.yaml --epoch '100' --viz
```
where `rope.yaml` can be replaced by other material yaml file. The rollout results will be saved to `./rollout/rollout-rope-model_100/`. The visualization can be found at `/short/cam_0/both.mp4`. 

### Training

To train the dynamics model for each material, please run the following command:
```shell
cd src
python dynamics/train/train.py --config config/dynamics/rope.yaml 
```
where `rope.yaml` can be replaced by other material yaml file. The training logs and checkpoints will be saved to `./log/`.


## 🦾 Inference and Planning

### Real robot

For inferencing the physical properties of unseen objects and carrying out manipulation tasks, a robot workspace similar to our setting is necessary. This includes an xArm 6 robot and four realsense D455 cameras. We highly suggest modifying our code instead of using them directly on other robot workspaces.

To calibrate the robot, please refer to
```
cd src
python planning/perception.py --calibrate
```
To interact with the object and optimize physics parameters, please refer to
```
python planning/random_interact.py --task_config <task config here, e.g., config/planning/granular.yaml> --use_ppo 
```
To execute planning tasks, please refer to
```
python planning/plan.py --task_config <task config here, e.g., config/planning/granular.yaml> --use_ppo
```

### Demo

In the case that a real robot environment is unavailable, we provide some [demo data](https://drive.google.com/drive/folders/1fYBlNAY1lhhRTYDskdzDfi7BQeXLa-8X?usp=sharing) to illustrate the physics parameter optimization ability of our method. Please download the demo data and save it as follows:
```
planning
├── dump
    ├── vis_demo
        ├── granular_1
            ├── interaction_0.npz
            ├── interaction_1.npz
            ├── ...
```
To try physics parameter optimization, run the following script:
```
cd src
python planning/demo/demo_granular_1.py
```
The result should show that the estimated parameter is around 0.04, indicating that the granularity is low. This aligns with the ground truth as the object is a pile of coffee beans, which has small granular sizes.

## 💗 Citation
If you find this repo useful for your research, please consider citing the paper
```
@inproceedings{zhang2024adaptigraph,
      title={AdaptiGraph: Material-Adaptive Graph-Based Neural Dynamics for Robotic Manipulation},
      author={Zhang, Kaifeng and Li, Baoyu and Hauser, Kris and Li, Yunzhu},
      booktitle={Proceedings of Robotics: Science and Systems (RSS)},
      year={2024}
    }
```

## 🙏 Acknowledgement

* Our Simulation environment is built upon [PyFleX](https://github.com/YunzhuLi/PyFleX), [SoftGym](https://github.com/Xingyu-Lin/softgym), [FlingBot](https://github.com/real-stanford/flingbot), and [Dyn-Res](https://github.com/WangYixuan12/dyn-res-pile-manip).
* Our dynamics model is built upon [VGPL](https://github.com/YunzhuLi/VGPL-Dynamics-Prior) and [RoboCook](https://github.com/hshi74/robocook?tab=readme-ov-file).

We appreciate the authors for their greate work and dedication to open source! 
