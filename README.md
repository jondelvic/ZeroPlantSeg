# ZeroPlantSeg

<p>
  <a href="https://arxiv.org/abs/2509.09116" target='_blank'>
    <img src="http://img.shields.io/badge/arXiv-2509.09116-b31b1b?logo=arxiv&logoColor=b31b1b" alt="ArXiv">
  </a>
</p>

This repository contains the official implementation of [Zero-shot Hierarchical Plant Segmentation via Foundation Segmentation Models and Text-to-image Attention](https://arxiv.org/abs/2509.09116)
(Junhao Xing, Ryohei Miyakawa, Yang Yang, Xinpeng Liu, [Risa Shinoda](https://sites.google.com/view/risashinoda/home), [Hiroaki Santo](https://sites.google.com/view/hiroaki-santo/), Yosuke Toda, [Fumio Okura](https://fokura.jp/)).

---
This fork adapts ZeroPlantSeg for overlapping cabbage leaf and plant instance segmentation on Windows (WSL2) with an NVIDIA RTX 5050 Laptop GPU. The original pipeline used detectron2 + OVSeg + GroundingDINO.

As of now, I'm having dependency hell issues making the original pipeline work. After some setup troubleshooting, I used SAM and OpenCLIP (for unnecessary grass) for leaf segmentation and centroid-based DBSCAN for plant grouping. This is a lighter pipeline and I somehow made it work on my RTX 5050 Laptop? 

***NOTE: this is a working progress experiment***

Here's a hand-picked *good* initial segmentation result overlay on the original sample image:
![Sample Result](init-goodresult_IMG_20260310_103832_(1)_plant_overlay.jpg) 

## Installation
### WSL2 Setup (RTX 5050)

**System requirements:**
- Windows 11, WSL2 enabled
- NVIDIA Studio Driver >= 525 (INSTALL THIS DRIVER ON WINDOWS NOT ON WSL!!!) *apparently studio driver is better and more stable for deep learning stuff*
- Ubuntu 22.04 inside WSL2
- Python 3.10

Open PowerShell as Administrator and run the ff command: 
```
wsl --install -d Ubuntu-22.04
```
After this, it will prompt you to set your username and password. Restart your machine then relaunch Ubuntu.

Perform Ubuntu system upgdate: 
```
sudo apt update && sudo apt upgrade -y
```

***[OPTIONAL]*** For my machine, I set the ff memory/CPU limits via a *.wslconfig* file stored at `C:\Users\<Your Username\.wslconfig`.
```
[wsl2]
memory=12GB
processors=6
swap=8GB
```
Restart WSL and relaunch Ubuntu after. This can also be changed via the WSL Settings executable found in your Start Menu.

### NVIDIA Driver + CUDA Setup
Install the NVDIA Studio Driver for RTX 5050 on Windows from the [nvdia website](https://www.nvidia.com/en-us/drivers/). Do a clean install and verify if it's working in PowerShell (Windows not WSL).
```
nvidia-smi #this should show ur GPU and driver version >= 525
```

Inside WSL, install CUDA Toolkit 12.8. The `.deb` URL below was valid as of June 13, 2026 and if it doesn't work, find the installer at https://developer.nvidia.com/cuda-toolkit-archive 
```
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-wsl-ubuntu-12-8-local_12.8.0-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-12-8-local_12.8.0-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/ # you will be prompted afterwards what the GPG key is since it's not yet installed
sudo apt update
sudo apt -y install cuda-toolkit-12-8
```

Add the ff to `~/.bashrc`:
```
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
```

Then rebuild: `source .bashrc` (assuming ur at root)

Afterwards, verify `nvidia-smi` inside WSL and it should show your GPU.
```
nvidia-smi
nvcc --version #this should report 12.8
```

### Python Environment using Miniconda
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda init
source ~/.bashrc
```

You will now be at `(base) username@linux:~$`
Before creating the Python 3.10 environment, accept the TOS.
```
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda create -n zeroplantseg python=3.10 -y
```

To activate the environment: 
```
conda activate zeroplantseg
```

---

**All steps onwards MUST be run insided the `(zeroplantseg)` conda environment. Make sure to activate environment when you open a new terminal. DO NOT INSTALL THE PACKAGES INTO `(base)`.**

**PyTorch Setup** <br>
Since stable PyTorch doesn't support sm_120 (Blackwell architectures) as of now, I used a nightly build for now. If you encounter issues, check `torch.__version__` against the current working version for me: `torch 2.13.0.dev20260610+cu130`
```
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130
```

Verify if GPU is visible:
```
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_capability(0))" #expected: True (12, 0)
```

**Dependencies** <br>
Use the RTX 5050-specific requirements file to download and install the dependencies and install `segment-anything` separately:
```
pip install -r requirements_5050_laptop.txt
pip install git+https://github.com/facebookresearch/segment-anything.git@6fdee8f2727f4506cfbbe553e23b895e27956588
```

**Checkpoints** <br>
```
mkdir -p checkpoints
wget -O checkpoints/sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```
> The original ZeroPlantSeg made use of SAM VIT-H + GroundDINO + OVSeg weights. This fork, for now, only needs SAM VIT-B.

## Usage

