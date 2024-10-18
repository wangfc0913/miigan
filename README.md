# <h1 align = "center">MIIGAN</h1>

<center>Fuchao Wang</center>

<center>Northeastern University, Shenyang, Liaoning, China</center>

**Abstract** XXXXXXX

<h2>Architecture</h2>

<img src="figs/MIIGANOverview.PNG" alt="Alt text" title="Architecture" style="zoom: 80%;" />

| ![](figs\MIIGAN-Gen.PNG) | ![](figs\MIIGAN-Disc.PNG) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |

<h2>Results</h2>

![](figs\DifferentModelMetric.PNG)

The code is referenced from InfraGAN [code](https://github.com/makifozkanoglu/InfraGAN).

## Prerequisites
- Linux
- GPU

## Getting Started
### Installation
- Clone this repo:
```bash
git clone https://github.com/wangfc0913/miigan.git
```
- Install dependencies by using the command below
```bash
pip install -r requirements.txt
```
**Note**:  The project uses torch version =2.1.1 and torchvision version =0.16.1 and cu118.

​             Here is the  [reference link](https://github.com/JCruan519/VM-UNet) for Mamba installation.

- Install python libraries [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate).
```bash
pip install visdom
pip install dominate
```

visdom is a visualization tool that requires opening an additional terminal and running the command `python -m visdom.server` when using it, and click the URL http://localhost:8097.

### Download datasets

Download  datasets from: Google Drive ([google](https://drive.google.com/file/d/1FXhYbDdqrrHERm8a20drlR18Ylj8iRDY/view?usp=drive_link)) or BaiDuNetdisk([baidu](https://pan.baidu.com/s/1r3h8XDoVDMhiVHeobV7qpg?pwd=zge6))

Place the extracted folder into the project. The project structure is as follows:

```text
- miigan
  -- data
  -- datasets
     --- DroneVehicle
         ---- test
         ---- train
     --- KAIST
         ...
     --- LLVIP
         ...
     --- VEDAI_512
         ...
  -- ...
  -- util
     --- ...
     --- plot.py  # After the training is completed, you can execute this file to compile the results.
```
### Train

```bash
sh train.sh
```
### Test

```bash
sh test.sh
```
### Evaluate

```bash
sh evaluate.sh
```

 # Citation
```

```
