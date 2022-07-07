# FSCE with graph
 
## This is instruction on training detectron2 with graph

```
We incoporate the code of paper: Spatial-aware graph relation network for large-scale object detection
into detectron2 pipeline
https://github.com/simblah/SGRN_torch

```

## Structure of code

```
FSCE:
-build
    lib.linux-x86_64-3.7
    lib.linux-x86_64-3.8
    temp.linux-x86_64-3.7
    temp.linux-x86_64-3.8

-checkpoints:
    coco
    vaipe
        faster_rcnn
        FRCNN
        SGRN
            R_50_FPN_all
                test0306/
                test3005/
                model_0029999.pth
    vaipe_duy
        base
            model_0049999.pth
        srgn
            model_0059999.pth
    R-50.pkl
- configs
- datasets
    cache
        coco_2014_train_gt_roidb.pkl
        coco_2014_val_gt_roidb.pkl
        vaipe_test_gt_roidb.pkl
        vaipe_train_gt_roidb.pkl
- demo
- fsdet
- FsDet.egg-info
- output
- tools
- vaipe_dataset_huyp
    annotations
        instances_test.json
        instances_train.json
    data
        test
            img\
        train
            img\
    base_names.pkl
    few_shot_names.pkl
    name2id.pkl

```

## Directory to data


```
There are two datatsets:
vaipe: (dataset of huy, used for few shot learning)
annotations:
    home/huyen/projects/FSCE/vaipe/annotations/instances_test.json
    home/huyen/projects/FSCE/vaipe/annotations/instances_train.json

images:
    /home/huyen/projects/FSCE/vaipe_dataset_huyp/data/test/img/images
    /home/huyen/projects/FSCE/vaipe_dataset_huyp/data/train/img/images

vaipe_duy: (dataset of duy, take as in prescription, for graph based detection)
annotations:
    /home/huyen/projects/inside_duy/data/pills/data_test/instances_test.json
    /home/huyen/projects/inside_duy/data/pills/data_train/instances_train.json

images:
    /home/huyen/projects/inside_duy/data/pills/data_train/images
    /home/huyen/projects/inside_duy/data/pills/data_train/labels
    /home/huyen/projects/inside_duy/data/pills/data_test/images
    /home/huyen/projects/inside_duy/data/pills/data_test/labels




```

## How to run the code:

```
( Data duy or huyp)
Train:

Faster r-cnn:
python tools/train_net.py --config-file=/home/huyen/projects/FSCE/configs/VAIPE-detection/faster_rcnn_R_50_FPN_base.yaml
(huyp dataset)
python tools/train_net.py --config-file=/home/huyen/projects/FSCE/configs/VAIPE-detection/faster_rcnn_R_50_FPN_base_duy.yaml
(duy dataset)

SGRN:
python tools/train_net.py --config-file=/home/huyen/projects/FSCE/configs/VAIPE-detection/faster_rcnn_srgn_R_50_base_batch.yaml
(huyp dataset)
python tools/train_net.py --config-file=/home/huyen/projects/FSCE/configs/VAIPE-detection/faster_rcnn_R_50_FPN_sgrn_duy.yaml
(duy dataset)



Test:

Faster r-cnn:
python tools/train_net.py --config-file=/home/huyen/projects/FSCE/configs/VAIPE-detection/faster_rcnn_R_50_FPN_base.yaml --eval-only
(huyp dataset)
Model checkpoint: /home/huyen/projects/FSCE/checkpoints/vaipe/faster_rcnn/R_50_FPN_all/test2705/model_0029999.pth
python tools/train_net.py --config-file=/home/huyen/projects/FSCE/configs/VAIPE-detection/faster_rcnn_R_50_FPN_base_duy.yaml --eval-only
(duy dataset)
Model checkpoint: /home/huyen/projects/FSCE/checkpoints/vaipe_duy/base/model_0049999.pth

SGRN:
python tools/train_net.py --config-file=/home/huyen/projects/FSCE/configs/VAIPE-detection/faster_rcnn_srgn_R_50_base_batch.yaml --eval-only
(huyp dataset)
Model checkpoint: /home/huyen/projects/FSCE/checkpoints/vaipe/SGRN/R_50_FPN_all/test0306/model_0059999.pth
python tools/train_net.py --config-file=/home/huyen/projects/FSCE/configs/VAIPE-detection/faster_rcnn_R_50_FPN_sgrn_duy.yaml --eval-only
(duy dataset)
Model checkpoint: /home/huyen/projects/FSCE/checkpoints/vaipe_duy/srgn/model_0059999.pth




```
## Link for data and model download

```
Checkpoints:
https://drive.google.com/drive/folders/1XPO3KgQyPDp_4t2SIs_geWwQ0qwnsWjg?usp=sharing

Dataset:
https://drive.google.com/drive/folders/1W48xvdDfyNn-IG_jSRV7xnvr7fuSR4t6?usp=sharing

```

## Environment setup:

```

pytorch  1.10.1
cuda 11.3
    (conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge: GTX3090)
torch_geometric
    pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.1+cu113.html
    pip install torch-sparse -f https://data.pyg.org/whl/torch-1.10.1+cu113.html
    pip install torch-geometric
    (https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

(need to build according to FSCE's codebase instructions before run the code)

```
