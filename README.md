# YOLOv2_Pytorch
This is a repository containing the implementation of YOLOv2.

This implementation is 100% pytorch implementation of YOLO_v2.

Original paper: [YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242) by Joseph Redmond and Ali Farhadi.

<p float="left">
  <img src="/images/out/dog.jpg" width="300" />
  <img src="/images/out/person.jpg" width="300" /> 
  <img src="/images/out/eagle.jpg" width="300" />
</p>
--------------------------------------------------------------------------------

## Requirements

- [Pytorch 0.3.0](https://pytorch.org/)
- [Numpy](http://www.numpy.org/)
- [Pillow](https://pillow.readthedocs.io/)
- [Python 3.6](https://www.python.org/)

### Installation
```bash
git clone https://github.com/mmderakhshani/YOLOv2_Pytorch.git
cd YOLOv2_Pytorch

# [Option 1] To replicate the conda environment:
conda env create -f environment.yml
source activate pytorch
# [Option 2] Install everything globaly.
```

## Quick Start

- Download the pre-trained model from (https://drive.google.com/file/d/0B21Wl06uvFPFSGlBdWF3RmhMa00/view) and put it in the model_data folder.
- Test the converted model on the small test set in `images/`.

```bash
python demo.py ./model_data/yolo.p
```

See `python demo.py --help` for more options.

--------------------------------------------------------------------------------

## More Details
`YOLOv2_Pytorch/pytorch_model` contains reference implementations of Darknet-19 and YOLO_v2.

## Known Issues and TODOs

- Evaluation on Pascal VOC 2007, 2010, 2012
- Evaluation on MSCOCO 2014: is ready to push
- Training script on MSCOCO 2014 dataset

## Darknets of Yore

YOLOv2_Pytorch stands on the shoulders of giants.

- :fire: [Darknet](https://github.com/pjreddie/darknet) :fire:
- [Darknet.Keras](https://github.com/sunshineatnoon/Darknet.keras) - The original D2K for YOLO_v1.
- [Darkflow](https://github.com/thtrieu/darkflow) - Darknet directly to Tensorflow.
- [caffe-yolo](https://github.com/xingwangsfu/caffe-yolo) - YOLO_v1 to Caffe.
- [yolo2-pytorch](https://github.com/longcw/yolo2-pytorch) - YOLO_v2 in PyTorch.
- [yad2k](https://github.com/allanzelener/YAD2K) - YOLO_v2 in Keras.

--------------------------------------------------------------------------------

