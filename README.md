## Code for eccv2020 paper: Fixing Localization Errors to Improve Image Classification

This repository contains the original code and the links for data and pretrained models. If you have any questions about [our paper](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700273.pdf), please feel free to contact [Me](https://github.com/GuoleiSun) (sunguolei.kaust AT gmail.com)   

![block images](https://github.com/GuoleiSun/HNC_loss/blob/master/diagram.png)

### HNC loss
To use our loss, please first generate CAMs following this [line](https://github.com/GuoleiSun/HNC_loss/blob/5d67612cb52780cc04d63344ee6c6672e3ef2a4b/imagenet/MODELS/model_resnet.py#L192).

Our loss can be found in [HNC_mse](https://github.com/GuoleiSun/HNC_loss/blob/5d67612cb52780cc04d63344ee6c6672e3ef2a4b/imagenet/train_imagenet_cam_loss.py#L65) and [HNC_kd](https://github.com/GuoleiSun/HNC_loss/blob/5d67612cb52780cc04d63344ee6c6672e3ef2a4b/imagenet/train_imagenet_cam_loss.py#L94). The usage of loss can be found in this [line](https://github.com/GuoleiSun/HNC_loss/blob/05459be169e0211e3c7dd9b2d9fd1b508712aecd/imagenet/train_imagenet_cam_loss.py#L277), where our loss takes two arguments: CAMs and ground-truth label.

Note: 1. you may need to tune lambda, which is the weight balancing both HNC loss and cross entropy loss. Higher weight means higher influence of HNC loss; 2. you may need to tune k, which is the number of negative CAMs to suppress. Smaller k means focusing on more confusing classes.

### ImageNet classification
1). prepare [ImageNet dataset](http://image-net.org/)

2). for running commands, please refer to './imagenet/running-script.sh'

### Citation
If you find this repository helpful, please consider citing:

@article{sun2020fixing,

  title={Fixing Localization Errors to Improve Image Classification},
  
  author={Sun, Guolei and Khan, Salman and Li, Wen and Cholakkal, Hisham and Khan, Fahad and Van Gool, Luc},
  
  journal={ECCV},
  
  year={2020}
}

### Acknowledgements 

This repository is based on [CBAM](https://github.com/Jongchan/attention-module), thanks for their excellent work.
