## Code for eccv2020 paper: Fixing Localization Errors to Improve Image Classification

This repository contains the original code and the links for data and pretrained models. If you have any questions about [our paper](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700273.pdf), please feel free to contact [Me](https://github.com/GuoleiSun) (sunguolei.kaust AT gmail.com)   

![block images](https://github.com/GuoleiSun/HNC_loss/blob/master/diagram.png)

### HNC loss
To use our loss, please first generate CAMs following this [line of code](https://github.com/GuoleiSun/HNC_loss/blob/5d67612cb52780cc04d63344ee6c6672e3ef2a4b/imagenet/MODELS/model_resnet.py#L192).

Our loss can be found in [HNC_mse](https://github.com/GuoleiSun/HNC_loss/blob/5d67612cb52780cc04d63344ee6c6672e3ef2a4b/imagenet/train_imagenet_cam_loss.py#L65) and [HNC_kd](https://github.com/GuoleiSun/HNC_loss/blob/5d67612cb52780cc04d63344ee6c6672e3ef2a4b/imagenet/train_imagenet_cam_loss.py#L94).
