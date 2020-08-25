## change the image path and run the corresponding command line for different backbone

image_path=./data/images

## For ResNet-101

python train_imagenet_cam_loss.py --ngpu 8 --workers 20 --arch resnet --depth 101 --epochs 120 --batch-size 256 --lr 0.1 --att-type no_atten --weight 0.07 --loss-type mse --prefix RESNET101_IMAGENET_cam_loss_mse $image_path

python train_imagenet_cam_loss.py --ngpu 8 --workers 20 --arch resnet --depth 101 --epochs 120 --batch-size 256 --lr 0.1 --att-type no_atten --weight 0.0065 --loss-type kd --prefix RESNET101_IMAGENET_cam_loss_kd $image_path


## For ResNet-152

python train_imagenet_cam_loss.py --ngpu 8 --workers 20 --arch resnet --depth 152 --epochs 120 --batch-size 256 --lr 0.1 --att-type no_atten --weight 0.07 --loss-type mse  --prefix RESNET152_IMAGENET_ori_cam_loss_mse $image_path

python train_imagenet_cam_loss.py --ngpu 8 --workers 20 --arch resnet --depth 152 --epochs 120 --batch-size 256 --lr 0.1 --att-type no_atten --weight 0.0048 --loss-type kd  --prefix RESNET152_IMAGENET_ori_cam_loss_kd $image_path


