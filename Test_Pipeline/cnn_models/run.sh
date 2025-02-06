CUDA_VISIBLE_DEVICES=0 python vgg.py --num_epochs 1 --batch_size 32  > log/01-vgg16_32.txt
CUDA_VISIBLE_DEVICES=0 python vgg.py --num_epochs 1 --batch_size 64  > log/02-vgg16_64.txt
CUDA_VISIBLE_DEVICES=0 python vgg.py --num_epochs 1 --batch_size 128 > log/03-vgg16_128.txt

CUDA_VISIBLE_DEVICES=0 python resnet50.py --num_epochs 1 --batch_size 32  > log/04-resnet50_32.txt
CUDA_VISIBLE_DEVICES=0 python resnet50.py --num_epochs 1 --batch_size 64  > log/05-resnet50_64.txt
CUDA_VISIBLE_DEVICES=0 python resnet50.py --num_epochs 1 --batch_size 128 > log/06-resnet50_128.txt

CUDA_VISIBLE_DEVICES=0 python Xception.py --num_epochs 1 --batch_size 32  > log/07-Xception_32.txt
CUDA_VISIBLE_DEVICES=0 python Xception.py --num_epochs 1 --batch_size 64  > log/08-Xception_64.txt
CUDA_VISIBLE_DEVICES=0 python Xception.py --num_epochs 1 --batch_size 128 > log/09-Xception_128.txt

CUDA_VISIBLE_DEVICES=0 python mobilenet.py --num_epochs 1 --batch_size 32  > log/10-mobilenet_32.txt
CUDA_VISIBLE_DEVICES=0 python mobilenet.py --num_epochs 1 --batch_size 64  > log/11-mobilenet_64.txt
CUDA_VISIBLE_DEVICES=0 python mobilenet.py --num_epochs 1 --batch_size 128 > log/12-mobilenet_128.txt

CUDA_VISIBLE_DEVICES=0 python efficientNet.py --num_epochs 1 --batch_size 32  > log/13-efficientNet_32.txt
CUDA_VISIBLE_DEVICES=0 python efficientNet.py --num_epochs 1 --batch_size 64  > log/14-efficientNet_64.txt
CUDA_VISIBLE_DEVICES=0 python efficientNet.py --num_epochs 1 --batch_size 128 > log/15-efficientNet_128.txt

CUDA_VISIBLE_DEVICES=0 python inception.py --num_epochs 1 --batch_size 32  > log/16-inception_32.txt
CUDA_VISIBLE_DEVICES=0 python inception.py --num_epochs 1 --batch_size 64  > log/17-inception_64.txt
CUDA_VISIBLE_DEVICES=0 python inception.py --num_epochs 1 --batch_size 128 > log/18-inception_128.txt