# DynaPP

Our code is based on https://github.com/ultralytics/yolov5.

## Please read 'guideline.pdf'

---
Inside 'Run.ipynb'

Common options ("val.py") (will be updated more)

"--pack": DynaPP acceleration

"--duration": key frame duration length

"--background": background amount (d%)

"--data": *.data path

"--weights": model.pt path(s)

"--dataset_name": dataset name


---

## Prepare hardware

Our code is based on PyTorch.

Installing PyTorch in jetson edge devices is different from general installation.

Jetson Nano : https://qengineering.eu/install-pytorch-on-jetson-nano.html

Nvidia Jetson TX2 : https://medium.com/hackers-terminal/installing-pytorch-torchvision-on-nvidias-jetson-tx2-81591d03ce32

Wheel installers : https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048

+ If error occurs as : usr/lib/aarch64-linux-gnu/libgomp.so.1: cannot allocate memory in static tls bloc

Write this before access jupyter notebook : export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

---

## Please download weight files below

### (Put the files in 'weights' folder)

https://drive.google.com/file/d/1LTSKE19bpygugylP9jMk2dtjdgcQZ1vu/view?usp=share_link

https://drive.google.com/file/d/19zIMTZzF9tqOnpDBxMkoKz6u7S3-x7CW/view?usp=share_link

---

## Please download datasets below 
### (Put the files in directory you want, and modify the code inside 'Run.ipynb'

These dataset annotations have been converted to yolo format.

##### AUAIR

https://drive.google.com/file/d/1syHeOWTO5cIw3pjE68TWQdhzZPfTsHTv/view?usp=share_link

##### VisDrone

https://drive.google.com/file/d/1f02BSNxu0QAkimABYEJeLMSR01Tk1Tnr/view?usp=share_link

##### UAVDT

https://drive.google.com/file/d/1MpPPzEgjuRH3DjwFE0jhDxscSzqMjPpW/view?usp=share_link

##### ImageVID

https://drive.google.com/file/d/1w_K7uV4C_VxM5NryFpJFQC8OtSZbPIde/view?usp=share_link



