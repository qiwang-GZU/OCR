><h1 id='main'>操作手册</h1>

><h2>目录</h2>
+ [初始工作](#init)
  + [文件结构](#structure)
+ [检测](#detector)
  + [PANet检测](#PANetdetector)
  + [Baseline检测](#Baselinedetector)
+ [识别](#recognizer)
    + [Baseline识别](#Baselinerecognizer)
    + [vedastr](#vedastr)

><h2 id='init'>初始工作</h2>
<h3>文件结构</h3>

Baseline文件结构如下：
```
menu_data
├──official_data
│     ├──train_image_common/*
│     ├──train_image_special/*
│     ├──train_label_common.json
│     ├──train_label_special.json
│     ├──test_image/*
├──output
│     ├──detector_test_output/*
│     ├──test_null.json
│     ├──test_submission.json
├──checkpoints
│     ├──detector/*
│     ├──recognizer/*
├──tmp_data
      ├──recognizer_txts/*
      ├──recognizer_images/*
```
official_data文件夹包含从竞赛下载的数据。在执行此任务期间output包含一些输出。checkpoints文件夹包含所有经过训练的对象检测和文本识别模型。tmp_data文件夹包含预测过程中的临时数据。

PANet中，checkpoints文件夹包括生成的模型，data文件夹包括训练需要的数据，output文件夹包括输出的结果。pretrained文件夹中包含预训练模型，本文选择resnet18作为主干网络。models包含所需模型源代码。

><h2 id='detector'>检测</h2>
<h3 id='PANetdetector'>PANet检测</h3>
使用[PANet(Pixel Aggregation Network, 像素聚合网络)](https://arxiv.org/pdf/1908.05900.pdf),作为检测网络。

### 推荐环境
```
Python 3.6+
Pytorch 1.1.0
torchvision 0.3
mmcv 0.2.12
editdistance
Polygon3
pyclipper
opencv-python 3.4.2.17
Cython
```

如果需要，可以安装需求文档
### **安装**
```shell script
pip install -r requirement.txt
./compile.sh
```

### **训练**
```shell script
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py ${CONFIG_FILE}
```
例如:
```shell script
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py config/pan/pan_r18_ic15.py
```
训练获得的模型保存在:**./checkpoints/** 中

### **测试**
```
python test.py ${CONFIG_FILE} ${CHECKPOINT_FILE}
```
例如：
```shell script
python test.py config/pan/pan_r18_ic15.py checkpoints/pan_r18_ic15/checkpoint.pth.tar
```
### **速度**
可以显示图片生成速度：
```shell script
python test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --report_speed
```
例如:
```shell script
python test.py config/pan/pan_r18_ic15.py checkpoints/pan_r18_ic15/checkpoint.pth.tar --report_speed
```

获得图片集**detector.zip**,获得的**test_null.json**

<h3 id='Baselinedetector'>Baseline检测</h3>

### **运行环境**
```shell script
CPU/GPU
```
此外，我们使用C++包加速对象检测任务的预测速度。要编译此程序包，请在detector/postprocess文件夹中运行以下命令：
```shell script
make
```
**经过验证，可以选择不使用C++加速包进行加速**

### **训练模型**
将下载的文件放入官方数据文件夹后，请在detector/config/resnet50.yaml中设置路径。然后运行以下命令：
```shell script
python detector/train.py
```
学习过程结束后，经过培训的模型将保存在：/path/to/menu data/checkpoints/detector/。


><h2 id='recognizer'>识别</h2>
<h3 id='Baselinerecognizer'>Baseline识别</h3>

使用[CRNN(卷积递归神经网络, Convolutional Recurrent Neural Network)](https://arxiv.org/pdf/1507.05717.pdf),作为识别网络。

### **推荐环境**
```shell script
tensorflow==1.14.0
opencv-python==4.1.0.25
bidict==0.19.0
yacs==0.1.8
Polygon==3.0.9
pyclipper==1.2.1
python3
```

### **训练识别模型**
首先进行预处理工作
```shell script
python recognizer/tools/extract_train_data.py \ 
    --save_train_image_path /path/to/tmp_data/recognizer_images \
    --save_train_txt_path  /path/to/tmp_data/recognizer_txts \
    --train_image_common_root_path /path/to/official_data/train_image_common \
    --common_label_json_file /path/to/official_data/train_label_common.json\
    --train_image_special_root_path /path/to/official_data/train_image_special \
    --special_label_json_file /path/to/official_data/train_label_special.json
```
这一步是将图片切割为小图片，执行时请注意路径问题。
然后运行：
```shell script
python recognizer/tools/from_text_to_label.py \
    --src_train_file_path /path/to/tmp_data/recognizer_txts/train.txt \
    --dst_train_file_path /path/to/tmp_data/recognizer_txts/real_train.txt\
    --dictionary_file_path recognizer/tools/dictionary/chars.txt
```
这一步将官方数据提取为标签，存储在tmp_data文件夹中。在这里使用chars.txt来存储所有字符。如果更改参数字典文件路径，请同时更改recognizer/tools/config.py中的路径参数。

执行下述命令运行训练：
```shell script
python recognizer/train.py \
    --model_save_dir /path/to/checkpoints/recognizer \
    --log_dir /path/to/output/recognizer_log \
    --image_dir /path/to/tmp_data/recognizer_images \
    --txt_dir /path/to/tmp_data/recognizer_txts
```
目标检测模型的输出将存储在/path/to/menu data/output/detector_test_output中。检测器将生成一个空json文件test_null.json。这些文件将用作文本识别的输入。

将detector生成的test_null.json复制到生成的test_null.json中，运行以下命令：
```shell script
python recognizer/predict.py \
    --char_path recognizer/tools/dictionary/chars.txt \
    --model_path /path/to/checkpoints/recognizer/*.h5 \
    --null_json_path /path/to/output/test_null.json \
    --test_image_path /path/to/output/detector_test_output \
    --submission_path /path/to/output/test_submission.json
```
生成的test_submission.json可以作为比赛输出。

<h3 id='vedastr'>vedastr识别</h3>

vedastr是一个基于PyTorch的开源场景文本识别工具箱。它的设计是灵活的为了支持场景文本识别任务的快速实现和评估。具体介绍可以查看[vedastr](https://www.github.com/Media-Smart/vedastr/)。  
vedastr支持几种流行场景文本识别框架，例如[CRNN](https://arxiv.org/abs/1507.05717)，[TPS-ResNet-BiLSTM-Attention](https://github.com/clovaai/deep-text-recognition-benchmark)，[Transformer](https://arxiv.org/pdf/1910.04396v1.pdf)等

### **推荐环境**
```shell script
Linux
Python 3.6+
PyTorch 1.4.0 or higher
CUDA 9.0 or higher
```

### **准备数据**
vedastr使用的是LMDB格式作为数据集。LMDB全称为 Lightning Memory-Mapped Database，就是快速内存映射型数据库，LMDB使用内存映射文件，可以提供更好的输入/输出性能，对于用于神经网络的大型数据集(比如 ImageNet)，可以将其存储在 LMDB 中。生成数据集的方式可以参考[ASTER: Attentional Scene Text Recognizer with Flexible Rectification](www.github.com/ayumiymk/aster.pytorch#data-preparation)中的
lib/tools/create_svtp_lmdb.py

### **训练识别模型**
首先进行预处理工作

```shell
cd ${vedastr_root}
mkdir ${vedastr_root}/data
```

执行下述命令运行训练：
```shell
# train using GPUs with gpu_id 0, 1, 2, 3
python tools/train.py configs/tps_resnet_bilstm_attn.py "0, 1, 2, 3" 
```

执行下述命令进行测试：
```shell
# test using GPUs with gpu_id 0, 1
./tools/dist_test.sh configs/tps_resnet_bilstm_attn.py path/to/checkpoint.pth "0, 1" 
```