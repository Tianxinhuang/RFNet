# RFNet
The codes for RFNet: Recurrent Forward Network for Dense Point Cloud Completion

## Environment
* TensorFlow 1.13.1
* Cuda 10.0
* Python 3.6.9
* lmdb 0.98  
* tensorpack 0.10.1
* numpy 1.14.5

## Dataset
The dataset can be found in [PCN](https://github.com/wentaoyuan/pcn)

## Usage

1. Compile

```
cd ./tf_ops
bash compile.sh
```

2. Train

```
Python3 vv_recon.py
```
Note that the paths of training data(`trainpath`) and validation data(`valpath`) should be edited according to your setting.

3. Test

```
Python3 recon_test.py
```
The paths of test data(`data_dir`) and lists(`list_path`) should be edited before testing.

## Citation


