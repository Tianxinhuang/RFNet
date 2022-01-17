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
The adopted dataset can be found in [PCN](https://github.com/wentaoyuan/pcn).

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
The qualitative results should be 
![image](https://github.com/Tianxinhuang/RFNet/blob/master/quali.png)
The quantitative results on the Known categories of ShapeNet in PCN would be
![image](https://github.com/Tianxinhuang/RFNet/blob/master/quan.png)

## Citation
If you find our work useful for your research, please cite:
```
@inproceedings{huang2021rfnet,
  title={RFNet: Recurrent Forward Network for Dense Point Cloud Completion},
  author={Huang, Tianxin and Zou, Hao and Cui, Jinhao and Yang, Xuemeng and Wang, Mengmeng and Zhao, Xiangrui and Zhang, Jiangning and Yuan, Yi and Xu, Yifan and Liu, Yong},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={12508--12517},
  year={2021}
}
```
