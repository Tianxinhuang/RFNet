#!/usr/bin/env bash
nvcc=/usr/local/cuda-9.0/bin/nvcc
cudalib=/usr/local/cuda-9.0/lib64/
TF_INC=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

$nvcc tf_auctionmatch_g.cu  -c -o tf_auctionmatch_g.cu.o -std=c++11  -I $TF_INC -DGOOGLE_CUDA=1\
 -x cu -Xcompiler -fPIC -O2

g++ tf_auctionmatch.cpp tf_auctionmatch_g.cu.o -o tf_auctionmatch_so.so -std=c++11 -shared -fPIC -I $TF_INC \
-I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework -lcudart -L $cudalib -O2
