#!/bin/bash

# Variables
split_size=128
mlp_size=1000
mode='CIFAR'
user='master'
rank=0
master_addr='192.168.101.87'
interface='eno2'

# Training with MLP dataset
# if [ $mode == 'MLP' ]
# then
# 	mkdir -p ./MLP_log/split_$split_size/mlp_$mlp_size
# 	tegrastats --interval 1000 --logfile ./MLP_log/split_$split_size/mlp_$mlp_size/$user\_usage.log &
# 	python3 log_bandwidth.py ./MLP_log/split_$split_size/mlp_$mlp_size/$user\_throughput.log &
# 	python3 pipeline_MLP.py --rank=$rank --master_addr=$master_addr --master_port=23456 --world_size=3 --interface=$interface --split_size=$split_size --mlp_size=$mlp_size
# fi

# Training with CIFAR dataset
if [ $mode == 'CIFAR' ]
then
	mkdir -p ./CIFAR_log/split_$split_size
	# sudo tegrastats --interval 1000 --logfile ./CIFAR_log/split_$split_size/$user\_usage.log &
	python3 log_bandwidth.py ./CIFAR_log/split_$split_size/$user\_throughput.log &
    # sudo python3 packet_sniffer.py --filename=$user\_$split_size ----ip1=192.168.101.31 --ip2=192.168.101.21 --ip3=192.168.101.24 &
	python3 vgg16_pipeline_CIFAR.py --rank=$rank --master_addr=$master_addr --master_port=23456 --world_size=3 --interface=$interface --split_size=$split_size
fi