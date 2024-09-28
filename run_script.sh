#!/bin/bash

# 切换到指定目录
cd /home/dongsheng/Code/ECNU/RAG_Critic

export PATH=$PATH:/home/dongsheng/anaconda3/bin
source /home/dongsheng/anaconda3/bin/activate
conda activate selfrag

# 设置 PYTHONPATH 环境变量
export PYTHONPATH=$PYTHONPATH:/home/dongsheng/Code/ECNU/RAG_Critic

# 运行第一个 Python 脚本
python Retrievel_Frame/VDB_construct.py

# 运行第二个 Python 脚本
python Retrievel_Frame/Recall.py
