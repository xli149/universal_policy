#!/bin/bash
#SBATCH --account=def-skelly
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=8G               # 1. 增加内存
#SBATCH --time=0-12:00         # 建议写全格式 DD-HH:MM

seed=${1:-1}                   # 2. 设置默认值，防止没传参数时报错

# 加载必要模块
module load python/3.11
module load mesa-glu           # 3. 运行 MuJoCo 环境通常需要这个图形库
module load mujoco

# 配置虚拟环境
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

# 安装依赖
pip install --no-index --upgrade pip
pip install imageio
pip install gymnasium
# 确保你的 requirements.txt 里已经包含了 tensorboard
pip install --no-index -r requirements.txt

# 4. 如果集群里没预装 tensorboard，你需要删掉 --no-index 单独装一次（前提是节点能上网）
# 或者在本地下载好 whl 传上来。如果能跑通则忽略这行。
# pip install tensorboard

# 运行程序
# 5. 确保 Python 脚本里处理了 -s 参数，有些脚本用 --seed
python co_evolution_main.py --seed $seed --num_proc 8
