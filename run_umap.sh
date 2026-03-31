#!/bin/bash

# 设置Python环境（如果需要的话）
# source /path/to/your/venv/bin/activate

# 创建日志目录
mkdir -p logs

# 获取当前时间作为日志文件名的一部分
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="logs/run_${timestamp}.log"

# 定义数据集数组（仅保留前14个，索引0–13）
merged_dataset=(
    "14"
    "13"
    "12"
    "11"
    "10"
    "9"
    "8"
    "7"
    "6"
    "5"
    "4"
    "3"
    "2"
    "1"
    "0"
)

# 定义数据集数组（仅保留前14个，索引0–13）

# 运行merged数据集
echo "开始运行merged数据集..." | tee -a "$log_file"
for dataset in "${merged_dataset[@]}"; do
    echo "正在运行数据集: $dataset " | tee -a "$log_file"
    python main.py --dataid "$dataset" 
    echo "----------------------------------------" | tee -a "$log_file"
done



echo "所有数据集运行完成！" | tee -a "$log_file" 
