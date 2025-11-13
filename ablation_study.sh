#!/bin/bash

# 确保结果目录和日志目录存在
mkdir -p ablation_results
mkdir -p ablation_logs

# 日志文件路径
MAIN_LOG="ablation_logs/main.log"

# 初始化主日志
echo "[$(date +'%Y-%m-%d %H:%M:%S')] 开始消融实验..." > "$MAIN_LOG"
echo "实验环境: $(uname -a)" >> "$MAIN_LOG"
echo "Python版本: $(python --version 2>&1)" >> "$MAIN_LOG"
echo "PyTorch版本: $(python -c 'import torch; print(torch.__version__)' 2>&1)" >> "$MAIN_LOG"
echo "----------------------------------------" >> "$MAIN_LOG"

# 运行实验并记录日志的函数
run_experiment() {
    local experiment_name="$1"
    local log_file="$2"
    shift 2
    local cmd="$@"

    echo "[$(date +'%Y-%m-%d %H:%M:%S')] 开始运行: $experiment_name" | tee -a "$MAIN_LOG"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] 命令: $cmd" >> "$log_file"

    # 执行命令并记录详细日志
    eval "$cmd" >> "$log_file" 2>&1
    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] $experiment_name 运行成功" | tee -a "$MAIN_LOG"
    else
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] $experiment_name 运行失败 (错误码: $exit_code)，详见日志: $log_file" | tee -a "$MAIN_LOG"
    fi
    echo "----------------------------------------" >> "$MAIN_LOG"
    return $exit_code
}
# 1. 完整模型（基准）：启用有序损失
run_experiment "完整模型（基准）" "ablation_logs/full_model.log" \
"python 消融实验.py \
--mode train_test \
--use_global_attention \
--use_gated_fusion \
--use_ordered_loss \
--loss_type ordered \
--data_path data/Reddit/reddit_clean.pkl \
--save_results \
--results_path ablation_results/full_model.csv"

# 2. w/o Global Attention：启用有序损失
run_experiment "w/o Global Attention" "ablation_logs/without_global_attention.log" \
"python 消融实验.py \
--mode train_test \
--use_gated_fusion \
--use_ordered_loss \
--loss_type ordered \
--data_path data/Reddit/reddit_clean.pkl \
--save_results \
--results_path ablation_results/without_global_attention.csv"

# 3. w/o Gated Fusion：启用有序损失
run_experiment "w/o Gated Fusion" "ablation_logs/without_gated_fusion.log" \
"python 消融实验.py \
--mode train_test \
--use_global_attention \
--use_ordered_loss \
--loss_type ordered \
--data_path data/Reddit/reddit_clean.pkl \
--save_results \
--results_path ablation_results/without_gated_fusion.csv"

# 4. w/o Ordered Loss：使用交叉熵损失
run_experiment "w/o Ordered Loss" "ablation_logs/without_ordered_loss.log" \
"python 消融实验.py \
--mode train_test \
--use_global_attention \
--use_gated_fusion \
--loss_type cross_entropy \
--data_path data/Reddit/reddit_clean.pkl \
--save_results \
--results_path ablation_results/without_ordered_loss.csv"

# 5. Word-level Only：启用有序损失
run_experiment "Word-level Only" "ablation_logs/word_level_only.log" \
"python 消融实验.py \
--mode train_test \
--use_ordered_loss \
--loss_type ordered \
--word_level_only \
--data_path data/Reddit/reddit_clean.pkl \
--save_results \
--results_path ablation_results/word_level_only.csv"
# 汇总结果并记录日志
echo "[$(date +'%Y-%m-%d %H:%M:%S')] 开始汇总结果..." | tee -a "$MAIN_LOG"
python - <<END >> "$MAIN_LOG" 2>&1
import pandas as pd
import os

result_files = {
    "Full HMULSES-DST": "ablation_results/full_model.csv",
    "w/o Global Attention": "ablation_results/without_global_attention.csv",
    "w/o Gated Fusion": "ablation_results/without_gated_fusion.csv",
    "w/o Ordered Loss": "ablation_results/without_ordered_loss.csv",
    "Word-level Only": "ablation_results/word_level_only.csv"
}

summary = []
for model_name, file_path in result_files.items():
    if os.path.exists(file_path):
        print(f"找到结果文件: {file_path}")
        df = pd.read_csv(file_path)
        metrics = dict(zip(df['metric'], df['value']))
        summary.append({
            "Model Variant": model_name,
            "Acc": round(metrics['accuracy'], 4),
            "FS": round(metrics['FScore'], 4),
            "Macro-F1": round(metrics['macro_f1'], 4)
        })
    else:
        print(f"警告：未找到 {model_name} 的结果文件 {file_path}")
        summary.append({
            "Model Variant": model_name,
            "Acc": "缺失",
            "FS": "缺失",
            "Macro-F1": "缺失"
        })

summary_df = pd.DataFrame(summary)
summary_df.to_csv("ablation_results/summary.csv", index=False)
print("\n消融实验汇总结果：")
print(summary_df.to_string(index=False))
END

echo "[$(date +'%Y-%m-%d %H:%M:%S')] 所有实验完成" | tee -a "$MAIN_LOG"
echo "汇总结果已保存至 ablation_results/summary.csv"
echo "详细日志请查看 ablation_logs 目录下的文件"