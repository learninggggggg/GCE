import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import time
import random
import argparse
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from ordered_set import OrderedSet
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

# 自定义模块
from utils.order_loss import loss_function, gr_metrics
from models.model import GRU_CNN_Attention as HMULSES_DST
from utils.dataset import TextDataset, collate_fn  # 可复用 utils 中的 dataset


# ========================
# 工具函数
# ========================

def set_seed(seed=2024):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def read_reddit_data(file_path='data/reddit_clean.pkl'):
    """读取 Reddit 数据"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件未找到: {file_path}")
    print(f"正在读取数据: {file_path}")
    reddit_data = pd.read_pickle(file_path)
    labels = [user['label'] for user in reddit_data]
    users = [user['user'] for user in reddit_data]
    return reddit_data, labels, users


from collections import Counter

def get_vocabulary(reddit_data, max_vocab_size=5000):
    """构建词汇表，限制最大词汇量"""
    word_counts = Counter()
    for user_data in reddit_data:
        for post in user_data['subreddit']:
            words = post.strip().split()
            word_counts.update(words)

    # 选取出现频率最高的词
    # 至少保留 <PAD> 和 <UNK>
    common_words = [word for word, count in word_counts.most_common(max_vocab_size - 2)]

    vocab = ["<PAD>", "<UNK>"] + common_words
    words_id = {word: idx for idx, word in enumerate(vocab)}
    print(f"词汇表大小: {len(vocab)}")
    return vocab, words_id

# ========================
# 早停机制
# ========================

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_fs_max = -np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_fs, model):
        score = val_fs
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_fs, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"早停计数: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_fs, model)
            self.counter = 0

    def save_checkpoint(self, val_fs, model):
        if self.verbose:
            print(f"FScore 提升 ({self.val_fs_max:.6f} --> {val_fs:.6f})，保存模型...")
        torch.save({
            'model_state_dict': model.state_dict(),
        }, self.path)
        self.val_fs_max = val_fs


# ========================
# 训练与验证函数
# ========================

def train_epoch(model, dataloader, optimizer, device, args):
    model.train()
    total_loss = 0.0
    out_result = []
    label_result = []

    for batch in tqdm(dataloader, desc="训练中"):
        text, label, text_masks, post_masks = batch
        text = text.to(device)
        label = label.to(device)
        text_masks = text_masks.to(device)
        post_masks = post_masks.to(device)

        optimizer.zero_grad()
        output = model(text, text_masks, post_masks)
        loss = loss_function(output, label, loss_type="ordered", expt_type=args.class_num, scale=1.8)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        out_result.extend(predicted.cpu().numpy().tolist())
        label_result.extend(label.cpu().numpy().tolist())

    acc = accuracy_score(label_result, out_result)
    f1 = f1_score(label_result, out_result, average='macro')
    GP, GR, FS, _ = gr_metrics(out_result, label_result)
    return total_loss / len(dataloader), acc, GP, GR, FS


def eval_epoch(model, dataloader, device, args):
    model.eval()
    total_loss = 0.0
    out_result = []
    label_result = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="验证中"):
            text, label, text_masks, post_masks = batch
            text = text.to(device)
            label = label.to(device)
            text_masks = text_masks.to(device)
            post_masks = post_masks.to(device)

            output = model(text, text_masks, post_masks)
            loss = loss_function(output, label, loss_type="ordered", expt_type=args.class_num, scale=1.8)
            total_loss += loss.item()

            _, predicted = torch.max(output.data, 1)
            out_result.extend(predicted.cpu().numpy().tolist())
            label_result.extend(label.cpu().numpy().tolist())

    acc = accuracy_score(label_result, out_result)
    f1 = f1_score(label_result, out_result, average='macro')
    GP, GR, FS, _ = gr_metrics(out_result, label_result)
    return total_loss / len(dataloader), acc, GP, GR, FS


# ========================
# 测试函数
# ========================

def test_model(model, dataloader, device):
    model.eval()
    out_result = []
    label_result = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="测试中"):
            text, label, text_masks, post_masks = batch
            text = text.to(device)
            label = label.to(device)
            text_masks = text_masks.to(device)
            post_masks = post_masks.to(device)

            output = model(text, text_masks, post_masks)
            _, predicted = torch.max(output.data, 1)

            out_result.extend(predicted.cpu().numpy().tolist())
            label_result.extend(label.cpu().numpy().tolist())

    # 计算指标
    acc = accuracy_score(label_result, out_result)
    f1 = f1_score(label_result, out_result, average='macro')
    GP, GR, FS, OE = gr_metrics(out_result, label_result)  # 使用原始定义

    return acc, GP, GR, FS, OE, out_result, label_result  # ✅ 返回预测和标签用于后续 F1 计算


# ========================
# 参数解析
# ========================

def parse_args():
    parser = argparse.ArgumentParser(description='HMULSES-DST: 训练与测试脚本')
    # 通用参数
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--gru_size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--class_num", type=int, default=5)
    parser.add_argument("--patience", type=int, default=10, help="早停耐心值")


    # 数据与路径
    parser.add_argument("--data_path", type=str, default='data/Reddit/reddit_clean.pkl', help='数据路径')
    parser.add_argument("--save_path", type=str, default='./models/HMULSES-DST.pth', help='最佳模型保存路径')
    parser.add_argument("--model_path", type=str, default='./models/HMULSES-DST.pth', help='测试时加载的模型路径')

    # 控制流程
    parser.add_argument("--mode", type=str, choices=['train', 'test', 'train_test'], default='train_test',
                        help="运行模式：仅训练 / 仅测试 / 训练+测试")
    parser.add_argument("--save_results", action='store_true', help="是否保存测试结果")
    parser.add_argument("--results_path", type=str, default='./test_results.csv', help="测试结果保存路径")

    return parser.parse_args()


# ========================
# 主函数
# ========================
def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('models', exist_ok=True)
    os.makedirs(os.path.dirname(args.results_path) if os.path.dirname(args.results_path) else '.', exist_ok=True)

    print(f"[{time.asctime()}] 开始运行...")
    print(f"设备: {device}")
    print(f"运行模式: {args.mode}")
    print(f"参数: {args}")

    # 读取数据
    print("读取 Reddit 数据...")
    reddit_data, labels, users = read_reddit_data(args.data_path)
    print(f"总样本数: {len(labels)}")

    # 构建词汇表
    print("构建词汇表...")
    vocab, words_id = get_vocabulary(reddit_data)

    # 交叉验证模式仅支持训练+测试或仅训练
    if args.mode == 'test':
        # 对于纯测试模式，仍使用原有划分方式
        print("纯测试模式，使用8:1:1划分数据集...")
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            users, labels, stratify=labels, test_size=0.2, random_state=args.seed
        )
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            test_texts, test_labels, stratify=test_labels, test_size=0.5, random_state=args.seed
        )

        # 创建数据集和加载器
        test_dataset = TextDataset(test_texts, test_labels, words_id, reddit_data, max_posts=50, max_words_per_post=100)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 collate_fn=collate_fn, num_workers=0)
    else:
        # 初始化K折交叉验证
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
        fold_results = []  # 存储每折的结果

        # 遍历每一折
        for fold, (train_idx, val_idx) in enumerate(kf.split(users), 1):
            print(f"\n===== 第 {fold} 折交叉验证 =====")

            # 根据索引划分训练集和验证集
            train_texts = [users[i] for i in train_idx]
            val_texts = [users[i] for i in val_idx]
            train_labels = [labels[i] for i in train_idx]
            val_labels = [labels[i] for i in val_idx]

            # 创建数据集
            train_dataset = TextDataset(train_texts, train_labels, words_id, reddit_data,
                                        max_posts=50, max_words_per_post=100)
            val_dataset = TextDataset(val_texts, val_labels, words_id, reddit_data,
                                      max_posts=50, max_words_per_post=100)

            # 创建数据加载器
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                      collate_fn=collate_fn, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                    collate_fn=collate_fn, num_workers=0)

            # 初始化模型（每折都重新初始化）
            model = HMULSES_DST(args=args, words_id=words_id, vocab_size=len(vocab), device=device).to(device)

            # 训练当前折
            print(f"第 {fold} 折开始训练...")
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
            # 每折保存不同的模型路径
            fold_save_path = args.save_path.replace('.pth', f'_fold{fold}.pth')
            early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=fold_save_path)

            for epoch in range(args.epochs):
                print(f"\nEpoch [{epoch + 1}/{args.epochs}]")

                # 训练
                train_loss, train_acc, train_GP, train_GR, train_FS = train_epoch(
                    model, train_loader, optimizer, device, args)
                print(
                    f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | GP: {train_GP:.4f} | GR: {train_GR:.4f} | FS: {train_FS:.4f}")

                # 验证
                val_loss, val_acc, val_GP, val_GR, val_FS = eval_epoch(
                    model, val_loader, device, args)
                print(
                    f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | GP: {val_GP:.4f} | GR: {val_GR:.4f} | FS: {val_FS:.4f}")

                # 调整学习率
                scheduler.step()

                # 早停
                early_stopping(val_FS, model)
                if early_stopping.early_stop:
                    print("早停触发，停止当前折训练")
                    break

            # 记录当前折的最佳结果
            fold_results.append({
                'fold': fold,
                'best_val_fs': early_stopping.val_fs_max,
                'model_path': fold_save_path
            })
            print(f"第 {fold} 折训练完成，最佳FScore: {early_stopping.val_fs_max:.4f}")

        # 打印交叉验证汇总结果
        print("\n===== 交叉验证汇总 =====")
        avg_fs = sum(res['best_val_fs'] for res in fold_results) / len(fold_results)
        for res in fold_results:
            print(f"第 {res['fold']} 折: 最佳FScore = {res['best_val_fs']:.4f}")
        print(f"平均FScore: {avg_fs:.4f}")

    # 测试模式（使用交叉验证中表现最好的模型）
    if args.mode in ['test', 'train_test'] and args.mode != 'test':  # 排除纯测试模式的重复处理
        # 选择最佳折的模型
        best_fold = max(fold_results, key=lambda x: x['best_val_fs'])
        print(f"\n选择第 {best_fold['fold']} 折的模型进行测试 (最佳FScore: {best_fold['best_val_fs']:.4f})")

        # 划分测试集（使用原始数据的20%作为测试集）
        _, test_texts, _, test_labels = train_test_split(
            users, labels, stratify=labels, test_size=0.2, random_state=args.seed
        )
        test_dataset = TextDataset(test_texts, test_labels, words_id, reddit_data,
                                   max_posts=50, max_words_per_post=100)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 collate_fn=collate_fn, num_workers=0)

        # 加载最佳模型
        model = HMULSES_DST(args=args, words_id=words_id, vocab_size=len(vocab), device=device).to(device)
        checkpoint = torch.load(best_fold['model_path'], map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ 加载最佳模型: {best_fold['model_path']}")

        # 执行测试
        test_acc, test_GP, test_GR, test_FS, test_OE, model_predictions, test_labels_list = test_model(
            model, test_loader, device)

        print("\n" + "=" * 50)
        print("最终测试结果:")
        print("=" * 50)
        print(f"Accuracy (Acc):        {test_acc:.4f}")
        print(f"General Precision (GP): {test_GP:.4f}")
        print(f"General Recall (GR):    {test_GR:.4f}")
        print(f"FScore (FS):            {test_FS:.4f}")
        print(f"Overestimation Error (OE): {test_OE:.4f}")
        print(f"Macro-F1:               {f1_score(test_labels_list, model_predictions, average='macro'):.4f}")
        print("=" * 50)

        # 保存结果
        if args.save_results:
            results = pd.DataFrame({
                'metric': ['accuracy', 'GP', 'GR', 'FScore', 'OE', 'macro_f1'],
                'value': [test_acc, test_GP, test_GR, test_FS, test_OE,
                          f1_score(test_labels_list, model_predictions, average='macro')]
            })
            results.to_csv(args.results_path, index=False)
            print(f"测试结果已保存至: {args.results_path}")

    print("所有任务完成。")


if __name__ == '__main__':
    main()