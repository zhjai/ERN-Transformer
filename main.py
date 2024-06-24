from load_data import load_data
from model import Model
from args import Config
# ---------------------------------------------------
import pandas as pd
from collections import Counter
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

config = Config()
train, targets, vocabs_size = load_data(config)  # 加载数据
config.n_vocab = vocabs_size + 1  # 补充词表大小，词表一定要多留出来一个

batch_size = config.batch_size

kf = KFold(n_splits=config.n_splits, shuffle=True, random_state=2021)  # 5折交叉验证

all_train_losses = []
all_val_losses = []
all_train_accuracies = []
all_val_accuracies = []

for fold, (train_idx, test_idx) in enumerate(kf.split(train, targets)):
    print('-' * 15, '>', f'Fold {fold+1}', '<', '-' * 15)
    x_train, x_val = train[train_idx], train[test_idx]
    y_train, y_val = targets[train_idx], targets[test_idx]
    
    M_train = len(x_train)
    M_val = len(x_val)
    if M_train % batch_size == 1:  # 因为模型里面有层标准化，训练中不能出现单条数据，至少为2条
        M_train -= 1
    if M_val % batch_size == 1:
        M_val -= 1
    x_train = torch.from_numpy(x_train).to(torch.long).to(config.device)
    x_val = torch.from_numpy(x_val).to(torch.long).to(config.device)
    y_train = torch.from_numpy(y_train).to(torch.long).to(config.device)
    y_val = torch.from_numpy(y_val).to(torch.long).to(config.device)

    model = Model(config)  # 调用transformer的编码器
    model.to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_func = nn.CrossEntropyLoss()  # 多分类的任务
    model.train()  # 模型中有BN和Droupout一定要添加这个说明
    print('开始迭代....')

    # 打印模型大小
    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model Size: {model_size / 1e6:.2f} million parameters')

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for step in range(config.num_epochs):
        print('step=', step + 1)
        epoch_train_loss, epoch_val_loss = 0, 0
        epoch_train_acc, epoch_val_acc = 0, 0
        L_val = -batch_size
        n_batches = 0

        with tqdm(np.arange(0, M_train, batch_size), desc='Training...') as tbar:
            for index in tbar:
                n_batches += 1
                L = index
                R = min(M_train, index + batch_size)
                L_val += batch_size
                L_val %= M_val
                R_val = min(M_val, L_val + batch_size)
                
                # -----------------训练内容------------------
                train_pre = model(x_train[L:R])  # 喂给 model训练数据 x, 输出预测值
                train_loss = loss_func(train_pre, y_train[L:R])
                val_pre = model(x_val[L_val:R_val])  # 验证集也得分批次，不然数据量太大内存爆炸
                val_loss = loss_func(val_pre, y_val[L_val:R_val])

                # 计算准确率
                train_acc = np.sum(np.argmax(np.array(train_pre.data.cpu()), axis=1) == np.array(y_train[L:R].data.cpu())) / (R - L)
                val_acc = np.sum(np.argmax(np.array(val_pre.data.cpu()), axis=1) == np.array(y_val[L_val:R_val].data.cpu())) / (R_val - L_val)

                # 更新累计损失和准确率
                epoch_train_loss += train_loss.item()
                epoch_val_loss += val_loss.item()
                epoch_train_acc += train_acc
                epoch_val_acc += val_acc

                # -----------------反向传播更新---------------
                optimizer.zero_grad()  # 清空上一步的残余更新参数值
                train_loss.backward()  # 以训练集的误差进行反向传播, 计算参数更新值
                optimizer.step()  # 将参数更新值施加到 net 的 parameters 上

        # 计算平均损失和准确率
        avg_train_loss = epoch_train_loss / n_batches
        avg_val_loss = epoch_val_loss / n_batches
        avg_train_acc = epoch_train_acc / n_batches
        avg_val_acc = epoch_val_acc / n_batches

        # 记录每个epoch的损失和准确率
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(avg_train_acc)
        val_accuracies.append(avg_val_acc)

        print(f'Epoch {step + 1}, Train Loss: {avg_train_loss}, Train Acc: {avg_train_acc}, Val Loss: {avg_val_loss}, Val Acc: {avg_val_acc}')

    # 保存每折的数据到全局列表
    all_train_losses.append(train_losses)
    all_val_losses.append(val_losses)
    all_train_accuracies.append(train_accuracies)
    all_val_accuracies.append(val_accuracies)

    # 删除模型以释放内存
    del model

# 保存训练和验证损失、准确率到CSV文件
train_losses_df = pd.DataFrame(all_train_losses)
val_losses_df = pd.DataFrame(all_val_losses)
train_accuracies_df = pd.DataFrame(all_train_accuracies)
val_accuracies_df = pd.DataFrame(all_val_accuracies)

train_losses_df.to_csv('train_losses.csv', index=False)
val_losses_df.to_csv('val_losses.csv', index=False)
train_accuracies_df.to_csv('train_accuracies.csv', index=False)
val_accuracies_df.to_csv('val_accuracies.csv', index=False)

# 绘制训练曲线并保存
for fold in range(config.n_splits):
    plt.figure(figsize=(12, 8))
    plt.plot(train_losses_df.iloc[fold], label='Train Loss')
    plt.plot(val_losses_df.iloc[fold], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Fold {fold + 1} Loss')
    plt.legend()
    plt.savefig(f'fold_{fold + 1}_loss_curve.png')
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.plot(train_accuracies_df.iloc[fold], label='Train Accuracy')
    plt.plot(val_accuracies_df.iloc[fold], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Fold {fold + 1} Accuracy')
    plt.legend()
    plt.savefig(f'fold_{fold + 1}_accuracy_curve.png')
    plt.close()