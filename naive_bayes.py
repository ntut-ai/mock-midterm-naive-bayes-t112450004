
import numpy as np
import pandas as pd
import os
import zipfile
import argparse
from collections import Counter

# --- Helper Function for Label Encoding ---

def str_column_to_int(dataset, column):
    """
    Convert a string column to integer codes.

    Args:
        dataset: List of lists representing the dataset.
        column: Index of the column to convert.

    Returns:
        A dictionary mapping string labels to integer codes.
    """
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup

# --- Naive Bayes Core Functions (adapted for nb_train and nb_predict) ---

def _calculate_likelihood(x, mean, std):
    """
    計算給定特徵值 x 在指定類別下 (由 mean 和 std 定義) 的機率密度 (使用高斯分佈)。

    Args:
        x: 單一特徵值 (float)
        mean: 該特徵在某類別下的平均值 (float)
        std: 該特徵在某類別下的標準差 (float)

    Returns:
        該特徵值在指定類別下的機率密度 (float)。
    """
    # 這裡實現高斯機率密度函數 (PDF)
    # PDF(x, mean, std) = 1 / (sqrt(2 * pi) * std) * exp(-((x - mean)^2 / (2 * std^2)))
    epsilon = 1e-6 # 一個小常數來避免除以零或 log(0)
    std = np.maximum(std, epsilon) # 替換掉小於 epsilon 的標準差


    # 計算機率密度
    exponent = np.exp(-((x - mean)**2) / (2 * std**2))
    denominator = np.sqrt(2 * np.pi) * std
    likelihood = exponent / denominator

    return likelihood

def nb_train(train_data):
    """
    根據訓練資料計算 Naive Bayes 模型參數。

    Args:
        train_data: 訓練資料 (numpy array)，最後一欄是標籤。

    Returns:
        一個字典，包含模型參數 (每個類別的先驗機率，以及每個特徵在每個類別下的平均值和標準差)。
    """
    # 將 numpy array 轉換回 list of lists，以便使用 str_column_to_int
    train_data_list = train_data.tolist()

    # 對標籤欄進行編碼 (如果需要，假設標籤在最後一欄)
    # 注意：nb_test.py 中似乎已經處理了標籤到 ID 的轉換，
    # 但為了完整性，這裡保留 str_column_to_int 的用法。
    # 如果您的原始資料標籤已經是數字，可以跳過此步驟。
    # 這裡假設 train_data_list 的最後一欄是字串標籤
    if isinstance(train_data_list[0][-1], str):
        label_id_dict = str_column_to_int(train_data_list, len(train_data_list[0]) - 1)
    else:
        # 如果標籤已經是數字，建立一個簡單的 ID 到標籤的對應 (如果需要)
        unique_labels = np.unique(np.array(train_data_list)[:, -1])
        label_id_dict = {label: i for i, label in enumerate(unique_labels)}


    # 將轉換後的資料轉回 numpy array (如果需要)
    # train_data_processed = np.array(train_data_list) # 這行可能不需要，取決於後續如何使用

    # 從處理後的資料中分離特徵和標籤
    # 假設處理後的標籤在最後一欄且已轉換為數字 ID
    X = np.array([row[:-1] for row in train_data_list]) # 特徵
    y = np.array([row[-1] for row in train_data_list]) # 標籤 (數字 ID)


    classes = np.unique(y)
    n_features = X.shape[1]
    n_classes = len(classes)

    # 初始化儲存參數的字典
    model = {}

    # 計算每個類別的參數
    for i, c_id in enumerate(classes):
        X_c = X[y == c_id]

        # 計算先驗機率 P(class)
        prior = len(X_c) / len(X)

        # 計算每個特徵在該類別下的平均值和標準差
        means = X_c.mean(axis=0)
        stds = X_c.std(axis=0)

        # 儲存參數
        model[c_id] = {
            'prior': prior,
            'means': means,
            'stds': stds
        }

    return model

def nb_predict(model, x):
    """
    使用 Naive Bayes 模型預測單一樣本的類別。

    Args:
        model: 訓練好的模型參數字典。
        x: 單一樣本的特徵向量 (numpy array of shape (n_features,))。

    Returns:
        預測的類別標籤 ID (整數).
    """
    posteriors_log = []

    # 計算每個類別的後驗機率的 log (P(class | x))
    # log(P(class | x)) ~ log(P(x | class)) + log(P(class))
    # log(P(x | class)) = sum(log(P(xi | class)))

    for class_id, params in model.items():
        # 計算先驗機率的 log
        prior_log = np.log(params['prior'])

        # 計算每個特徵在該類別下的 log 機率密度並加總
        likelihood_log_sum = np.sum(np.log(_calculate_likelihood(x, params['means'], params['stds'])))

        # 計算後驗機率的 log (未歸一化)
        posterior_log = prior_log + likelihood_log_sum
        posteriors_log.append((class_id, posterior_log))

    # 找到具有最高後驗機率 log 值的類別 ID
    best_class_id = max(posteriors_log, key=lambda item: item[1])[0]

    return best_class_id
