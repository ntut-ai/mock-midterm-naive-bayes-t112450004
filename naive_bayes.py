import pytest
import numpy as np
import pandas as pd
import os
from naive_bayes import nb_train, nb_predict

# --- 輔助函數：載入資料 ---

def load_data(filename):
    """
    從 'data' 資料夾載入 CSV 檔案。
    假設 CSV 檔案沒有標頭 (header)。
    """
    # 獲取當前測試檔案所在的目錄
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 組合 'data' 資料夾和檔案的路徑
    data_path = os.path.join(test_dir, 'data', filename)
    
    # 檢查檔案是否存在
    if not os.path.exists(data_path):
        # 如果檔案不存在，跳過這個測試
        pytest.skip(f"資料檔案未找到: {data_path}")
        
    try:
        # 使用 pandas 載入 CSV，假設沒有標頭 (header=None)
        df = pd.read_csv(data_path, header=None)
        
        # 將 DataFrame 轉換為 numpy array，這是 nb_train 所需的格式
        return df.to_numpy()
    except Exception as e:
        # 如果載入失敗，則測試失敗
        pytest.fail(f"無法載入資料檔案 {data_path}: {e}")

# --- 測試案例 1：檢查模型結構 ---

def test_naive_bayes_model():
    """
    測試 nb_train 函數是否產生正確的模型結構。
    (對應 test_naive_bayes_model)
    """
    # 1. 載入訓練資料 (假設檔名為 iris_train.csv)
    # 您的 data 資料夾中必須有這個檔案
    train_data = load_data('iris_train.csv')
    
    # 2. 訓練模型
    model = nb_train(train_data)
    
    # 3. 檢查模型是否為字典
    assert isinstance(model, dict), "模型應該是一個字典 (dict)"
    
    # 4. 檢查模型是否包含所有類別
    # 假設標籤已經是數字 (例如 0, 1, 2)
    labels = train_data[:, -1]
    unique_labels = np.unique(labels)
    assert len(model.keys()) == len(unique_labels), "模型應包含所有類別的參數"
    
    # 5. 檢查第一個類別的參數結構
    first_class_id = list(model.keys())[0]
    class_params = model[first_class_id]
    
    assert 'prior' in class_params, "每個類別應包含 'prior' (先驗機率)"
    assert 'means' in class_params, "每個類別應包含 'means' (平均值)"
    assert 'stds' in class_params, "每個類別應包含 'stds' (標準差)"
    
    # 6. 檢查參數的格式是否正確
    assert isinstance(class_params['prior'], (float, np.floating)), "'prior' 應該是浮點數"
    assert isinstance(class_params['means'], np.ndarray), "'means' 應該是 numpy array"
    assert isinstance(class_params['stds'], np.ndarray), "'stds' 應該是 numpy array"
    
    # 7. 檢查特徵數量是否一致
    n_features = train_data.shape[1] - 1
    assert len(class_params['means']) == n_features, "平均值陣列的長度應等於特徵數量"
    assert len(class_params['stds']) == n_features, "標準差陣列的長度應等於特徵數量"

    # 8. 檢查所有先驗機率總和是否為 1
    total_prior = sum(params['prior'] for params in model.values())
    # 使用 pytest.approx 處理浮點數的比較
    assert total_prior == pytest.approx(1.0), "所有類別的先驗機率總和應為 1.0"

# --- 測試案例 2：檢查分類器準確度 ---

def test_naive_bayes_classifier():
    """
    測試分類器的整體準確度。
    (對應 test_naive_bayes_classifier)
    """
    # 1. 載入訓練和測試資料
    # 您的 data 資料夾中必須有這兩個檔案
    train_data = load_data('iris_train.csv')
    test_data = load_data('iris_test.csv')
    
    if test_data.shape[0] == 0:
        pytest.fail("測試資料為空，無法計算準確度")

    # 2. 訓練模型
    model = nb_train(train_data)
    
    correct_predictions = 0
    total_predictions = test_data.shape[0]
    
    # 3. 遍歷測試資料進行預測
    for row in test_data:
        x_test = row[:-1]  # 特徵 (除了最後一欄)
        y_true = row[-1]   # 真正的標籤 (最後一欄)
        
        # 進行預測
        y_pred = nb_predict(model, x_test)
        
        # 4. 比較預測結果和真實標籤
        # 假設標籤是數字 (int 或 float)
        if int(y_pred) == int(y_true):
            correct_predictions += 1
            
    # 5. 計算準確度
    accuracy = correct_predictions / total_predictions
    
    print(f"\n[測試] 分類器準確度: {accuracy * 100:.2f}%")
    
    # 6. 斷言準確度高於一個門檻值
    # 對於 Iris 資料集，Naive Bayes 應該有不錯的表現 (例如 > 85%)
    assert accuracy > 0.85, f"準確度 {accuracy:.2f} 低於 0.85 的門檻"