import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import seaborn as sns

# 设置数据目录路径
data_dir = 'C:\\Users\\X\\Desktop\\AF\\data'

# 高通滤波器函数
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def highpass_filter(data, cutoff=1, fs=2048, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# 低通滤波器函数
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff=50, fs=2048, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# 读取和处理单个CSV文件
def load_and_preprocess_file(file_path):
    try:
        data = pd.read_csv(file_path, header=None).values
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, None

    # 对数据进行高通和低通滤波
    filtered_data = np.apply_along_axis(highpass_filter, 0, data)
    filtered_data = np.apply_along_axis(lowpass_filter, 0, filtered_data)

    # 提取标签
    labels = np.array([0] * 52 + [1] * 52)

    return filtered_data, labels

# 获取所有CSV文件
file_list = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

accuracies = []
f1_scores = []
confusion_matrices = []

for file_name in file_list:
    file_path = os.path.join(data_dir, file_name)
    X, y = load_and_preprocess_file(file_path)

    if X is None or y is None:
        continue

    # 检查数据处理结果
    print(f"Processed file: {file_name}")
    print(f"First 5 data points: {X[:5]}")
    print(f"First 5 labels: {y[:5]}")

    # 数据标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 数据归一化
    normalizer = MinMaxScaler()
    X = normalizer.fit_transform(X)

    # 重塑数据以适应Conv1D层
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # 5-fold交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_accuracies = []
    fold_f1_scores = []
    fold_cms = []

    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # 构建CNN-LSTM模型
        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1), kernel_regularizer=l2(0.01)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(50, return_sequences=True, kernel_regularizer=l2(0.01)))
        model.add(Flatten())
        model.add(Dense(100, activation='relu', kernel_regularizer=l2(0.01)))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # 使用EarlyStopping防止过拟合
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

        # 训练模型
        with tf.device('/GPU:0'):  # 确保使用GPU进行训练
            history = model.fit(X_train, y_train, epochs=35, batch_size=32, validation_data=(X_val, y_val), callbacks=[es], verbose=1)

        # 预测
        y_pred = (model.predict(X_val) > 0.5).astype("int32")

        # 计算准确率和F1得分
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        fold_accuracies.append(accuracy)
        fold_f1_scores.append(f1)

        # 计算并保存混淆矩阵
        cm = confusion_matrix(y_val, y_pred)
        fold_cms.append(cm)

    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    mean_f1 = np.mean(fold_f1_scores)
    std_f1 = np.std(fold_f1_scores)
    accuracies.append((file_name, mean_accuracy, std_accuracy))
    f1_scores.append((file_name, mean_f1, std_f1))

    # 保存每个文件的混淆矩阵到一个列表
    avg_cm = np.mean(fold_cms, axis=0).astype(int)
    confusion_matrices.append((file_name, avg_cm))

    # 混淆矩阵可视化
    cm_df = pd.DataFrame(avg_cm, index=['Actual Negative', 'Actual Positive'], columns=['Predicted Negative', 'Predicted Positive'])
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix for {file_name}')
    plt.savefig(f'confusion_matrix_{file_name}.png')
    plt.close()

# 保存所有文件的准确率和F1得分为CSV文件
results_df = pd.DataFrame(accuracies, columns=['File', 'Mean Accuracy', 'Accuracy Std'])
results_df['Mean F1 Score'] = [f1 for _, f1, _ in f1_scores]
results_df['F1 Std'] = [std for _, _, std in f1_scores]
results_df.to_csv('results1.csv', index=False)

# 保存所有文件的混淆矩阵为一个CSV文件
with open('all_confusion_matrices.csv', 'w') as f:
    f.write('File,Actual Negative,Actual Positive,Predicted Negative,Predicted Positive\n')
    for file_name, cm in confusion_matrices:
        f.write(f'{file_name},{cm[0,0]},{cm[0,1]},{cm[1,0]},{cm[1,1]}\n')

# 打印所有文件的准确率和F1得分
for file_name, accuracy, std_acc in accuracies:
    f1, std_f1 = next((f1, std) for f, f1, std in f1_scores if f == file_name)
    print(f"File: {file_name}, Mean Accuracy: {accuracy:.4f} ± {std_acc:.4f}, Mean F1 Score: {f1:.4f} ± {std_f1:.4f}")
