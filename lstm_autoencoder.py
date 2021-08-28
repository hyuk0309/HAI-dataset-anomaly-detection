# TensorFlow 및 기타 라이브러리 가져오기.
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
from pylab import rcParams

from pathlib import Path

from keras import optimizers, Sequential
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import auc, roc_curve


from numpy.random import seed
seed(7)

SEED = 123  # used to help randomly select the data points
DATA_SPLIT_PCT = 0.2

rcParams['figure.figsize'] = 8, 6
LABELS = ["Normal", "Attack"]


# autoencoder model을 이용한 HAI dataset에서의 이상 탐지 분석.

# HAI dataset 가져오기.
train_dataset_paths = sorted([x for x in Path("./data/train/").glob("*.csv")])
# print(train_dataset_paths)

test_dataset_paths = sorted([x for x in Path("./data/test/").glob("*.csv")])
# print(test_dataset_paths)


def dataframe_from_csv(target):
    return pd.read_csv(target).rename(columns=lambda x: x.strip())


def dataframe_from_csvs(targets):
    return pd.concat([dataframe_from_csv(x) for x in targets])


train_df_raw = dataframe_from_csvs(train_dataset_paths)
test_df_raw = dataframe_from_csvs(test_dataset_paths)


# 사용하지 않는 columns 제거.
train_df = train_df_raw.drop(['time', 'attack_P1', 'attack_P2', 'attack_P3'], axis=1)
test_df = test_df_raw.drop(['time', 'attack_P1', 'attack_P2', 'attack_P3'], axis=1)

# LSTM에 맞게 input data 준비하기.
train_input_X = train_df.loc[:, train_df.columns != 'attack'].values
train_input_y = train_df['attack'].values
n_features = train_input_X.shape[1]

test_input_X = test_df.loc[:, test_df.columns != 'attack'].values
test_input_y = test_df['attack'].values


def temporalize(X, y, lookback):
    output_X = []
    output_y = []
    for i in range(len(X) - lookback - 1):
        t = []
        for j in range(1, lookback + 1):
            t.append(X[[(i + j + 1)], :])
        output_X.append(t)
        output_y.append(y[i + lookback + 1])
    return np.squeeze(np.array(output_X)), np.array(output_y)


lookback = 5
train_X, train_y = temporalize(train_input_X, train_input_y, lookback)
test_X, test_y = temporalize(test_input_X, test_input_y, lookback)

# spilt data.
X_train, X_valid, y_train, y_valid = train_test_split(np.array(train_X),
                                                      np.array(train_y), test_size=DATA_SPLIT_PCT, random_state=SEED)
X_test = np.array(test_X)
y_test = np.array(test_y)

X_train = X_train.reshape(X_train.shape[0], lookback, n_features)
X_valid = X_valid.reshape(X_valid.shape[0], lookback, n_features)
X_test = X_test.reshape(X_test.shape[0], lookback, n_features)


# standardize 하기.
def flatten(X):
    flattened_X = np.empty((X.shape[0], X.shape[2]))
    for i in range(X.shape[0]):
        flattened_X[i] = X[i, (X.shape[1]-1), :]
    return (flattened_X)

def scale(X, scaler):
    for i in range(X.shape[0]):
        X[i, :, :] = scaler.transform(X[i, :, :])
    return X


# train data의 평균과 표본을 이용해 표준화 진행
scaler = StandardScaler().fit(flatten(X_train))
X_train_scaled = scale(X_train, scaler)

X_valid_scaled = scale(X_valid, scaler)
X_test_scaled = scale(X_test, scaler)


# make LSTM Autoencoder.
timesteps = X_train_scaled.shape[1]
n_features = X_train_scaled.shape[2]



lstm_autoencoder = Sequential()
# Encoder.
lstm_autoencoder.add(LSTM(32, activation='relu', input_shape=(timesteps, n_features), return_sequences=True))
lstm_autoencoder.add(LSTM(16, activation='relu', return_sequences=False))
lstm_autoencoder.add(RepeatVector(timesteps))
# Decoder.
lstm_autoencoder.add(LSTM(16, activation='relu', return_sequences=True))
lstm_autoencoder.add(LSTM(32, activation='relu', return_sequences=True))
lstm_autoencoder.add(TimeDistributed(Dense(n_features)))

lstm_autoencoder.summary()

# train the model.
epochs = 5
batch = 64
lr = 0.0001

adam = optimizers.Adam(lr)
lstm_autoencoder.compile(loss='mse', optimizer=adam)

lstm_autoencoder_history = lstm_autoencoder.fit(X_train_scaled, X_train_scaled,
                                                epochs=epochs,
                                                batch_size=batch,
                                                validation_data=(X_valid_scaled, X_valid_scaled),
                                                verbose=2).history

# plotting the change in the loss over the epochs.
plt.plot(lstm_autoencoder_history['loss'], linewidth=2,
         label='Train')
plt.plot(lstm_autoencoder_history['val_loss'], linewidth=2,
         label='Valid')
plt.legend(loc='upper right')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()


# precision_recall_curve.
valid_x_predictions = lstm_autoencoder.predict(X_valid_scaled)
mse = np.mean(np.power(flatten(X_valid_scaled) - flatten(valid_x_predictions), 2), axis=1)

error_df = pd.DataFrame({'Reconstruction_error': mse,
                         'True_class': y_valid.tolist()})
precision_rt, recall_rt, threshold_rt = precision_recall_curve(error_df.True_class,
                                                               error_df.Reconstruction_error)

plt.plot(threshold_rt, precision_rt[1:], label="Precision", linewidth=5)
plt.plot(threshold_rt, recall_rt[1:], label="Recall", linewidth=5)
plt.title('Precision and recall for different threshold values')
plt.xlabel('Threshold')
plt.ylabel('Precision/Recall')
plt.legend()
plt.show()

# estimate the classification. using constant threshold = 0.3
test_x_predictions = lstm_autoencoder.predict(X_test_scaled)
mse = np.mean(np.power(flatten(X_test_scaled) - flatten(test_x_predictions), 2), axis=1)
error_df = pd.DataFrame({'Reconstruction_error': mse,
                         'True_class': y_test.tolist()})

threshold_fixed = 0.3
groups = error_df.groupby('True_class')
fig, ax = plt.subplots()

for name, group in groups:
    ax.plot(group.index, group.Reconstruction_error, marker='o', ms=3.5,
            linestyle='', label="Attack" if name == 1 else "Normal")
ax.hlines(threshold_fixed, ax.get_xlim()[0], ax.get_xlim()[1],
          colors='r', zorder=100, label='Threshold')
ax.legend()
plt.title("Reconstruction error for different classes")
plt.ylabel("Reconstruction error")
plt.xlabel("Data point index")
plt.show()

# Test Accuracy
pred_y = [1 if e > threshold_fixed else 0 for e in error_df.Reconstruction_error.values]

conf_matrix = confusion_matrix(error_df.True_class, pred_y)

plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS,
            annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

# ROC Curve and AUC
false_pos_rate, true_pos_rate, thresholds = roc_curve(error_df.True_class, error_df.Reconstruction_error)
roc_auc = auc(false_pos_rate, true_pos_rate)

plt.plot(false_pos_rate, true_pos_rate, linewidth=5, label='AUC = %0.3f'% roc_auc)
plt.plot([0, 1], [0, 1], linewidth=5)
plt.xlim([-0.01, 1])
plt.ylim([0, 1.01])
plt.legend(loc='lower right')
plt.title('Receiver operating characteristic curve (ROC) ')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()



