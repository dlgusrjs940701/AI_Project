#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 데이터 로드
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
columns = [
    "mpg",
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "model_year",
    "origin",
]
data = pd.read_csv(url, names=columns, delim_whitespace=True, na_values="?")

# 결측값 처리
data = data.dropna()

# 입력 변수와 출력 변수 분리
X = data.drop("mpg", axis=1)
y = data["mpg"]

# 범주형 변수 처리
X = pd.get_dummies(X, columns=["origin"], prefix="origin")

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 데이터 스케일링
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[2]:


# 신경망 모델 정의
model = Sequential(
    [
        Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
        Dense(32, activation="relu"),
        Dense(1),  # 출력층 (연비는 실수 값)
    ]
)

# 모델 컴파일
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# 모델 학습
history = model.fit(
    X_train, y_train, validation_split=0.2, epochs=100, batch_size=32, verbose=1
)


# In[3]:


# 평가
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# 예측
y_pred = model.predict(X_test)

# 결과 시각화
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.xlabel("Actual MPG")
plt.ylabel("Predicted MPG")
plt.title("Actual vs Predicted MPG")
plt.show()


# In[ ]:




