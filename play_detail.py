import tensorflow as tf
from tensorflow.keras.models import load_model, Model
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# 1. モデルのロードとデータの準備
# ------------------------------

# 訓練済みモデルのロード
model = load_model('mnist_digit_recognition.h5')

# MNISTデータセットのロード
mnist = tf.keras.datasets.mnist
(_, _), (x_test, y_test) = mnist.load_data()

# データの前処理
x_test = x_test / 255.0  # 正規化

# ------------------------------
# 2. 特定のテストサンプルの選択
# ------------------------------

sample_index = 0  # 任意のインデックスに変更可能
x_sample = x_test[sample_index]
y_true = y_test[sample_index]

# 入力データの表示
plt.imshow(x_sample, cmap='gray')
plt.title(f"True Label: {y_true}")
plt.axis('off')
plt.show()

# ------------------------------
# 3. 中間層のアクティベーションの取得
# ------------------------------

layer_name = 'hidden_layer'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)

intermediate_output = intermediate_layer_model.predict(np.expand_dims(x_sample, axis=0))

print(f"\n中間層（{layer_name}）のアクティベーションの形状: {intermediate_output.shape}")
print(f"中間層（{layer_name}）のアクティベーションの値:\n{intermediate_output}")

# ------------------------------
# 4. モデル全体の予測の取得
# ------------------------------

prediction = model.predict(np.expand_dims(x_sample, axis=0))
predicted_label = np.argmax(prediction)

print(f"\nモデルの予測ラベル: {predicted_label}")
print(f"予測確率: {prediction}")

# ------------------------------
# 5. 手動で計算を実装
# ------------------------------

# 重みとバイアスの取得
hidden_layer_weights, hidden_layer_biases = model.get_layer('hidden_layer').get_weights()
output_layer_weights, output_layer_biases = model.get_layer('dense_1').get_weights()

# 入力データのフラット化
x_flatten = x_sample.flatten()  # (784,)

print(f"\n入力データ（1次元）の形状: {x_flatten.shape}")
print(f"入力データの値:\n{x_flatten}")

# 隠れ層の計算
z_hidden = np.dot(x_flatten, hidden_layer_weights) + hidden_layer_biases
print(f"\n隠れ層の線形結合結果 (z):\n{z_hidden}")

# ReLU活性化関数の適用
a_hidden = np.maximum(z_hidden, 0)
print(f"\n隠れ層のアクティベーション (ReLU適用後):\n{a_hidden}")

# 出力層の計算
z_output = np.dot(a_hidden, output_layer_weights) + output_layer_biases
print(f"\n出力層の線形結合結果 (z):\n{z_output}")

# ソフトマックス関数の適用
def softmax(x):
    e_x = np.exp(x - np.max(x))  # 安定性のために最大値を引く
    return e_x / e_x.sum()

a_output = softmax(z_output)
print(f"\n出力層のアクティベーション (Softmax適用後):\n{a_output}")

# 予測ラベルの取得
predicted_label_manual = np.argmax(a_output)
print(f"\n手動計算による予測ラベル: {predicted_label_manual}")
