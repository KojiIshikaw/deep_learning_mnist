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

# モデルの概要を表示
model.summary()

# 各レイヤーの重みとバイアスを表示
for layer in model.layers:
    weights = layer.get_weights()
    if weights:  # レイヤーが重みとバイアスを持っている場合
        w, b = weights
        print(f"Layer: {layer.name}")
        print(f"Weights shape: {w.shape}")
        print(f"Biases shape: {b.shape}")
        print("Weights:")
        print(w)
        print("Biases:")
        print(b)
        print("-" * 50)
    else:  # レイヤーが重みとバイアスを持っていない場合
        print(f"Layer: {layer.name} has no weights.")
        print("-" * 50)

# ------------------------------
# 2. 特定のテストサンプルの選択
# ------------------------------

sample_index = 0  # 任意のインデックスに変更可能
x_sample = x_test[sample_index]
y_true = y_test[sample_index]

# ------------------------------
# 3. 中間層のアクティベーションの取得
# ------------------------------

layer_name = 'hidden_layer'  # ここを適切なレイヤー名に変更
intermediate_layer_model = Model(inputs=model.inputs,  # 修正: model.input → model.inputs
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
hidden_layer_weights, hidden_layer_biases = model.get_layer(layer_name).get_weights()
output_layer_weights, output_layer_biases = model.get_layer('dense').get_weights()

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

# ------------------------------
# 6. テストサンプルの表示（参考）
# ------------------------------

# 入力データの表示
plt.imshow(x_sample, cmap='gray')
plt.title(f"True Label: {y_true}")
plt.axis('off')
plt.show()