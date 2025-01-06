import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# MNISTデータセットのロード
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# データの前処理
x_train, x_test = x_train / 255.0, x_test / 255.0  # 正規化

# モデルの構築
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),           # 入力層
    layers.Dense(128, activation='relu', name='hidden_layer'),  # 隠れ層に名前を付ける
    layers.Dropout(0.2),                            # ドロップアウト層
    layers.Dense(10, activation='softmax')          # 出力層
])

# モデルのコンパイル
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# モデルの概要表示
model.summary()

# モデルの訓練
history = model.fit(x_train, y_train, epochs=5, 
                    validation_data=(x_test, y_test))

# モデルの評価
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nテスト精度:', test_acc)

# モデルの保存
model.save('mnist_digit_recognition.h5')

# 学習過程の可視化
plt.plot(history.history['accuracy'], label='訓練精度')
plt.plot(history.history['val_accuracy'], label='検証精度')
plt.xlabel('エポック')
plt.ylabel('精度')
plt.legend(loc='lower right')
plt.title('モデルの精度')
plt.show()
