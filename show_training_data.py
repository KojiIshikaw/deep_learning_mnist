import tensorflow as tf
import matplotlib.pyplot as plt

# MNISTデータセットのロード
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 訓練データからランダムに10枚の画像を表示
plt.figure(figsize=(15, 3))
for i in range(10):
    plt.subplot(1, 10, i+1)
    j = i + 100
    plt.imshow(x_train[j], cmap='gray')
    plt.title(f"Label: {y_train[j]}")
    plt.axis('off')
plt.show()
