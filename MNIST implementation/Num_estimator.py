import tensorflow as tf

mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()

x_train=tf.keras.utils.normalize(x_train)
x_test=tf.keras.utils.normalize(x_test)

model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))
    
model.compile(optimizer="adam",
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
model.fit(x_train,y_train, epochs=3)
import matplotlib.pyplot as plt

plt.imshow(x_train[1],cmap=plt.cm.binary)
plt.show()
val_loss, val_acc= model.evaluate(x_test,y_test)
print(val_loss, val_acc)
model.save('num_pred')
new_model=tf.keras.models.load_model('num_pred')
predictions=new_model.predict(x_test)
import numpy as np
print(np.argmax(predictions[1]))
plt.imshow(x_test[1],cmap=plt.cm.binary)
plt.show()
