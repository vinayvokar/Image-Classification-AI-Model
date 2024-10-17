Image Classificayion mode (dog Vs cat)

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kagle/
     

!kaggle datasets download -d salader/dogs-vs-cats
     
Dataset URL: https://www.kaggle.com/datasets/salader/dogs-vs-cats
License(s): unknown
Downloading dogs-vs-cats.zip to /content
 98% 1.05G/1.06G [00:10<00:00, 207MB/s]
100% 1.06G/1.06G [00:10<00:00, 107MB/s]

import zipfile
zip_ref=zipfile.ZipFile('/content/dogs-vs-cats.zip')
zip_ref.extractall('/content')
zip_ref.close()
     

import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,BatchNormalization,Dropout
     

keras.utils.image_dataset_from_directory(
    directory='/content/train',
    labels="inferred",
    label_mode="int",
    class_names=None,
    batch_size=32,
    image_size=(256, 256),
)
     
Found 20000 files belonging to 2 classes.
<_PrefetchDataset element_spec=(TensorSpec(shape=(None, 256, 256, 3), dtype=tf.float32, name=None), TensorSpec(shape=(None,), dtype=tf.int32, name=None))>

train_ds=keras.utils.image_dataset_from_directory(
    directory='/content/train',
    labels="inferred",
    label_mode="int",
    class_names=None,
    batch_size=32,
    image_size=(256, 256),
)

test_ds=keras.utils.image_dataset_from_directory(
    directory='/content/test',
    labels="inferred",
    label_mode="int",
    class_names=None,
    batch_size=32,
    image_size=(256, 256),
)
     
Found 20000 files belonging to 2 classes.
Found 5000 files belonging to 2 classes.
**Normalization **

def process(image,label):
  image=tf.cast(image/255,tf.float32)
  return image,label

train_ds=train_ds.map(process)
test_ds=test_ds.map(process)
     
Creating CNN model

model = Sequential()

model.add(Conv2D(32,kernel_size=(3,3),padding='valid',activation='relu',input_shape=(256,256,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Conv2D(64,kernel_size=(3,3),padding='valid',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Conv2D(128,kernel_size=(3,3),padding='valid',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1,activation='sigmoid'))
     
/usr/local/lib/python3.10/dist-packages/keras/src/layers/convolutional/base_conv.py:107: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(activity_regularizer=activity_regularizer, **kwargs)


     

model.summary()


     
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                      │ (None, 254, 254, 32)        │             896 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization                  │ (None, 254, 254, 32)        │             128 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d (MaxPooling2D)         │ (None, 127, 127, 32)        │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_1 (Conv2D)                    │ (None, 125, 125, 64)        │          18,496 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization_1                │ (None, 125, 125, 64)        │             256 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_1 (MaxPooling2D)       │ (None, 62, 62, 64)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_2 (Conv2D)                    │ (None, 60, 60, 128)         │          73,856 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization_2                │ (None, 60, 60, 128)         │             512 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_2 (MaxPooling2D)       │ (None, 30, 30, 128)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ flatten (Flatten)                    │ (None, 115200)              │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 128)                 │      14,745,728 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 128)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 64)                  │           8,256 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_1 (Dropout)                  │ (None, 64)                  │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_2 (Dense)                      │ (None, 1)                   │              65 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 14,848,193 (56.64 MB)
 Trainable params: 14,847,745 (56.64 MB)
 Non-trainable params: 448 (1.75 KB)

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
     

from tensorflow.keras.callbacks import EarlyStopping
     

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
     

result=model.fit(train_ds,epochs=50,validation_data=test_ds, callbacks=[early_stopping])
     
Epoch 1/50
625/625 ━━━━━━━━━━━━━━━━━━━━ 60s 86ms/step - accuracy: 0.9658 - loss: 0.1041 - val_accuracy: 0.8240 - val_loss: 0.7164
Epoch 2/50
625/625 ━━━━━━━━━━━━━━━━━━━━ 50s 80ms/step - accuracy: 0.9781 - loss: 0.0644 - val_accuracy: 0.7660 - val_loss: 1.3628
Epoch 3/50
625/625 ━━━━━━━━━━━━━━━━━━━━ 53s 85ms/step - accuracy: 0.9810 - loss: 0.0623 - val_accuracy: 0.8110 - val_loss: 0.7097
Epoch 4/50
625/625 ━━━━━━━━━━━━━━━━━━━━ 79s 80ms/step - accuracy: 0.9836 - loss: 0.0465 - val_accuracy: 0.7188 - val_loss: 2.1235
Epoch 5/50
625/625 ━━━━━━━━━━━━━━━━━━━━ 50s 80ms/step - accuracy: 0.9866 - loss: 0.0442 - val_accuracy: 0.7848 - val_loss: 0.8071
Epoch 6/50
625/625 ━━━━━━━━━━━━━━━━━━━━ 82s 80ms/step - accuracy: 0.9883 - loss: 0.0384 - val_accuracy: 0.7810 - val_loss: 0.6691
Epoch 7/50
625/625 ━━━━━━━━━━━━━━━━━━━━ 53s 84ms/step - accuracy: 0.9870 - loss: 0.0426 - val_accuracy: 0.8122 - val_loss: 0.6695
Epoch 8/50
625/625 ━━━━━━━━━━━━━━━━━━━━ 51s 81ms/step - accuracy: 0.9875 - loss: 0.0371 - val_accuracy: 0.8284 - val_loss: 1.2552
Epoch 9/50
625/625 ━━━━━━━━━━━━━━━━━━━━ 50s 80ms/step - accuracy: 0.9908 - loss: 0.0296 - val_accuracy: 0.8024 - val_loss: 1.2231
Epoch 10/50
625/625 ━━━━━━━━━━━━━━━━━━━━ 85s 86ms/step - accuracy: 0.9887 - loss: 0.0357 - val_accuracy: 0.8390 - val_loss: 0.7993
Epoch 11/50
625/625 ━━━━━━━━━━━━━━━━━━━━ 81s 85ms/step - accuracy: 0.9935 - loss: 0.0233 - val_accuracy: 0.8252 - val_loss: 0.8189


     
Check Accuracy using Graph

import matplotlib.pyplot as plt

plt.plot(result.history['accuracy'],color='red',label='train')
plt.plot(result.history['val_accuracy'],color='green',label='validation')

plt.legend()
plt.show()
     

Last dataset result : overfitting
image.png


plt.plot(result.history['loss'],color='red',label='train')
plt.plot(result.history['val_loss'],color='green',label='validation')

plt.legend()
plt.show()
     


import cv2

test_img=cv2.imread('/content/test/dogs/dog.10033.jpg')
plt.imshow(test_img)
     
<matplotlib.image.AxesImage at 0x7de66bed7940>


test_img.shape

     
(396, 490, 3)

test_img = cv2.resize(test_img,(256,256))
     

test_input= test_img.reshape((1,256,256,3))
     

model.predict(test_input)
     
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step
array([[1.]], dtype=float32)
Last Model Result :Model predicted dog as cat
Screenshot 2024-10-17 121920.png


import pickle
     

pickle.dump(model,open('model.pkl','wb'))
     
