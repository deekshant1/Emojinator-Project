#model
import numpy as np
from keras.layers import Dense, Flatten, Conv2D
from keras.layers import MaxPooling2D, Dropout
from keras.utils import np_utils, print_summary, to_categorical
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import pandas as pd
import keras.backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def keras_model():
   data = pd.read_csv("train_foo.csv")
   # X = data.values[:, :-1] / 255.0
   # Y = data["character"].values
   dataset = np.array(data)
   X = dataset
   Y = dataset


   n_classes = 12

   # Let's split the data into train and test data
   x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=12)

   # Encode the categories
   le = LabelEncoder()
   Y_train = le.fit_transform(y_train)
   Y_test = le.transform(y_test)
   train_y = to_categorical(Y_train, n_classes)
   test_y = to_categorical(Y_test, n_classes)


   image_x = 50
   image_y = 50

   X_train = x_train.reshape(x_train.shape[0], image_x, image_y, 1)
   X_train = X_train/255.
   X_test = x_test.reshape(x_test.shape[0], image_x, image_y, 1)
   X_test = X_test/255.
   
   
   model, callbacks_list = keras_model(image_x, image_y)
   model.fit(X_train, train_y, validation_data=(X_test, test_y), epochs=2, batch_size=64, callbacks=callbacks_list)
   scores = model.evaluate(X_test, test_y, verbose=0)
   print("CNN Error: %.2f%%" % (100 - scores[1] * 100))
   model.save('emojinator.h5')


def keras_images(image_x, image_y):
   num_of_classes = 12
   model = Sequential
   model.add(Conv2D(32, (5, 5), input_shape=(image_x, image_y, 1), activation='relu'))
   model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
   model.add(Conv2D(64, (5, 5), activation='sigmoid'))
   model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
   model.add(Flatten())
   model.add(Dense(1024, activation='relu'))
   model.add(Dropout(0.6))
   model.add(Dense(num_of_classes, activation='softmax'))

   model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
   filepath = "HandEmo.h5"
   checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
   callbacks_list = [checkpoint1]

   return model, callbacks_list

keras_model()
