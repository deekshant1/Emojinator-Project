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




# data  = pd.read_csv("train_foo.csv")
# dataset = np.array(data)
# np.random.shuffle(dataset)
# X = dataset
# Y = dataset
# X = X[:, 1:2501]
# Y = Y[:, 0]
#
#
# X_train = X[0:10000, :]
# X_train = X_train/255.
# X_test = X[10000:11200, :]
# X_test = X_test/255.
#
# Y = Y.reshape(Y.shape[0], 1)
# Y_train = Y[0:10000, :]
# Y_train = Y_train.T
# Y_test = Y[10000:11200, :]
# Y_test = Y_test.T
# # Y_train = Y_train.astype('float32')
# # Y_test = Y_test.astype('float32')
#
# print("numbere of training examples = " + str(X_train.shape[0]))
# print("number of test examples = " + str(X_test.shape[0]))
# print("X_train shape: " + str(X_train.shape))
# print("Y_train shape " + str(Y_train.shape))
# print("X_test shape" + str(X_test.shape))
# print("Y_test shape " + str(Y_test.shape))
#
# image_x = 50
# image_y = 50
#
# train_y = np_utils.to_categorical(Y_train)
# test_y = np_utils.to_categorical(Y_test)
# train_y = train_y.reshape(train_y.shape[1], train_y.shape[2])
# test_y = test_y.reshape(test_y.shape[1], test_y.shape[2])
# X_train = X_train.reshape(X_train.shape[0], 50, 50, 1)
# X_test = X_test.reshape(X_test.shape[0], 50, 50, 1)
# print("X_train shape: " + str(X_train.shape))
# print("X_test shape: " + str(X_test.shape))




model, callbacks_list = keras_images(image_x, image_y)
model.fit(X_train, train_y, validation_data=(X_test, test_y), epochs=3, batch_size=64,
         callbacks=callbacks_list)
scores = model.evaluate(X_test, test_y, verbose=0)
print("CNN Error: %.2f%%" % (100 - scores[1] * 100))
print_summary(model)

model.save('HandEmo.h5')

# import numpy as np
# import keras
# from keras import layers
# from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
# from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
# from keras.utils import np_utils
# from keras.models import Sequential
# from keras.callbacks import ModelCheckpoint
# import pandas as pd
#
# import keras.backend as K
#
# def keras_model(image_x, image_y):
#     num_of_classes = 12
#     model = Sequential()
#     model.add(Conv2D(32, (5, 5), input_shape=(image_x, image_y, 1), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
#     model.add(Conv2D(64, (5, 5), activation='sigmoid'))
#     model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
#     model.add(Flatten())
#     model.add(Dense(1024, activation='relu'))
#     model.add(Dropout(0.6))
#     model.add(Dense(num_of_classes, activation='softmax'))
#
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     filepath = "emojinator.h5"
#     checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
#     callbacks_list = [checkpoint1]
#
#     return model, callbacks_list
#
# def main():
#     data = pd.read_csv("train_foo.csv")
#     dataset = np.array(data)
#     np.random.shuffle(dataset)
#     X = dataset
#     Y = dataset
#     X = X[:, 1:2501]
#     Y = Y[:, 0]
#
#     X_train = X[0:11000, :]
#     X_train = X_train / 255
#     X_test = X[11000:12201, :]
#     X_test = X_test / 255
#
#     # Reshape
#     Y = Y.reshape(Y.shape[0], 1)
#     Y_train = Y[0:11000, :]
#     Y_train = Y_train.T
#     Y_test = Y[11000:12201, :]
#     Y_test = Y_test.T
#
#     print("number of training examples = " + str(X_train.shape[0]))
#     print("number of test examples = " + str(X_test.shape[0]))
#     print("X_train shape: " + str(X_train.shape))
#     print("Y_train shape: " + str(Y_train.shape))
#     print("X_test shape: " + str(X_test.shape))
#     print("Y_test shape: " + str(Y_test.shape))
#     image_x = 50
#     image_y = 50
#
#     train_y = np_utils.to_categorical(Y_train)
#     test_y = np_utils.to_categorical(Y_test)
#     train_y = train_y.reshape(train_y.shape[1], train_y.shape[2])
#     test_y = test_y.reshape(test_y.shape[1], test_y.shape[2])
#     X_train = X_train.reshape(X_train.shape[0], 50, 50, 1)
#     X_test = X_test.reshape(X_test.shape[0], 50, 50, 1)
#     print("X_train shape: " + str(X_train.shape))
#     print("X_test shape: " + str(X_test.shape))
#
#     model, callbacks_list = keras_model(image_x, image_y)
#     model.fit(X_train, train_y, validation_data=(X_test, test_y), epochs=2, batch_size=64,
#               callbacks=callbacks_list)
#     scores = model.evaluate(X_test, test_y, verbose=0)
#     print("CNN Error: %.2f%%" % (100 - scores[1] * 100))
#
#     model.save('emojinator.h5')
#
#
# main()
# # #model
# # import numpy as np
# # from keras.layers import Dense, Flatten, Conv2D
# # from keras.layers import MaxPooling2D, Dropout
# # from keras.utils import np_utils, print_summary, to_categorical
# # from keras.models import Sequential
# # from keras.callbacks import ModelCheckpoint
# # import pandas as pd
# # # Scikit learn for preprocessing
# # from sklearn.model_selection import train_test_split
# # from sklearn.preprocessing import LabelEncoder
# #
# # import keras.backend as K
# #
# # data = pd.read_csv("train_foo.csv")
# # dataset = np.array(data)
# # np.random.shuffle(dataset)
# # X = dataset
# # Y = dataset
# # X = X[:, 0:2501]
# # Y = Y[:, 1]
# #
# # # Let's split the data into train and test data
# # x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.10, random_state=42)
# # num_of_classes = 12
# # # Encode the categories
# # le = LabelEncoder()
# # Y_train = le.fit_transform(y_train)
# # Y_test = le.transform(y_test)
# # train_y = to_categorical(Y_train, num_of_classes)
# # test_y = to_categorical(Y_test, num_of_classes)
# #
# #
# # image_x = 50
# # image_y = 50
# #
# # X_train = x_train.reshape(x_train.shape[0], image_x, image_y, 1)
# # X_test = x_test.reshape(x_test.shape[0], image_x, image_y, 1)
# #
# #
# #
# # # X_train = X[0:12000, :]
# # # X_train = X_train / 255.
# # # X_test = X[12000:13201, :]
# # # X_test = X_test / 255.
# # #
# # # # Reshape
# # # Y = Y.reshape(Y.shape[0], 1)
# # # Y_train = Y[0:12000, :]
# # # Y_train = Y_train.T
# # # Y_test = Y[12000:13201, :]
# # # Y_test = Y_test.T
# # #
# # # # data  = pd.read_csv("train_foo.csv")
# # # # dataset = np.array(data)
# # # # np.random.shuffle(dataset)
# # # # X = dataset
# # # # Y = dataset
# # # # X = X[:, 1:2501]
# # # # Y = Y[:, 0]
# # # #
# # # #
# # # # X_train = X[0:10000, :]
# # # # X_train = X_train/255.
# # # # X_test = X[10000:11200, :]
# # # # X_test = X_test/255.
# # # #
# # # # Y = Y.reshape(Y.shape[0], 1)
# # # # Y_train = Y[0:10000, :]
# # # # Y_train = Y_train.T
# # # # Y_test = Y[10000:11200, :]
# # # # Y_test = Y_test.T
# # # #  # Y_train = Y_train.astype('float32')
# # # #  # Y_test = Y_test.astype('float32')
# # #
# # # print("numbere of training examples = " + str(X_train.shape[0]))
# # # print("number of test examples = " + str(X_test.shape[0]))
# # # print("X_train shape: " + str(X_train.shape))
# # # print("Y_train shape " + str(Y_train.shape))
# # # print("X_test shape" + str(X_test.shape))
# # # print("Y_test shape " + str(Y_test.shape))
# # #
# # # image_x = 50
# # # image_y = 50
# # #
# # # train_y = np_utils.to_categorical(Y_train)
# # # test_y = np_utils.to_categorical(Y_test)
# # # train_y = train_y.reshape(train_y.shape[1], train_y.shape[2])
# # # test_y = test_y.reshape(test_y.shape[1], test_y.shape[2])
# # # X_train = X_train.reshape(X_train.shape[0], 50, 50, 1)
# # # X_test = X_test.reshape(X_test.shape[0], 50, 50, 1)
# # # print("X_train shape: " + str(X_train.shape))
# # # print("X_test shape: " + str(X_test.shape))
# #
# #
# # def keras_images(image_x, image_y):
# #     num_of_classes = 12
# #     model = Sequential()
# #     model.add(Conv2D(32, (5,5), input_shape=(image_x, image_y, 1), activation='relu'))
# #     model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
# #     model.add(Conv2D(64, (5,5), activation='sigmoid'))
# #     model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
# #     model.add(Flatten())
# #     model.add(Dense(1024, activation='relu'))
# #     model.add(Dropout(0.6))
# #     model.add(Dense(num_of_classes, activation='softmax'))
# #
# #     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# #     filepath = "HandEmo.h5"
# #     checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
# #     callbacks_list = [checkpoint1]
# #
# #     return model, callbacks_list
# #
# # model, callbacks_list = keras_images(image_x, image_y)
# # model.fit(X_train, train_y, validation_data=(X_test, test_y), epochs=3, batch_size=64,
# #           callbacks=callbacks_list)
# # scores = model.evaluate(X_test, test_y, verbose=0)
# # print("CNN Error: %.2f%%" % (100 -scores[1] * 100))
# # print_summary(model)
# #
# # model.save('HandEmo.h5')
# #
# # #import numpy as np
# # # import keras
# # # from keras import layers
# # # from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
# # # from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
# # # from keras.utils import np_utils
# # # from keras.models import Sequential
# # # from keras.callbacks import ModelCheckpoint
# # # import pandas as pd
# # #
# # # import keras.backend as K
# # #
# # # def keras_model(image_x, image_y):
# # #     num_of_classes = 12
# # #     model = Sequential()
# # #     model.add(Conv2D(32, (5, 5), input_shape=(image_x, image_y, 1), activation='relu'))
# # #     model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
# # #     model.add(Conv2D(64, (5, 5), activation='sigmoid'))
# # #     model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
# # #     model.add(Flatten())
# # #     model.add(Dense(1024, activation='relu'))
# # #     model.add(Dropout(0.6))
# # #     model.add(Dense(num_of_classes, activation='softmax'))
# # #
# # #     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# # #     filepath = "emojinator.h5"
# # #     checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
# # #     callbacks_list = [checkpoint1]
# # #
# # #     return model, callbacks_list
# # #
# # # def main():
# # #     data = pd.read_csv("train_foo.csv")
# # #     dataset = np.array(data)
# # #     np.random.shuffle(dataset)
# # #     X = dataset
# # #     Y = dataset
# # #     X = X[:, 1:2501]
# # #     Y = Y[:, 0]
# # #
# # #     X_train = X[0:12000, :]
# # #     X_train = X_train / 255.
# # #     X_test = X[12000:13201, :]
# # #     X_test = X_test / 255.
# # #
# # #     # Reshape
# # #     Y = Y.reshape(Y.shape[0], 1)
# # #     Y_train = Y[0:12000, :]
# # #     Y_train = Y_train.T
# # #     Y_test = Y[12000:13201, :]
# # #     Y_test = Y_test.T
# # #
# # #     print("number of training examples = " + str(X_train.shape[0]))
# # #     print("number of test examples = " + str(X_test.shape[0]))
# # #     print("X_train shape: " + str(X_train.shape))
# # #     print("Y_train shape: " + str(Y_train.shape))
# # #     print("X_test shape: " + str(X_test.shape))
# # #     print("Y_test shape: " + str(Y_test.shape))
# # #     image_x = 50
# # #     image_y = 50
# # #
# # #     train_y = np_utils.to_categorical(Y_train)
# # #     test_y = np_utils.to_categorical(Y_test)
# # #     train_y = train_y.reshape(train_y.shape[1], train_y.shape[2])
# # #     test_y = test_y.reshape(test_y.shape[1], test_y.shape[2])
# # #     X_train = X_train.reshape(X_train.shape[0], 50, 50, 1)
# # #     X_test = X_test.reshape(X_test.shape[0], 50, 50, 1)
# # #     print("X_train shape: " + str(X_train.shape))
# # #     print("X_test shape: " + str(X_test.shape))
# # #
# # #     model, callbacks_list = keras_model(image_x, image_y)
# # #     model.fit(X_train, train_y, validation_data=(X_test, test_y), epochs=10, batch_size=64, callbacks=callbacks_list)
# # #     scores = model.evaluate(X_test, test_y, verbose=0)
# # #     print("CNN Error: %.2f%%" % (100 - scores[1] * 100))
# # #
# # #     model.save('emojinator.h5')
# # #
# # #
# # # main()