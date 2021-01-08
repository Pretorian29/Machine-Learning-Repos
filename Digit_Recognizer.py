#Competition Description
'''
MNIST ("Modified National Institute of Standards and Technology") is the de facto “hello world” 
dataset of computer vision. Since its release in 1999, this classic dataset of handwritten images 
has served as the basis for benchmarking classification algorithms. As new machine learning techniques emerge,
MNIST remains a reliable resource for researchers and learners alike.

In this competition, your goal is to correctly identify digits from a dataset of tens of thousands of handwritten images. 
We’ve curated a set of tutorial-style kernels which cover everything from regression to neural networks. 
We encourage you to experiment with different algorithms to learn first-hand what works well and how techniques compare.
'''

# 'Digit Recognizer' - Competition - Accuracy Acquired: 0.98889
#-----------------------------------------------------------------------------

# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras 
import seaborn as sns
from pandas import Series
from numpy.random import randn

print('')
print("Libraries imported...")
print('')
#                                                            Data Preprocessing
#-----------------------------------------------------------------------------

#Loading data and visualizing shape
df_train = pd.read_csv('../input/digit-recognizer/train.csv')
df_test = pd.read_csv('../input/digit-recognizer/test.csv')
#print(df_train.shape)
#print(df_test.shape)

#df_test.head()

#df_train.describe()

#df_train.isnull().any().describe()

#Defining X and Y withing train set and visualizing shape
X_train = df_train.iloc[:,1:]
y_train = df_train.iloc[:,0]
#print(X_train.shape)
#print(y_train.shape)

g = sns.countplot(y_train) #Show the counts of observations in each categorical bin using bars.
y_train.value_counts() #Return a Series containing counts of unique values.

#Normalizing Pixel values  
X_train = X_train/255.0
df_test = df_test/255.0

#Reshaping train data values
X_train = X_train.values.reshape(-1,28,28)
for i in range(6, 9):
    plt.subplot(3,3, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(X_train[i],cmap = 'gray')
    plt.title(y_train[i]);

#Reshaping to match test data
X_train = X_train.reshape(-1,28,28,1)
df_test = df_test.values.reshape(-1,28,28,1)

print('')
print("Data loaded and processed...")
print('')

#                                                                     Model
#-----------------------------------------------------------------------------

model = keras.models.Sequential([
    keras.layers.Conv2D(32,kernel_size = (3,3),input_shape = (28,28,1)),
    keras.layers.Conv2D(32,kernel_size = (3,3),activation = 'relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Dropout(0.2),
    
    keras.layers.Conv2D(64,kernel_size = (3,3),activation='relu'),
    keras.layers.Conv2D(64,kernel_size = (3,3),activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Dropout(0.2),
    
    keras.layers.Flatten(),
    keras.layers.Dense(480,activation = 'relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10,activation = 'softmax')
    ])

model.compile(optimizer = 'adam',loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])

model_hist = model.fit(X_train,y_train,epochs = 3,validation_split = 0.2)

#                                                            Predicting Values
#-----------------------------------------------------------------------------
# Predicting the Test set results

predictions = model.predict(df_test)
predictions = np.argmax(predictions,axis=1)
print("Predictions done...")

#                                                        Evaluation of Results
#-----------------------------------------------------------------------------


acc = model_hist.history['accuracy']
val_acc = model_hist.history['val_accuracy']
loss = model_hist.history['loss']
val_loss = model_hist.history['val_loss']
epochs = range(len(acc))

#Graphing our training and validation
plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.ylabel('loss') 
plt.xlabel('epoch')
plt.legend()
plt.show()

plt.figure()
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.ylabel('accuracy') 
plt.xlabel('epoch')
plt.legend()
plt.show()

model.summary()

#                                                            Printing Results
#-----------------------------------------------------------------------------

output = pd.DataFrame({'ImageId': list(range(1, len(predictions)+1)), 'Label': predictions})
output.to_csv('Pretorian_submission.csv', index = False)
print("Submission file successfully created!")


#-----------------------------------------------------------------------------
__author__ = 'Pretorian29 (2020)'


