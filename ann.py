

import numpy as np
# from keras.datasets import mnist
from tensorflow.keras.models import Sequential
from keras.layers import Dense
# from keras.layers import Dropout,Flatten
from keras.utils import np_utils

np.random.seed(10)

# Loading the data set using pandas as data frame format 
import pandas as pd
train = pd.read_csv("D:/50_Startups (2).csv")
test = pd.read_csv("D:/50_Startups (2).csv")

test.columns = test.columns.str.replace("R&D Spend","A")
train.columns = train.columns.str.replace("R&D Spend","B")

# Label Encoder
from sklearn.preprocessing import LabelEncoder
# creating instance of labelencoder
labelencoder = LabelEncoder()

# Data Split into Input and Output variables
X = train.iloc[:, 0:9]
Y = test.iloc[:, 0:9]


train.columns
X['State']= labelencoder.fit_transform(X['State'])
Y['State']= labelencoder.fit_transform(Y['State'])

train = X
test = Y

# Separating the data set into 2 parts - all the inputs and label columns
# converting the integer type into float32 format 
x_train = train.iloc[:,1:].values.astype("float32")
x_test = test.iloc[:,1:].values.astype("float32")
y_train = train.B.values.astype("float32")
y_test = test.A.values.astype("float32")

# Normalizing the inputs to fall under 0-1 by 
# diving the entire data with 255 (max pixel value)
x_train = x_train/2
x_test = x_test/2

# one hot encoding outputs for both train and test data sets 
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# Storing the number of classes into the variable num_of_classes 
num_of_classes = y_test.shape[1]
x_train.shape
y_train.shape
x_test.shape
y_test.shape

# Creating a user defined function to return the model for which we are
# giving the input to train the ANN mode
def design_mlp():
    # Initializing the model 
    model = Sequential()
    model.add(Dense(1000,input_dim =4,activation="relu"))
    model.add(Dense(100,activation="tanh"))
    model.add(Dense(200,activation="tanh"))
    model.add(Dense(130,activation="tanh"))
    model.add(Dense(num_of_classes,activation="softmax"))
    model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
    return model

# building a cnn model using train data set and validating on test data set
model = design_mlp()

# fitting model on train data
model.fit(x=x_train,y=y_train,batch_size=500,epochs=5)

# Evaluating the model on test data  
eval_score_test = model.evaluate(x_test,y_test,verbose = 1)
print ("Accuracy: %.3f%%" %(eval_score_test[1]*100)) 
# accuracy on test data set

# accuracy score on train data 
eval_score_train = model.evaluate(x_train,y_train,verbose=0)
print ("Accuracy: %.3f%%" %(eval_score_train[1]*100)) 
# accuracy on train data set


###fIREFOREST####

# Import necessary libraries for MLP and reshaping the data structres
import numpy as np
# from keras.datasets import mnist
from tensorflow.keras.models import Sequential
from keras.layers import Dense
# from keras.layers import Dropout,Flatten
from keras.utils import np_utils

np.random.seed(10)

# Loading the data set using pandas as data frame format 
import pandas as pd
train = pd.read_csv("D:/fireforests.csv")
test = pd.read_csv("D:/fireforests.csv")


# Separating the data set into 2 parts - all the inputs and label columns
# converting the integer type into float32 format 
x_train = train.iloc[:,2:].values.astype("float32")
x_test = test.iloc[:,2:].values.astype("float32")
y_train = train.DMC.values.astype("float32")
y_test = test.DMC.values.astype("float32")

# Normalizing the inputs to fall under 0-1 by 
# diving the entire data with 255 (max pixel value)
x_train = x_train/13
x_test = x_test/13

# one hot encoding outputs for both train and test data sets 
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# Storing the number of classes into the variable num_of_classes 
num_of_classes = y_test.shape[1]
x_train.shape
y_train.shape
x_test.shape
y_test.shape

# Creating a user defined function to return the model for which we are
# giving the input to train the ANN mode
def design_mlp():
    # Initializing the model 
    model = Sequential()
    model.add(Dense(1000,input_dim =28,activation="relu"))
    model.add(Dense(100,activation="tanh"))
    model.add(Dense(200,activation="tanh"))
    model.add(Dense(130,activation="tanh"))
    model.add(Dense(num_of_classes,activation="softmax"))
    model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
    return model

# building a cnn model using train data set and validating on test data set
model = design_mlp()

# fitting model on train data
model.fit(x=x_train,y=y_train,batch_size=500,epochs=5)

# Evaluating the model on test data  
eval_score_test = model.evaluate(x_test,y_test,verbose = 1)
print ("Accuracy: %.3f%%" %(eval_score_test[1]*100)) 
# accuracy on test data set

# accuracy score on train data 
eval_score_train = model.evaluate(x_train,y_train,verbose=0)
print ("Accuracy: %.3f%%" %(eval_score_train[1]*100)) 
# accuracy on train data set





#####CONCRETE#######
Import necessary libraries for MLP and reshaping the data structres
import numpy as np
# from keras.datasets import mnist
from tensorflow.keras.models import Sequential
from keras.layers import Dense
# from keras.layers import Dropout,Flatten
from keras.utils import np_utils

np.random.seed(10)

# Loading the data set using pandas as data frame format 
import pandas as pd
train = pd.read_csv("D:/concrete.csv")
test = pd.read_csv("D:/concrete.csv")



# Separating the data set into 2 parts - all the inputs and label columns
# converting the integer type into float32 format 
x_train = train.iloc[:,1:].values.astype("float32")
x_test = test.iloc[:,1:].values.astype("float32")
y_train = train.B.values.astype("float32")
y_test = test.A.values.astype("float32")

# Normalizing the inputs to fall under 0-1 by 
# diving the entire data with 255 (max pixel value)
x_train = x_train/2
x_test = x_test/2

# one hot encoding outputs for both train and test data sets 
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# Storing the number of classes into the variable num_of_classes 
num_of_classes = y_test.shape[1]
x_train.shape
y_train.shape
x_test.shape
y_test.shape

# Creating a user defined function to return the model for which we are
# giving the input to train the ANN mode
def design_mlp():
    # Initializing the model 
    model = Sequential()
    model.add(Dense(1000,input_dim =4,activation="relu"))
    model.add(Dense(100,activation="tanh"))
    model.add(Dense(200,activation="tanh"))
    model.add(Dense(130,activation="tanh"))
    model.add(Dense(num_of_classes,activation="softmax"))
    model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
    return model

# building a cnn model using train data set and validating on test data set
model = design_mlp()

# fitting model on train data
model.fit(x=x_train,y=y_train,batch_size=500,epochs=5)

# Evaluating the model on test data  
eval_score_test = model.evaluate(x_test,y_test,verbose = 1)
print ("Accuracy: %.3f%%" %(eval_score_test[1]*100)) 
# accuracy on test data set

# accuracy score on train data 
eval_score_train = model.evaluate(x_train,y_train,verbose=0)
print ("Accuracy: %.3f%%" %(eval_score_train[1]*100)) 
# accuracy on train data set
#######################################################


#######FLOAT######
# Import necessary libraries for MLP and reshaping the data structres
import numpy as np
# from keras.datasets import mnist
from tensorflow.keras.models import Sequential
from keras.layers import Dense
# from keras.layers import Dropout,Flatten
from keras.utils import np_utils

np.random.seed(10)

# Loading the data set using pandas as data frame format 
import pandas as pd
train = pd.read_csv("D:/RPL.csv")
test = pd.read_csv("D:/RPL.csv")



# Separating the data set into 2 parts - all the inputs and label columns
# converting the integer type into float32 format 
x_train = train.iloc[:,1:].values.astype("float32")
x_test = test.iloc[:,1:].values.astype("float32")
y_train = train.B.values.astype("float32")
y_test = test.A.values.astype("float32")

# Normalizing the inputs to fall under 0-1 by 
# diving the entire data with 255 (max pixel value)
x_train = x_train/2
x_test = x_test/2

# one hot encoding outputs for both train and test data sets 
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# Storing the number of classes into the variable num_of_classes 
num_of_classes = y_test.shape[1]
x_train.shape
y_train.shape
x_test.shape
y_test.shape

# Creating a user defined function to return the model for which we are
# giving the input to train the ANN mode
def design_mlp():
    # Initializing the model 
    model = Sequential()
    model.add(Dense(1000,input_dim =4,activation="relu"))
    model.add(Dense(100,activation="tanh"))
    model.add(Dense(200,activation="tanh"))
    model.add(Dense(130,activation="tanh"))
    model.add(Dense(num_of_classes,activation="softmax"))
    model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
    return model

# building a cnn model using train data set and validating on test data set
model = design_mlp()

# fitting model on train data
model.fit(x=x_train,y=y_train,batch_size=500,epochs=5)

# Evaluating the model on test data  
eval_score_test = model.evaluate(x_test,y_test,verbose = 1)
print ("Accuracy: %.3f%%" %(eval_score_test[1]*100)) 
# accuracy on test data set

# accuracy score on train data 
eval_score_train = model.evaluate(x_train,y_train,verbose=0)
print ("Accuracy: %.3f%%" %(eval_score_train[1]*100)) 
# accuracy on train data set
