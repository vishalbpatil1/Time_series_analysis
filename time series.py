#import libraries
import pandas as pd
import requests
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM ,Dropout
from tensorflow.keras.layers import Dense
import plotly.graph_objects as go

from tensorflow. keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import warnings 
warnings.filterwarnings('ignore')

class CustomSaver(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if epoch%5 == 0: # == 20:  # or save after some epoch, each k-th epoch etc.
            self.model.save("/content/drive/MyDrive/time_series_RNN/model_lstm_4_step{}.h5".format(epoch))

class stock:
    def create_model(data,n_step=4):
        size=data.shape[1]
        # define model
        model = Sequential()
        model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(size,1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dense(n_step))
        model.compile(optimizer='adam', loss='mse')
        return model
    def train_model(model, X_train, y_train, X_test, y_test, epochs, batch_size):
        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='min')
        #save weights after every 20
        saver = CustomSaver()
        ### train the model on train data with adam optimiser and mape loss function 
        model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=epochs, batch_size=batch_size, callbacks=[reduce_lr_loss, saver])
        return model

    def prepare_multivariate_train_data(data,col_name='Open',dep_no=60, n_step=4):
        df=data[col_name]
        Train_arr=df.values.reshape(len(df),1)
        X_Train = []
        Y_Train = []
        dep_no=dep_no
        for i in range(dep_no,Train_arr.shape[0]+1-n_step):
            # X_Train 0-59 
            X_Train.append(Train_arr[i-dep_no:i,0])
            Y_Train.append([Train_arr[i:i+n_step,0][j] for j in range(n_step)])
        # Convert into Numpy Array
        X_Train = np.array(X_Train)
        Y_Train = np.array(Y_Train)
        X_Train= np.reshape(X_Train,newshape=(X_Train.shape[0], X_Train.shape[1], 1))
        Y_Train= np.reshape(Y_Train,newshape=(Y_Train.shape[0], Y_Train.shape[1], 1))
        return df,col_name,X_Train,Y_Train,dep_no



data=pd.read_csv(r'C:\Users\Vishal\visionnlp_\stock\Final_data_train.csv')
data.columns=['Date','Value']

minimum=data['Value'].min()
maximum=data['Value'].max()
data['Value']=data['Value'].apply(lambda x:(x-minimum)/(maximum-minimum))

# splitting dataframe by row index 
data_train = data.iloc[:162000,:] 
data_test = data.iloc[162000:,:] 

df_train,col_name_train,X_Train,Y_Train,dep=stock.prepare_multivariate_train_data(data=data_train,col_name='Value',dep_no=12,n_step=4)
df_test,col_name_test,X_Test,Y_Test,dep=stock.prepare_multivariate_train_data(data=data_test,col_name='Value',dep_no=12,n_step=4)
model=stock.create_model(data=X_Train,n_step=4)
print(model.summary())

train_model = stock.train_model(model, X_train=X_Train, y_train=Y_Train, X_test=X_Test, y_test=Y_Test, epochs=50, batch_size=32)


# prediction
model=load_model('model_lstm_4_step35.h5')   
pred_df=pd.DataFrame(pred_)
pred_df.columns=['step_1_pred','step_2_pred','step_3_pred','step_4_pred']
pred_df['step_1_actual']=[Y_Test[i][0][0] for i in range(len(Y_Test))]
pred_df['step_2_actual']=[Y_Test[i][1][0] for i in range(len(Y_Test))]
pred_df['step_3_actual']=[Y_Test[i][2][0] for i in range(len(Y_Test))]
pred_df['step_4_actual']=[Y_Test[i][3][0] for i in range(len(Y_Test))]
pred_df=pred_df.apply(lambda y :y*(maximum-minimum)+minimum)
pred_df['Date']=data_test["Date"][12:-3].to_list()
pred_df=pred_df[['Date','step_1_actual','step_1_pred','step_2_actual','step_2_pred','step_3_actual','step_3_pred','step_4_actual','step_4_pred']]
print(pred_df)



# one step forecasting performance
y_true=pred_df['step_1_pred'].values
y_pred=pred_df['step_1_actual'].values
mse=mean_squared_error(y_true, y_pred)
rmse=np.sqrt(mse)
print("RMSE for model :",rmse)

R_sqr=r2_score(y_true,y_pred)
print("R squred value :",R_sqr)

def mda(actual,predicted):
#Mean Directional Accuracy
   return np.mean((np.sign(actual[1:] - actual[:-1]) == np.sign(predicted[1:] - actual[:-1])).astype(int))
mda=mda(actual=y_true,predicted=y_pred)
print("Mean Directional Accuracy :",mda)