from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM

# vanilla LSTM
nsteps = 1
nfeatures = 3
model = Sequential()
#model.add(LSTM(50, activation='relu', input_shape=(nfeatures, nsteps))) : for method with train_test_split 
model.add(LSTM(50, activation='relu', input_shape=(nsteps, nfeatures)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
