from keras.layers import Bidirectional
bidir_model = Sequential()
n_steps=1
n_features =3
bidir_model.add(Bidirectional(LSTM(10, activation='relu'), input_shape=(n_steps, n_features)))
bidir_model.add(Dense(1))
bidir_model.compile(optimizer='adam', loss='mse')


# fit model
bidir_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)


# prediction sample demonstration
x_input = array([70, 80, 90])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = bidir_model.predict(x_input, verbose=0)
print(yhat)
