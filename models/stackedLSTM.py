stacked_model = Sequential()

nunits = 50
n_steps= 3
n_features=1
# LSTM ( units,activation, recurrent_activation)
# recurrent_activation: Activation function to use for the recurrent step (see activations). Default: hard sigmoid (hard_sigmoid
stacked_model.add(LSTM(nunits, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
stacked_model.add(LSTM(nunits, activation='relu'))
stacked_model.add(Dense(1))
stacked_model.compile(optimizer='adam', loss='mse')

# fit model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)

# prediction sample demonstration
x_input = array([70, 80, 90])
x_input = x_input.reshape((1, n_features, n_steps))
yhat = model.predict(x_input, verbose=0)
print(yhat)
