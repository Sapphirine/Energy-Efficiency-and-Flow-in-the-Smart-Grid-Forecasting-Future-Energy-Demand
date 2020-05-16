The interactive options in the app are given by

-  prediction length
- day of months
- starting hour (of where to predict)

It is also possible to choose region 
- California, or
- New England

Also, you can check boxes for visualizing

- early morning
- morning
- afternoon
- evening

In update_trend_graph, the trend is default showing the series at indices 1 to 30 in the return dcc.Graph line. Please change the indices to the desired indices where you want to view the trend.

In update_graph, the model is set to the default DeepAR model. It is possible to copy and paste another model in this function.

Also, to restart the app, rerun steps 3, 4 (where you enter 'y' for yes), 5-9. 
