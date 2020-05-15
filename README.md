# Energy-Efficiency-and-Flow-in-the-Smart-Grid-Forecasting-Future-Energy-Demand
Forecasting future (renewable) energy demand in the smart grid

Commands to run in Google colab to be able to import necessary libraries for models based on gluonts:

1. pip install gluonts
2. !pip install mxnet
3. pip install pydantic==1.4
4. pip install python-dateutil==2.8.1

Commands to load file from Google drive:

1. from google.colab import drive 
2. drive.mount('/content/gdrive')
3. Enter authorization code
4. %cd '/content/gdrive/My Drive'
5. e.g. for saved California dataset, 
<p> import pandas as pd </p>
<p> caisofile= 'CaisoJan19March19windsolarAvg.csv' </p>
<p> caiso_dataset = pd.read_csv(caisofile) </p>



