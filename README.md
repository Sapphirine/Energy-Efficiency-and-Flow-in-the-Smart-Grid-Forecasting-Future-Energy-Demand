# Energy-Efficiency-and-Flow-in-the-Smart-Grid-Forecasting-Future-Energy-Demand
Forecasting future (renewable) energy demand in the smart grid

Commands to run in Google colab to be able to import necessary libraries for models based on gluonts:

1. pip install gluonts
2. !pip install mxnet
3. pip install pydantic==1.4
4. pip install python-dateutil==2.8.1

Commands to load file from Google drive:

1. from google.colab import drive
drive.mount('/content/gdrive')
2. Enter authorization code
3. %cd '/content/gdrive/My Drive'
4. e.g. for saved California dataset,
import pandas as pd
caisofile= 'CaisoJan19March19windsolarAvg.csv'
caiso_dataset = pd.read_csv(caisofile)



