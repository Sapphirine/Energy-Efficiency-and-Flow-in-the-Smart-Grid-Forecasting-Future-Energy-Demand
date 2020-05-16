# Energy-Efficiency-and-Flow-in-the-Smart-Grid-Forecasting-Future-Energy-Demand
Forecasting future (renewable) energy demand in the smart grid

Commands to run in Google colab to be able to import necessary libraries for models based on gluonts:

1. pip install gluonts
2. !pip install mxnet
3. pip install pydantic==1.4
4. pip install python-dateutil==2.8.1

Commands to load file from Google drive into colab (must have file saved in drive):

1. from google.colab import drive 
2. drive.mount('/content/gdrive')
3. Enter authorization code
4. %cd '/content/gdrive/My Drive'
5. e.g. for saved California dataset, 
<p> import pandas as pd </p>
<p> caisofile= 'CaisoJan19March19windsolarAvg.csv' </p>
<p> caiso_dataset = pd.read_csv(caisofile) </p>


Commands to run the demo dashboard app: 
1. run the commands above to install (correct versions of) gluonts, mxnet, pydantic, and python-dateutil
2. pip install dash==1.8.0
3. !wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
   <p> !unzip ngrok-stable-linux-amd64.zip </p>

4. get_ipython().system_raw('./ngrok http 8050 &')
5. ! curl -s http://localhost:4040/api/tunnels | python3 -c \
    "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"
6. Copy the output of step 5 (this is the link where the dashboard app will load)
7. Save the app by running %%writefile my_app7.py where the code of the app is in the same cell
8. !python my_app7.py 
9. Open new tab and paste link from step 6. 


Short summary of python notebooks:

- LSTM_with_covariates.ipynb contains the keras-based LSTM models and experiments

- data_preprocessing_replearning.ipynb	contains data preprocessing techniques on 1 minute electricity data in nyc 

- deep_learning_gluon2.ipynb contains the gluonts-based deep learning models and experiments

- visualizations_deep_learning_gluon.ipynb contain visualizations and apps

