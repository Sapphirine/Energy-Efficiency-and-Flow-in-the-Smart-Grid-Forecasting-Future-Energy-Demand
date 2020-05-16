%%writefile my_app7.py
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import pandas as pd 

import gluonts
import mxnet 

from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer

from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator

from gluonts.evaluation.backtest import make_evaluation_predictions

caiso_url = 'https://raw.githubusercontent.com/sy2657/representation_learning/master/caiso_jan2020_systemloadresource_tacnorth.csv'
#ne_url ='https://raw.githubusercontent.com/sy2657/representation_learning/master/neweng_jan2020.csv'

ne_url = 'https://raw.githubusercontent.com/sy2657/representation_learning/master/NewEnglandNov19WindSolar.csv'
edataset = pd.read_csv(ne_url) # energy dataset

purl1 ='https://raw.githubusercontent.com/sy2657/representation_learning/master/deepareNewEngNov201index100to150csv9len30'
predictions1 = pd.read_csv(purl1)

apredictions1 = predictions1['iso']

#newengland = 'NewEnglandNov19WindSolar.csv'
#neweng_dataset = pd.read_csv(newengland)
#neweng_dataset =neweng_dataset.rename(columns={"ws_forecast (sum) (california_iso)": "iso"})
#edataset = neweng_dataset

#normalize

emax = edataset['iso'].max()
#print(ne_max)
emin = edataset['iso'].min()

elen = len(edataset)
a = list(range(elen))

from gluonts.dataset.common import ListDataset

datestart = edataset['DateTime'][0] 

training_data = ListDataset([{"start":datestart, "target": edataset.iso.values}], freq = "1H")
#norm_training_data = ListDataset([{"start":datestart, "target": normalized.iso.values}], freq = "1H")

app = dash.Dash()

app.layout= html.Div(children = [
                                 html.Div(children="Energy forecasting"),
                                 html.Div([ html.P('Enter (custom) prediction length, day of month, and starting hour')]),
                                 dcc.Input(id='input', value=30, type='number', placeholder="prediction length"),
                                 dcc.Input(id='input-day', value=15, type='number', placeholder="day of month"), # choose start date and hour 
                                 dcc.Input(id='input-hour', value=11, type='number', placeholder="starting hour of day"),
                                 dcc.RadioItems(id='chooseregion', options = [{'label':'California', 'value':'ca'}, {'label':'New England', 'value':'ne'}], value='ne'),
                                 # dropdown of region -> use this input to multiple callbacks 
                                 html.Div(id='output-graph'), # time series and predictions (and conf. intervals 
                                 dcc.Checklist(id='hier', options=[{'label':'early morning', 'value':'earlymorning'},
                                               {'label':'morning', 'value': 'morning'},
                                               {'label':'afternoon', 'value':'afternoon'},
                                               {'label':'evening', 'value':'evening'}], value=['morning']),
                                 html.Div(id='output-graph3'), # output graph for hour of day 
                                 # input for trend: window size 
                                 dcc.Input(id='input-window', value=3, type='number', placeholder ="trend window length"),
                                 html.Div(id='output-graph2')# another output graph for trend
                                 # display recommended power consumption 
])



@app.callback(Output(component_id = 'output-graph3', component_property='children'),
              [Input(component_id='hier', component_property='value'), Input(component_id='chooseregion', component_property='value')])
def update_hour_graph(input_values, reg_value):
  if reg_value=='ca':
    edataset2 =  pd.read_csv(caiso_url)
  if reg_value=='ne':
    edataset2 = pd.read_csv(ne_url)
  morning_iso = []
  afternoon_iso = []
  evening_iso = []
  earlymorning_iso = []
  dtime = []
  nanarray = []
  for index, row in edataset2.iterrows():
    date_time = row['DateTime']
    iso_val = row['iso']
    split_date_time = date_time.split()
    split_date_time1 = split_date_time[1]
    hour_min_sec = split_date_time1.split(":")
    hour= float(hour_min_sec[0])
    dtime.append(date_time)
    nanarray.append(float('nan'))
    if 0 <= hour < 6:
      earlymorning_iso.append(iso_val)
      morning_iso.append(float('nan'))
      afternoon_iso.append(float('nan'))
      evening_iso.append(float('nan'))
    if 6 <= hour <12:
      morning_iso.append(iso_val)
      afternoon_iso.append(float('nan'))
      evening_iso.append(float('nan'))
      earlymorning_iso.append(float('nan'))
    if 12 <= hour < 18:
      afternoon_iso.append(iso_val)
      morning_iso.append(float('nan'))
      evening_iso.append(float('nan'))
      earlymorning_iso.append(float('nan'))
    if 18 <= hour < 24:
      evening_iso.append(iso_val)
      afternoon_iso.append(float('nan'))
      morning_iso.append(float('nan'))
      earlymorning_iso.append(float('nan'))
  max_lim = 300
  min_lim = 100
  ivalues = input_values
  if 'earlymorning' not in ivalues:
    earlymorning_iso = nanarray
  if 'morning' not in ivalues:
    morning_iso = nanarray
  if 'afternoon' not in ivalues:
    afternoon_iso = nanarray
  if 'evening' not in ivalues:
    evening_iso = nanarray
  return dcc.Graph(id='ex-graph3',
                   figure = {
                       'data': [{'x': dtime[min_lim:max_lim], 'y': earlymorning_iso[min_lim:max_lim], 'type':'line', 'name':'early morning' },
                                {'x': dtime[min_lim:max_lim], 'y': morning_iso[min_lim:max_lim], 'type': 'line', 'name':' morning'},
                                {'x': dtime[min_lim:max_lim], 'y': afternoon_iso[min_lim:max_lim], 'type': 'line', 'name':'afternoon'},
                                {'x': dtime[min_lim:max_lim], 'y': evening_iso[min_lim:max_lim], 'type': 'line', 'name': 'evening'}],
                       'layout':{'title': 'Hier. time series divided by hour of day'}
                   })
  
# trend update
@app.callback(Output(component_id = 'output-graph2', component_property='children'),
              [Input(component_id='input-window', component_property='value')])

def update_trend_graph(input_win):
  arr = edataset['iso']
  s = pd.Series(arr)
  edataset['rolling_mean'] = s.rolling(input_win).mean()
  from scipy.stats import linregress
  trend_interval = 5 
  a = list(range(1, trend_interval+1))
  div_by_trend = 0
  # array holding the trend
  array_trend = []
  temp_trend= []
  plot_trend = [] # array of pts 
  temp_plot_trend = []
  for ind in edataset.index:
    if div_by_trend==trend_interval:
      div_by_trend=1
      # calculate slope
      trend = linregress(a, temp_trend)
      a_extend = [trend[0]]*trend_interval 
      array_trend.extend(a_extend)
      # calculate the pts to be plotted
      yintercept = temp_trend[0]
      temp_plot_trend.append(yintercept) 
      for j in range(1,trend_interval):
        temp_plot_trend.append(yintercept+ trend[0]*j)
      plot_trend.extend(temp_plot_trend)
      temp_plot_trend=[]
      temp_trend= []
      temp_trend.append(edataset['rolling_mean'][ind]) # change 'iso' to rollingmean 
      continue
    temp_trend.append(edataset['rolling_mean'][ind])
    div_by_trend = div_by_trend+1
  # last set of values
  last_a = list(range(1, div_by_trend+1))
  last_trend = linregress(last_a, temp_trend)
  last_extend = [last_trend[0]]*div_by_trend
  array_trend.extend(last_extend)
  # plot
  yintercept = temp_trend[0]
  temp_plot_trend.append(yintercept)
  for j in range(1, div_by_trend):
    temp_plot_trend.append(yintercept+ last_trend[0]*j)
  plot_trend.extend(temp_plot_trend)
  return html.Div(dcc.Graph(id='ex-graph2',
                   figure = {
                       'data': [{'x': edataset.DateTime[1:30], 'y': edataset.iso[1:30], 'type':'line', 'name':'orig. time series values' },
                                {'x': edataset.DateTime[1:30], 'y': plot_trend[1:30], 'type': 'line', 'name':'trend lines '}],
                       'layout':{'title': 'DeepARE prediction'}
                   }))

  

@app.callback(
    Output(component_id='output-graph', component_property='children'),
    [Input(component_id ='input', component_property ='value'),
     Input(component_id ='input-day', component_property='value'), # input_data2
     Input(component_id ='input-hour', component_property= 'value') # input_data3
]
)

def update_graph(input_data, input_data2, input_data3):
  # gluonts 
  test_index = 0 
  ex_day = 15
  ex_hour = 11
  for index, row in edataset.iterrows():
    date_time = row['DateTime']
    iso_val = row['iso']
    split_date_time = date_time.split()
    split_date_time1 = split_date_time[1]
    hour_min_sec = split_date_time1.split(":")
    #print("hour min sec", hour_min_sec)
    hour= float(hour_min_sec[0])
    # determine day 
    dat = split_date_time[0]
    dat1 = dat.split("-")
    day = float(dat1[1])
    if day == ex_day and hour == ex_hour:
      test_index = index
      break
  # test series 
  #norm_test = normalized[test_index: test_index+input_data]
  #norm_test_dataset = ListDataset([{"start": date_time, "target": norm_test.iso.values}], freq = "1H")
  test_dataset = edataset[test_index:test_index + 50]
  testing_data = ListDataset([{"start": date_time, "target": test_dataset.iso.values}], freq = "1H")
  estimator1 = DeepAREstimator(freq="1H", prediction_length=input_data, trainer=Trainer(epochs=10,batch_size=10))
  predictor1 = estimator1.train(training_data= training_data) 
  deepare_forecast_it, deepare_ts_it = make_evaluation_predictions(dataset=testing_data, predictor=predictor1, num_samples=10) 
  deepare_forecasts = list(deepare_forecast_it)
  deepare_tss = list(deepare_ts_it)
  dm = deepare_forecasts[0].mean
  # rescale them to the values 
  # first normalize 
  normdm = []
  origdm =[]
  dmin= min(dm)
  dmax = max(dm)
  den = dmax- dmin
  dlen = len(dm)
  for i in range(0, dlen):
    dval =(dm[i]- dmin)/float(den)
    normdm.append((dm[i]- dmin)/float(den))
    #then rescale
    origval = dval*(emax-emin) + emin
    origdm.append(origval)
  
  return html.Div(dcc.Graph(id='ex-graph',
                   figure = {
                       'data': [{'x': edataset.DateTime, 'y': edataset.iso, 'type':'line', 'name':'orig. time series values' },
                                {'x': edataset.DateTime[100:130], 'y': origdm, 'type': 'line', 'name':'prediction in test series'}],
                       'layout':{'title': 'DeepARE prediction'}
                   }))

# update graph: simple feedforward method
  
if __name__=='__main__':
  app.run_server(debug=True)
