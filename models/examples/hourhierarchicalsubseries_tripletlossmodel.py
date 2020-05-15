# here is a simple example of how to run the triplet loss feedforward network on an hour zone divided dataset


# read data from url
ne_url = 'https://raw.githubusercontent.com/sy2657/representation_learning/master/NewEnglandNov19WindSolar.csv'
neweng_dataset = pd.read_csv(ne_url) 

# alternatively, read data from google drive
import pandas as pd 
newengland = 'NewEnglandNov19WindSolar.csv'
neweng_dataset = pd.read_csv(newengland)

# normalize

normalized_neweng = neweng_dataset
# normalize new eng dataframe
ne_max = neweng_dataset['iso'].max()
#print(ne_max)
ne_min = neweng_dataset['iso'].min()
normalized_neweng['iso'] = (neweng_dataset['iso'] - ne_min)/(ne_max - ne_min)

# convert into hour zone subseries dataframes

colnames=["DateTime", "iso"]
# define empty dfs 
morningdf = pd.DataFrame(columns=colnames)
afternoondf = pd.DataFrame(columns=colnames)
eveningdf = pd.DataFrame(columns= colnames)
earlymorningdf= pd.DataFrame(columns= colnames)

morning_time = []
morning_iso = []

afternoon_time = []
afternoon_iso = []

evening_time = []
evening_iso = []

earlymorning_time = []
earlymorning_iso = []

for index, row in normalized_neweng.iterrows(): # normalized_neweng
    date_time = row['DateTime']
    iso_val = row['iso']
    split_date_time = date_time.split()
    split_date_time1 = split_date_time[1]
    hour_min_sec = split_date_time1.split(":")
    #print("hour min sec", hour_min_sec)
    hour= hour_min_sec[0]
    minute = hour_min_sec[1]   
    #print("hour", hour) 
    hour = float(hour)
    # determine zone
    if 0 <= hour <= 6:
      earlymorning_time.append(date_time)
      earlymorning_iso.append(iso_val)
    if 6 <= hour <=12:
      morning_time.append(date_time)
      morning_iso.append(iso_val)
    if 12 <= hour <= 18:
      afternoon_time.append(date_time)
      afternoon_iso.append(iso_val)
    if 18 <= hour <= 24:
      evening_time.append(date_time)
      evening_iso.append(iso_val)

# set dataframes

morningdf["DateTime"] = morning_time
morningdf["iso"] = morning_iso
earlymorningdf["DateTime"] = earlymorning_time
earlymorningdf["iso"]=earlymorning_iso
afternoondf["DateTime"] = afternoon_time
afternoondf["iso"] = afternoon_iso
eveningdf["DateTime"] = evening_time
eveningdf["iso"] = evening_iso

# create ListDataset obj.s for training and testing

# extract the first timestamp from each series
morning_first_ts = morning_time[0]
earlymorning_first_ts= earlymorning_time[0]
afternoon_first_ts = afternoon_time[0]
evening_first_ts = evening_time[0]

# training
morningtrain = ListDataset(
    [{"start":morning_first_ts , "target": morningdf.iso.values}],
    freq = "1H"
)

afternoontrain = ListDataset(
[{"start":afternoon_first_ts , "target": afternoondf.iso.values}],
    freq = "1H"
)

earlymorningtrain = ListDataset(
    [{"start":earlymorning_first_ts , "target": earlymorningdf.iso.values}],
    freq = "1H"
)

eveningtrain = ListDataset(
    [{"start":evening_first_ts , "target": eveningdf.iso.values}],
    freq = "1H"
)

# testing
teststart = 36
testend = 42
mtest = morningdf[teststart:testend]
atest = afternoondf[teststart:testend]
etest = eveningdf[teststart:testend]
emtest = earlymorningdf[teststart:testend]

morningtest = ListDataset([{"start":mtest["DateTime"][teststart], "target":mtest.iso.values}], freq = "1H")  
afternoontest = ListDataset([{"start":atest["DateTime"][teststart], "target": atest.iso.values}], freq="1H")
eveningtest = ListDataset([{"start":etest["DateTime"][teststart], "target": etest.iso.values}], freq="1H")
earlymorningtest = ListDataset([{"start":emtest["DateTime"][teststart], "target": emtest.iso.values}], freq="1H")

# run choose_xneg file to load this function 
# define other_subseries based on the subseries you chose; e.g. choose morning
other_subseries = [eveningdf, earlymorningdf, afternoondf] 

# run compute_triplet_loss file to load this function

# run tripletlossfeedforwardmodel to load this function : make sure predlen within hybrid_forward method matches pred_length below
# make sure it is named MyEstimator1 

import mxnet as mx
pred_length = 20
estimator1 = MyEstimator1(prediction_length= pred_length, context_length=2*pred_length, freq="1H",num_cells=10, trainer=Trainer(ctx="cpu", epochs=5, learning_rate=1e-3, hybridize=False, num_batches_per_epoch=20))

predictor1 = estimator1.train(morningtrain) # input training data

# test on morning test data

from gluonts.evaluation.backtest import make_evaluation_predictions

forecast_it, ts_it = make_evaluation_predictions(
    dataset=morningtest,  # test dataset  norm_test_NE_dataset, test_NE_dataset
    predictor=predictor1,  # predictor
    num_samples=30,  # number of sample paths we want for evaluation
)

# accuracy eval. and generate plot

forecasts = list(forecast_it)
tss = list(ts_it)

from gluonts.evaluation import Evaluator
# step 3b 2 metrics
evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(morningtest))

# print metric
item_metrics.head()

# plot
d_triploss_ts_entry = tss[0]
d_triploss_forecast_entry= forecasts[0]
plot_prob_forecasts(d_triploss_ts_entry, d_triploss_forecast_entry)
