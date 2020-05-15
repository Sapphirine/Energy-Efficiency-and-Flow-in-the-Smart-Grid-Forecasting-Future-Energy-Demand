# hierarchical order dataframes creation
# hour of the day: morning 6 am to 12 pm, afternoon 12 pm to 6 pm, evening 6 pm to 12 am, early morning 12 am to 6 am 

# input: set dataframe
df = normalized_neweng
# input: test indices
teststart = 36
testend = 42 

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

for index, row in df.iterrows(): # normalized_neweng
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

# extract the first timestamp from each series
morning_first_ts = morning_time[0]
earlymorning_first_ts= earlymorning_time[0]
afternoon_first_ts = afternoon_time[0]
evening_first_ts = evening_time[0]


morningtrain = ListDataset(
    [{"start":morning_first_ts , "target": morningdf.iso.values}],
    freq = "1H"
)

afternoontrain = ListDataset(
[{"start":afternoon_first_ts , "target": afternoondf.iso.values}],
    freq = "1H"
)

eveningtrain = ListDataset(
    [{"start":evening_first_ts , "target": eveningdf.iso.values}],
    freq = "1H"
)

earlymorningtrain = ListDataset(
    [{"start":earlymorning_first_ts , "target": earlymorningdf.iso.values}],
    freq = "1H"
)

mtest = morningdf[teststart:testend]
morningtest = ListDataset([{"start":mtest["DateTime"][teststart], "target": mtest.iso.values}], freq="1H")

atest = afternoondf[teststart:testend]
afternoontest = ListDataset([{"start":atest["DateTime"][teststart], "target": atest.iso.values}], freq="1H")

etest = eveningdf[teststart:testend]
eveningtest = ListDataset([{"start":etest["DateTime"][teststart], "target": etest.iso.values}], freq="1H")

emtest = earlymorningdf[teststart:testend]
earlymorningtest = ListDataset([{"start":emtest["DateTime"][teststart], "target": emtest.iso.values}], freq="1H")

# output: morningdf, earlymorningdf, afternoondf, eveningdf
# additional output: training and test sets (morningtrain, morningtest, afternoontrain, ... ) 
