# calculate trend

# output: trend array and an array that can be plotted to visualize the trend lines 

from scipy.stats import linregress

df1 = caiso_dataset # set dataframe 

# create trend 
trend_interval = 10

# a is just [1,2,3,4,5, ..., trend_interval] (the x-values)
a = list(range(1, trend_interval+1))

div_by_trend = 0

# array holding the trend
array_trend = []
temp_trend= []

plot_trend = [] # array of pts 
temp_plot_trend = []

for ind in df1.index: 
  if div_by_trend==trend_interval:
    div_by_trend=1
    # calculate slope
    #print('a,', a)
    #print('temptrend,', temp_trend)
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
    temp_trend.append(df1['iso'][ind]) # change 'val' to 'iso' 
    continue
  temp_trend.append(df1['iso'][ind])
  div_by_trend = div_by_trend+1
  
# also add for the last set of values
#for i in range(div_by_trend):
last_a = list(range(1, div_by_trend+1))
last_trend = linregress(last_a, temp_trend)
last_extend = [last_trend[0]]*div_by_trend
array_trend.extend(last_extend)
# plot
yintercept = temp_trend[0]
temp_plot_trend.append(yintercept)
for j in range(1, div_by_trend):
  temp_plot_trend.append(yintercept+ trend[0]*j)
plot_trend.extend(temp_plot_trend)

# output: array_trend and plot_trend
