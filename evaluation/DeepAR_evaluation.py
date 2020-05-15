# accuracy of deepar estimator
from gluonts.evaluation.backtest import make_evaluation_predictions

#input: # test_dataset, predictor (defined in DeepARmodel.py) 

deepare_forecast_it, deepare_ts_it = make_evaluation_predictions(dataset= test_dataset, 
                                                                 predictor=predictor, num_samples=10)
deepare_forecasts = list(deepare_forecast_it)
deepare_tss = list(deepare_ts_it)

# output: dm:mean values of test window , dq: quantile of level 0.5 (can change this to other levels)

dm = deepare_forecasts[0].mean

dq = deepare_forecasts[0].quantile(0.5) 

# simple plot of predictions output: 
import matplotlib.pyplot as plt
a = list(range(30))
plt.plot( a, dm)
plt.show()

# accuracy metric evaluation:
from gluonts.evaluation import Evaluator

evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
agg_metrics, item_metrics = evaluator(iter(deepare_tss), iter(deepare_forecasts), num_series=len(test_dataset))

# output: examine metrics
item_metrics.head()
