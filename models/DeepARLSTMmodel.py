from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer

estimator = DeepAREstimator(freq="1H", prediction_length=30, trainer=Trainer(epochs=10,batch_size=10), cell_type='lstm')
predictor = estimator.train(training_data= training_data_neweng ) 
