# custom gluonts model
from gluonts.model.estimator import GluonEstimator
from gluonts.model.predictor import Predictor, RepresentableBlockPredictor
from gluonts.core.component import validated
from gluonts.trainer import Trainer 
from gluonts.support.util import copy_parameters
from gluonts.transform import ExpectedNumInstanceSampler, Transformation, InstanceSplitter
from mxnet.gluon import HybridBlock 
from mxnet import gluon
from gluonts.dataset.field_names import FieldName

from mxnet.gluon import nn, loss as gloss
import mxnet as mx
import mxnet.ndarray as nd
from mxnet import nd, autograd, gluon
from mxnet.gluon.data.vision import transforms

# input: predlen set as 6: keep it as the same as pred_length 
# input: training data ListDataset, here it is set as morningtrain

class MyNetwork(gluon.HybridBlock):
    def __init__(self, prediction_length, num_cells, **kwargs):
        super().__init__(**kwargs)
        self.prediction_length = prediction_length
        self.num_cells = num_cells
    
        with self.name_scope():
            # Set up a 4 layer neural network that directly predicts the target values
            self.nn = mx.gluon.nn.HybridSequential()
            self.nn.add(mx.gluon.nn.Dense(units=self.num_cells, activation='relu'))
            self.nn.add(mx.gluon.nn.Dense(units=self.num_cells, activation='relu'))
            self.nn.add(mx.gluon.nn.Dense(units=self.num_cells, activation='relu'))
            self.nn.add(mx.gluon.nn.Dense(units=self.prediction_length, activation='softrelu'))

class MyTrainNetwork1(MyNetwork):    
    def hybrid_forward(self, F, past_target, future_target): # add input 
        prediction = self.nn(past_target)
        # calculate L1 loss with the future_target to learn the median

        # change the loss 
        plen = len(prediction)
        
        predlen = 6
        xneg = choose_xneg(predlen)

        # compare each sample with other (prediction and future target)
        array_trip_loss =[]
        for i in range(0, plen):
          pred_i = prediction[i]
          ft_i = future_target[i]
          triploss_i = compute_triplet_loss(ft_i, xneg, pred_i)
          array_trip_loss.append(triploss_i)

        #triploss= tloss(nd.array(future_target), nd.array(xneg), nd.array(prediction))

        #print("triplet loss:", triploss)
        
        #return (prediction - future_target).abs().mean(axis=-1)
        #print("hloss", hloss)
        #return hloss
        return array_trip_loss

class MyPredNetwork1(MyTrainNetwork1):
    # The prediction network only receives past_target and returns predictions
    def hybrid_forward(self, F, past_target):
        prediction = self.nn(past_target)
        return prediction.expand_dims(axis=1)

class MyEstimator1(GluonEstimator):
  @validated()
  def __init__(
      self,
      prediction_length:int,
      context_length:int,
      freq:str,
      num_cells:int,
      trainer: Trainer=Trainer()
  ) -> None:
      super().__init__(trainer=trainer)
      self.prediction_length = prediction_length
      self.context_length = context_length
      self.freq = freq
      self.num_cells = num_cells
  def create_transformation(self):
    return InstanceSplitter(
                    target_field=FieldName.TARGET,
                    is_pad_field=FieldName.IS_PAD,
                    start_field=FieldName.START,
                    forecast_start_field=FieldName.FORECAST_START,
                    train_sampler=ExpectedNumInstanceSampler(num_instances=1),
                    past_length=self.context_length,
                    future_length=self.prediction_length
                )
  def create_training_network(self) -> MyTrainNetwork1:
    return MyTrainNetwork1(
        prediction_length = self.prediction_length,
        num_cells= self.num_cells
    )
  def create_predictor(self, transformation:Transformation, trained_network: HybridBlock) -> Predictor:
    prediction_network = MyPredNetwork1(
        prediction_length= self.prediction_length,
        num_cells = self.num_cells
    )

    copy_parameters(trained_network, prediction_network)

    return RepresentableBlockPredictor(
        input_transform=transformation,
        prediction_net=prediction_network,
        batch_size = self.trainer.batch_size,
        freq = self.freq,
        prediction_length = self.prediction_length,
        ctx=self.trainer.ctx
    )
  
import mxnet as mx
pred_length = 6

estimator1 = MyEstimator1(prediction_length= pred_length, context_length=2*pred_length, freq="1H",num_cells=10, trainer=Trainer(ctx="cpu", epochs=5, learning_rate=1e-3, hybridize=False, num_batches_per_epoch=20))

predictor1 = estimator1.train(morningtrain)
