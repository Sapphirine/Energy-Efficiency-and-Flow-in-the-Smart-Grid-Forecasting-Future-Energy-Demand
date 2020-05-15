import mxnet
from mxnet.gluon import nn, loss as gloss

import mxnet as mx
import mxnet.ndarray as nd
from mxnet import nd, autograd, gluon
from mxnet.gluon.data.vision import transforms

# L2 Loss
loss2 = gloss.L2Loss()

# sample data 
x = nd.ones((2,))
y = nd.ones((2,)) * 2
loss2(x, y)

# Huber loss
loss_huber = gloss.HuberLoss(rho=0.85) # threshold rho 

loss = gloss.SoftmaxCrossEntropyLoss()
x = nd.array([[1, 10], [8, 2]])
y = nd.array([0, 1])
loss(x, y)
