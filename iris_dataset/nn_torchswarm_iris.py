from __future__ import absolute_import
try:
    # now it can reach class A of file_a.py in folder a 
    # by relative import
    from torchswarm.pso import ParicleSwarmOptimizer  
    from models.irismodel import Model
except (ModuleNotFoundError, ImportError) as e:
    print("{} fileure".format(type(e.name)))
else:
    print("Import succeeded")


import time
from sklearn.datasets import load_iris
import numpy as np
from torch.utils.data import Dataset, DataLoader
from keras.utils import to_categorical
import torch
from torch.autograd import Variable
import torch.nn.functional as F


# from torchswarm.particle_swarm_optimizaion import ParicleSwarmOptimizer


class BCELossWithPSO(torch.autograd.Function):  
    @staticmethod
    def forward(ctx , y, y_pred, sum_cr, eta, gbest):
        ctx.save_for_backward(y, y_pred)
        ctx.sum_cr = sum_cr
        ctx.eta = eta
        ctx.gbest = gbest
        return F.binary_cross_entropy(y,y_pred)

    @staticmethod
    def backward(ctx, grad_output):
        yy, yy_pred= ctx.saved_tensors
        sum_cr = ctx.sum_cr
        eta = ctx.eta
        grad_input = torch.neg((sum_cr/eta) * (ctx.gbest - yy))
        return grad_input, grad_output, None, None, None


#Load Iris
iris = load_iris()
X=iris.data
y=iris.target

y = to_categorical(y)

batch_size = 5

SwarmSize = 100


class BCELoss:
    def __init__(self, y):
        self.y = y
        self.fitness = torch.nn.BCELoss()
    def evaluate(self, x):
        # print(x, self.y)
        return self.fitness(x, self.y)




class PrepareData(Dataset):

    def __init__(self, X, y):
        if not torch.is_tensor(X):
            self.X = torch.from_numpy(X)
        if not torch.is_tensor(y):
            self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

ds = PrepareData(X=X, y=y)
ds = DataLoader(ds, batch_size=batch_size, shuffle=True)




model = Model(n_features=4, n_neurons=50)

cost_func = torch.nn.BCELoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)

num_epochs = 10

myloss = BCELossWithPSO.apply
p = ParicleSwarmOptimizer(batch_size, SwarmSize, 3)


losses = []
accs = []
for e in range(num_epochs):
    batch_losses=[]
    for ix, (_x, _y) in enumerate(ds):
        
        #=========make inpur differentiable=======================
        tic = time.monotonic()
        _x = Variable(_x).float()
        _y = Variable(_y, ).float()
        _y.requires_grad = False
        p = ParicleSwarmOptimizer(batch_size, SwarmSize, 3)
        p.optimize(BCELoss(_y))
        #========forward pass=====================================
        yhat = model(_x).float()
        for i in range(25):
            c1r1, c2r2, gbest = p.run_one_iter(verbosity=False)
            # print(gbest)
        # print("==========================")
        loss = myloss(yhat, _y, c1r1+c2r2, 0.1, gbest)
        acc = torch.eq(yhat.round(), _y).float().mean()# accuracy

        #=======backward pass=====================================
        optimizer.zero_grad() # zero the gradients on each pass before the update
        loss.backward() # backpropagate the loss through the model
        optimizer.step() # update the gradients w.r.t the loss

        accs.append(acc.item())
        toc = time.monotonic()
        batch_losses.append(loss.item())
        print("Batch : {}| Loss: {} | Time: {}".format(ix, loss.item(), toc-tic))
    losses.append(sum(batch_losses)/30)
    if e % 1 == 0:
        print("[{}/{}], loss: {} acc: {}".format(e,
        num_epochs, np.round(sum(batch_losses)/30, 3), np.round(acc.item(), 3)))

print(losses)