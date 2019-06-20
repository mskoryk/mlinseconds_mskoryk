# HelloXor is a HelloWorld of Machine Learning.
import time
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import PolynomialFeatures
from ..utils import solutionmanager as sm
from ..utils import gridsearch as gs
import numpy as np

class SolutionModel(nn.Module):
    def __init__(self, input_size, output_size, solution):
        super(SolutionModel, self).__init__()        
        self.input_size = input_size
        # sm.SolutionManager.print_hint("Hint[1]: Increase hidden size")
        self.hidden_size = solution.hidden_size              
        self.linear1 = nn.Linear(self.input_size, self.hidden_size)
        self.act1 = solution.act1
        self.linear2 = nn.Linear(self.hidden_size, output_size)
        self.act2 = solution.act2
        self.coef1 = solution.coef1
        self.coef2 = solution.coef2
        self.loss = solution.loss

    def forward(self, x):
        # This is the best solution with 7.0 average steps (MSE and learning rate=1, weight_decay=0.002, optimizer Adam)
        # x = self.linear1(x)
        # x = torch.relu(x / 5)
        # x = self.linear2(x)
        # x = torch.sigmoid(x / 5)

        x = self.linear1(x)
        x = self.act1(x / self.coef1)
        x = self.linear2(x)
        x = self.act2(x / self.coef2)
        return x

    def calc_error(self, output, target):
        # This is loss function
        if self.loss == 'mse':
            return ((output-target)**2).sum()
        elif self.loss == 'logloss':
            return -(target * torch.log(output) + (1 - target) * torch.log(1 - output)).sum()
        else:
            return ((output-target)**2).sum()

    def calc_predict(self, output):
        # Simple round output to predict value
        return output.round()

class Solution():
    def __init__(self):
        # Control speed of learning
        self.learning_rate = 1
        self.momentum = 0.99
        self.weight_decay = 0
        self.coef1 = 1
        self.coef2 = 1
        self.act1 = torch.relu
        self.act2 = torch.sigmoid
        self.hidden_size = 24 
        self.loss = 'mse'
        # Control number of hidden neurons

        # Grid search settings, see grid_search_tutorial
        ## first grid search
        ## BEST PARAMS are
        ## act1-relu hidden_size-24 learning_rate-1 loss-mse momentum-0.99 weight_decay-0
        ## average number of steps is 4.8
        # self.learning_rate_grid = [0.1, 0.5, 0.7, 1]        
        # self.hidden_size_grid = [10, 24]      
        # self.momentum_grid = [0, 0.9, 0.99]  
        # self.act1_grid = [torch.tanh, torch.relu, torch.sigmoid]
        # self.weight_decay_grid = [0, 0.002]
        # self.loss_grid = ['mse', 'logloss']

        ## second grid search
        ## showed no improvement -- stick with current solution
        self.learning_rate_grid = [0.5, 1, 2, 5]        
        self.coef1_grid = [1, 0.1, 5, 10]
        self.coef2_grid = [1, 0.1, 5, 10]
        

        # grid search will initialize this field
        self.grid_search = None
        # grid search will initialize this field
        self.iter = 0
        # This fields indicate how many times to run with same arguments
        self.iter_number = 10

    # Return trained model
    def train_model(self, train_data, train_target, context):
        # Uncommend next line to understand grid search
#        return self.grid_search_tutorial()
        # Model represent our neural network
        model = SolutionModel(train_data.size(1), train_target.size(1), self)
        # Optimizer used for training neural network
        # sm.SolutionManager.print_hint("Hint[2]: Learning rate is too small", context.step)
        # optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
        while True:
            # Report step, so we know how many steps
            context.increase_step()
            # model.parameters()...gradient set to zero
            optimizer.zero_grad()
            # evaluate model => model.forward(data)
            output = model(train_data)
            # if x < 0.5 predict 0 else predict 1
            predict = model.calc_predict(output)
            # Number of correct predictions
            correct = predict.eq(train_target.view_as(predict)).long().sum().item()
            # Total number of needed predictions
            total = predict.view(-1).size(0)
            # No more time left or learned everything, stop training
            time_left = context.get_timer().get_time_left()
            if time_left < 0.1 or correct == total:
            # if correct == total:
                break
            # calculate error
            error = model.calc_error(output, train_target)
            # calculate deriviative of model.forward() and put it in model.parameters()...gradient
            error.backward()
            # print progress of the learning
            self.print_stats(context.step, error, correct, total)
            # update model: model.parameters() -= lr * gradient
            optimizer.step()
        return model

    def print_stats(self, step, error, correct, total):
        if step % 1000 == 0:
            print("Step = {} Correct = {}/{} Error = {}".format(step, correct, total, error.item()))

    def grid_search_tutorial(self):
        # During grid search every possible combination in field_grid, train_model will be called
        # iter_number times. This can be used for automatic parameters tunning.
        if self.grid_search:
            print("[HelloXor] learning_rate={} iter={}".format(self.learning_rate, self.iter))
            self.grid_search.add_result('iter', self.iter)
            if self.iter == self.iter_number-1:
                print("[HelloXor] chose_str={}".format(self.grid_search.choice_str))
                print("[HelloXor] iters={}".format(self.grid_search.get_results('iter')))
                stats = self.grid_search.get_stats('iter')
                print("[HelloXor] Mean={} Std={}".format(stats[0], stats[1]))
        else:
            print("Enable grid search: See run_grid_search in the end of file")
            exit(0)
###
###
### Don't change code after this line
###
###
class Limits:
    def __init__(self):
        self.time_limit = 2.0
        self.size_limit = 100
        self.test_limit = 1.0

class DataProvider:
    def __init__(self):
        self.number_of_cases = 10

    def create_data(self):
        data = torch.FloatTensor([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0]
            ])
        target = torch.FloatTensor([
            [0.0],
            [1.0],
            [1.0],
            [0.0]
            ])
        return (data, target)

    def create_case_data(self, case):
        data, target = self.create_data()
        return sm.CaseData(case, Limits(), (data, target), (data, target))

class Config:
    def __init__(self):
        self.max_samples = 10000

    def get_data_provider(self):
        return DataProvider()

    def get_solution(self):
        return Solution()

run_grid_search = False
# Uncomment next line if you want to run grid search
# run_grid_search = True
if run_grid_search:
    gs.GridSearch().run(Config(), case_number=1, random_order=False, verbose=True)
else:
    # If you want to run specific case, put number here
    sm.SolutionManager().run(Config(), case_number=-1)
