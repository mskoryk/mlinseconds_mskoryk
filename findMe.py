# There are random function from 8 inputs and X random inputs added.
# We split data in 2 parts, on first part you will train and on second
# part we will test
import time
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ..utils import solutionmanager as sm
from ..utils import gridsearch as gs
import numpy as np

class SolutionModel(nn.Module):
    def __init__(self, input_size, output_size, solution):
        super(SolutionModel, self).__init__()        
        self.weight_initialization = solution.weight_initialization
        self.output_weight_initialization = solution.output_weight_initialization
        self.input_size = input_size
        self.batch_norm = solution.batch_norm
        self.input_dropout = solution.input_dropout
        self.dropout = solution.dropout
        self.n_hidden_layers = solution.n_hidden_layers
        self.hidden_size = solution.hidden_size              
        # sm.SolutionManager.print_hint("Hint[1]: Increase hidden size")
        self.linear_layers = nn.ModuleList([nn.Linear(self.input_size if layer == 0 else self.hidden_size,
                                                self.hidden_size if layer != self.n_hidden_layers else output_size)
                                    for layer in range(self.n_hidden_layers + 1)])
        self.batch_norm_layers = nn.ModuleList([nn.BatchNorm1d(self.hidden_size, track_running_stats=False)
                                    for layer in range(self.n_hidden_layers)])

        self.dropout_layers = nn.ModuleList([torch.nn.Dropout(p=self.dropout) for layer in range(self.n_hidden_layers)])
        self.input_dropout_layer = torch.nn.Dropout(p=self.input_dropout)
        
        
        self.act = solution.act
        self.output_act = solution.output_act                

        if self.weight_initialization == 'normal':
            # print('normal init')
            for layer in self.linear_layers[:-1]:
                torch.nn.init.normal_(layer.weight, 0, np.sqrt(1. / layer.in_features))            
        elif self.weight_initialization == 'normal2':
            # print('2 * normal init')
            for layer in self.linear_layers[:-1]:
                torch.nn.init.normal_(layer.weight, 0, np.sqrt(2. / layer.in_features))            
        elif self.weight_initialization == 'xavier_normal':
            # print('xavier normal init')
            for layer in self.linear_layers[:-1]:
                torch.nn.init.xavier_normal(layer.weight)            
        elif self.weight_initialization == 'xavier_uniform':
            # print('xavier uniform init')
            for layer in self.linear_layers[:-1]:
                torch.nn.init.xavier_uniform_(layer.weight)            
        elif self.weight_initialization == 'kaiming_uniform':
            # print('kaiming uniform init')
            for layer in self.linear_layers[:-1]:
                torch.nn.init.kaiming_uniform_(layer.weight)            
        elif self.weight_initialization == 'kaiming_normal':
            # print('kaiming normal init')
            for layer in self.linear_layers[:-1]:
                torch.nn.init.kaiming_normal(layer.weight)        
        
        
        if self.output_weight_initialization ==  'normal':
            # print('output normal init')
            torch.nn.init.normal_(self.linear_layers[-1].weight, 0, np.sqrt(1. / self.linear_layers[-1].in_features))
        elif self.output_weight_initialization == 'normal2':
            # print('output 2 * normal init')
            torch.nn.init.normal_(self.linear_layers[-1].weight, 0, np.sqrt(2. / self.linear_layers[-1].in_features))
        elif self.output_weight_initialization == 'xavier_normal':
            # print('output xavier init')
            torch.nn.init.xavier_normal(self.linear_layers[-1].weight)
        elif self.output_weight_initialization == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(self.linear_layers[-1].weight)
        elif self.output_weight_initialization == 'kaiming_normal':
            torch.nn.init.kaiming_normal(self.linear_layers[-1].weight)
        elif self.output_weight_initialization == 'kaiming_uniform':
            torch.nn.init.kaiming_uniform_(self.linear_layers[-1].weight)                              
        self.loss = solution.loss        


    def forward(self, x):
        if self.input_dropout > 0:
            x = self.input_dropout_layer(x)
        for linear_layer, batchnorm_layer, dropout_layer in zip(self.linear_layers[:-1], self.batch_norm_layers, self.dropout_layers):
            x = linear_layer(x)
            if self.batch_norm:
                x = batchnorm_layer(x)
            x = self.act(x)
            if self.dropout > 0:
                x = dropout_layer(x)
        x = self.linear_layers[-1](x)
        x = self.output_act(x)
        return x

    def calc_error(self, output, target):
        # This is loss function
        if self.loss == 'mse':
            return ((output-target)**2).sum()
        elif self.loss == 'logloss':
            return -(target * torch.log(output) + (1 - target) * torch.log(1 - output)).sum()
        elif self.loss == 'BCELoss':
           return nn.BCELoss()(output,target)
        elif self.loss == 'L1Loss':
           return nn.L1Loss()(output,target)
        else:
            return ((output-target)**2).sum()
        

    def calc_predict(self, output):
        # Simple round output to predict value
        return output.round()

class Solution():
    def __init__(self):
        # Control speed of learning
        self.learning_rate = 0.001
        self.momentum = 0.9
        self.alpha = 0.999
        self.weight_decay = 0.002     
        self.batch_norm = True  
        self.act = torch.relu
        self.n_hidden_layers = 4
        self.output_act = torch.sigmoid
        self.hidden_size = 30 

        self.input_dropout = 0.0001
        self.dropout = 0.01

        self.optimizer = 'RMSProp'


        self.output_weight_initialization = 'default'
        self.weight_initialization = 'xavier_uniform'
        
        self.loss = 'BCELoss'
        # Control number of hidden neurons
        
        self.learning_rate_grid = [0.001, 0.005, 0.01, 0.05, 0.1]
        # self.input_dropout_grid = [0, 0.001, 0.01, 0.1]
        # self.dropout_grid = [0, 0.001, 0.01, 0.1]
        self.optimizer_grid = ['Adam', 'RMSProp']
        self.n_hidden_layers_grid = [2,3,4,5]
        # self.hidden_size_grid = [10, 20, 30, 50]
        self.weight_initialization_grid = ['kaiming_uniform', 'xavier_univorm']
        self.weight_decay_grid = [0, 0.00002, 0.002]
        # self.momentum_grid = [0, 0.9, 0.99]
        # self.alpha_grid = [0.6, 0.99, 0.999]              

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
        if self.optimizer == 'Adam':
            # optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, betas=(self.momentum, self.alpha))
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, betas=(self.momentum, self.alpha))
        else:
            # optimizer = optim.RMSprop(model.parameters(), lr=self.learning_rate, momentum=self.momentum,
            #                       alpha=self.alpha, weight_decay=self.weight_decay, eps=1e-08, centered=False)
            optimizer = optim.RMSprop(model.parameters(), lr=self.learning_rate, momentum=self.momentum,
                                  alpha=0.99, weight_decay=self.weight_decay, eps=1e-08, centered=False)
        # optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)        
        step = 0
        # while True:
        while step < 200:
            step += 1
            # Report step, so we know how many steps
            context.increase_step()
            # model.parameters()...gradient set to zero
            optimizer.zero_grad()
            # evaluate model => model.forward(data)
            output = model(train_data)
            # if x < 0.5 predict 0 else predict 1
            # model.eval()
            # output_eval = model(train_data)
            # predict = model.calc_predict(output_eval)
            # # Number of correct predictions
            # correct = predict.eq(train_target.view_as(predict)).long().sum().item()
            # # print("at step {} total number of correct: {}".format(step, correct))
            # # Total number of needed predictions
            # total = predict.view(-1).size(0)
            # model.train()
            # No more time left or learned everything, stop training
            time_left = context.get_timer().get_time_left()
            if time_left < 0.1:
            # if correct == total:
                break
            # calculate error            
            error = model.calc_error(output, train_target)
            # calculate deriviative of model.forward() and put it in model.parameters()...gradient
            error.backward()
            # print progress of the learning
            # self.print_stats(context.step, error, correct, total)
            # update model: model.parameters() -= lr * gradient
            optimizer.step()
        return model

    def print_stats(self, step, error, correct, total):
        if step % 1000 == 0:
            pass
            # print("Step = {} Correct = {}/{} Error = {}".format(step, correct, total, error.item()))

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
        self.size_limit = 1000000
        self.test_limit = 1.0

class DataProvider:
    def __init__(self):
        self.number_of_cases = 10

    def create_data(self, data_size, input_size, random_input_size, seed):
        torch.manual_seed(seed)
        function_size = 1 << input_size
        function_input = torch.ByteTensor(function_size, input_size)
        for i in range(function_input.size(0)):
            fun_ind = i
            for j in range(function_input.size(1)):
                input_bit = fun_ind&1
                fun_ind = fun_ind >> 1
                function_input[i][j] = input_bit
        function_output = torch.ByteTensor(function_size).random_(0, 2)

        if data_size % function_size != 0:
            raise "Data gen error"

        data_input = torch.ByteTensor(data_size, input_size).view(-1, function_size, input_size)
        target = torch.ByteTensor(data_size).view(-1, function_size)
        for i in range(data_input.size(0)):
            data_input[i] = function_input
            target[i] = function_output
        data_input = data_input.view(data_size, input_size)
        target = target.view(data_size)
        if random_input_size > 0:
            data_random = torch.ByteTensor(data_size, random_input_size).random_(0, 2)
            data = torch.cat([data_input, data_random], dim=1)
        else:
            data = data_input
        perm = torch.randperm(data.size(1))
        data = data[:,perm]
        perm = torch.randperm(data.size(0))
        data = data[perm]
        target = target[perm]
        return (data.float(), target.view(-1, 1).float())

    def create_case_data(self, case):
        data_size = 256*32
        input_size = 8
        random_input_size = min(32, (case-1)*4)

        data, target = self.create_data(2*data_size, input_size, random_input_size, case)
        return sm.CaseData(case, Limits(), (data[:data_size], target[:data_size]), (data[data_size:], target[data_size:])).set_description("{} inputs and {} random inputs".format(input_size, random_input_size))

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
    gs.GridSearch().run(Config(), case_number=1, random_order=False, verbose=False)
else:
    # If you want to run specific case, put number here
    sm.SolutionManager().run(Config(), case_number=-1)
