# You need to learn a function with n inputs.
# For given number of inputs, we will generate random function.
# Your task is to learn it
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
        # sm.SolutionManager.print_hint("Hint[1]: Increase hidden size")
        self.hidden_size1 = solution.hidden_size              
        self.hidden_size2 = solution.hidden_size
        self.hidden_size3 = solution.hidden_size
        self.hidden_size4 = solution.hidden_size
        self.hidden_size5 = solution.hidden_size
        # self.hidden_size6 = solution.hidden_size
        # self.hidden_size7 = solution.hidden_size
        # self.hidden_size8 = solution.hidden_size
        # self.hidden_size9 = solution.hidden_size
        # self.hidden_size10 = solution.hidden_size

        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(self.hidden_size1, track_running_stats=False)
            self.batch_norm2 = nn.BatchNorm1d(self.hidden_size2, track_running_stats=False)
            self.batch_norm3 = nn.BatchNorm1d(self.hidden_size3, track_running_stats=False)
            self.batch_norm4 = nn.BatchNorm1d(self.hidden_size4, track_running_stats=False)
            self.batch_norm5 = nn.BatchNorm1d(self.hidden_size5, track_running_stats=False)
        
        self.act1 = solution.act1
        self.act2 = solution.act1
        self.act3 = solution.act1
        self.act4 = solution.act1
        self.act5 = solution.act1
        # self.act6 = solution.act6
        # self.act7 = solution.act7
        # self.act8 = solution.act8
        # self.act9 = solution.act9
        # self.act10 = solution.act10
        self.act6 = solution.act6
        

        self.linear1 = nn.Linear(self.input_size, self.hidden_size1)        
        self.linear2 = nn.Linear(self.hidden_size1, self.hidden_size2)
        self.linear3 = nn.Linear(self.hidden_size2, self.hidden_size3)
        self.linear4 = nn.Linear(self.hidden_size3, self.hidden_size4)
        self.linear5 = nn.Linear(self.hidden_size4, self.hidden_size5)
        # self.linear6 = nn.Linear(self.hidden_size5, self.hidden_size6)
        # self.linear7 = nn.Linear(self.hidden_size6, self.hidden_size7)
        # self.linear8 = nn.Linear(self.hidden_size7, self.hidden_size8)
        # self.linear9 = nn.Linear(self.hidden_size8, self.hidden_size9)
        # self.linear10 = nn.Linear(self.hidden_size9, self.hidden_size10)
        self.linear6 = nn.Linear(self.hidden_size5, output_size)                
        if self.weight_initialization == 'normal':
            # print('normal init')
            torch.nn.init.normal_(self.linear1.weight, 0, np.sqrt(1. / self.linear1.in_features))
            torch.nn.init.normal_(self.linear2.weight, 0, np.sqrt(1. / self.linear2.in_features))
            torch.nn.init.normal_(self.linear3.weight, 0, np.sqrt(1. / self.linear3.in_features))
            torch.nn.init.normal_(self.linear4.weight, 0, np.sqrt(1. / self.linear4.in_features))
            torch.nn.init.normal_(self.linear5.weight, 0, np.sqrt(1. / self.linear5.in_features))
            # torch.nn.init.normal_(self.linear6.weight, 0, np.sqrt(1. / self.linear6.in_features))
            # torch.nn.init.normal_(self.linear7.weight, 0, np.sqrt(1. / self.linear7.in_features))
            # torch.nn.init.normal_(self.linear8.weight, 0, np.sqrt(1. / self.linear8.in_features))
            # torch.nn.init.normal_(self.linear9.weight, 0, np.sqrt(1. / self.linear9.in_features))
            # torch.nn.init.normal_(self.linear10.weight, 0, np.sqrt(1. / self.linear10.in_features))
        elif self.weight_initialization == 'normal2':
            # print('2 * normal init')
            torch.nn.init.normal_(self.linear1.weight, 0, np.sqrt(2. / self.linear1.in_features))
            torch.nn.init.normal_(self.linear2.weight, 0, np.sqrt(2. / self.linear2.in_features))
            torch.nn.init.normal_(self.linear3.weight, 0, np.sqrt(2. / self.linear3.in_features))
            torch.nn.init.normal_(self.linear4.weight, 0, np.sqrt(2. / self.linear4.in_features))
            torch.nn.init.normal_(self.linear5.weight, 0, np.sqrt(2. / self.linear5.in_features))
            # torch.nn.init.normal_(self.linear6.weight, 0, np.sqrt(2. / self.linear6.in_features))
            # torch.nn.init.normal_(self.linear7.weight, 0, np.sqrt(2. / self.linear7.in_features))
            # torch.nn.init.normal_(self.linear8.weight, 0, np.sqrt(2. / self.linear8.in_features))
            # torch.nn.init.normal_(self.linear9.weight, 0, np.sqrt(2. / self.linear9.in_features))
            # torch.nn.init.normal_(self.linear10.weight, 0, np.sqrt(2. / self.linear10.in_features))
        elif self.weight_initialization == 'xavier_normal':
            # print('xavier init')
            torch.nn.init.xavier_normal(self.linear1.weight)
            torch.nn.init.xavier_normal(self.linear2.weight)
            torch.nn.init.xavier_normal(self.linear3.weight)
            torch.nn.init.xavier_normal(self.linear4.weight)
            torch.nn.init.xavier_normal(self.linear5.weight)
            # torch.nn.init.xavier_normal(self.linear6.weight)
            # torch.nn.init.xavier_normal(self.linear7.weight)
            # torch.nn.init.xavier_normal(self.linear8.weight)
            # torch.nn.init.xavier_normal(self.linear9.weight)
            # torch.nn.init.xavier_normal(self.linear10.weight)
        elif self.weight_initialization == 'xavier_uniform':
            # print('xavier init')
            torch.nn.init.xavier_uniform(self.linear1.weight)
            torch.nn.init.xavier_uniform(self.linear2.weight)
            torch.nn.init.xavier_uniform(self.linear3.weight)
            torch.nn.init.xavier_uniform(self.linear4.weight)
            torch.nn.init.xavier_uniform(self.linear5.weight)
            # torch.nn.init.xavier_uniform(self.linear6.weight)
            # torch.nn.init.xavier_uniform(self.linear7.weight)
            # torch.nn.init.xavier_uniform(self.linear8.weight)
            # torch.nn.init.xavier_uniform(self.linear9.weight)
            # torch.nn.init.xavier_uniform(self.linear10.weight)
        elif self.weight_initialization == 'kaiming_uniform':
            # print('xavier init')
            torch.nn.init.kaiming_uniform(self.linear1.weight)
            torch.nn.init.kaiming_uniform(self.linear2.weight)
            torch.nn.init.kaiming_uniform(self.linear3.weight)
            torch.nn.init.kaiming_uniform(self.linear4.weight)
            torch.nn.init.kaiming_uniform(self.linear5.weight)
            # torch.nn.init.kaiming_uniform(self.linear6.weight)
            # torch.nn.init.kaiming_uniform(self.linear7.weight)
            # torch.nn.init.kaiming_uniform(self.linear8.weight)
            # torch.nn.init.kaiming_uniform(self.linear9.weight)
            # torch.nn.init.kaiming_uniform(self.linear10.weight)
        elif self.weight_initialization == 'kaiming_normal':
            # print('xavier init')
            torch.nn.init.kaiming_normal(self.linear1.weight)
            torch.nn.init.kaiming_normal(self.linear2.weight)
            torch.nn.init.kaiming_normal(self.linear3.weight)
            torch.nn.init.kaiming_normal(self.linear4.weight)
            torch.nn.init.kaiming_normal(self.linear5.weight)
            # torch.nn.init.kaiming_normal(self.linear6.weight)
            # torch.nn.init.kaiming_normal(self.linear7.weight)
            # torch.nn.init.kaiming_normal(self.linear8.weight)
            # torch.nn.init.kaiming_normal(self.linear9.weight)
            # torch.nn.init.kaiming_normal(self.linear10.weight)
        
        
        
        
        if self.output_weight_initialization ==  'normal':
            # print('output normal init')
            torch.nn.init.normal_(self.linear6.weight, 0, np.sqrt(1. / self.linear6.in_features))
        elif self.output_weight_initialization == 'normal2':
            # print('output 2 * normal init')
            torch.nn.init.normal_(self.linear6.weight, 0, np.sqrt(2. / self.linear6.in_features))
        elif self.output_weight_initialization == 'xavier_normal':
            # print('output xavier init')
            torch.nn.init.xavier_normal(self.linear6.weight)
        elif self.output_weight_initialization == 'xavier_uniform':
            torch.nn.init.xavier_uniform(self.linear6.weight)
        elif self.output_weight_initialization == 'kaiming_normal':
            torch.nn.init.kaiming_normal(self.linear6.weight)
        elif self.output_weight_initialization == 'kaiming_uniform':
            torch.nn.init.kaiming_uniform(self.linear6.weight)
        
                      
        self.loss = solution.loss        


    def forward(self, x):
        x = self.linear1(x)
        if self.batch_norm:
            x = self.batch_norm1(x)
        x = self.act1(x)
        x = self.linear2(x)
        if self.batch_norm:
            x = self.batch_norm2(x)
        x = self.act2(x)  
        x = self.linear3(x)
        if self.batch_norm:
            x = self.batch_norm3(x)
        x = self.act3(x)  
        x = self.linear4(x)
        if self.batch_norm:
            x = self.batch_norm4(x)
        x = self.act4(x)                    
        x = self.linear5(x)
        if self.batch_norm:
            x = self.batch_norm5(x)
        x = self.act5(x)
        # x = self.linear6(x)
        # x = self.act6(x)
        # x = self.linear7(x)
        # x = self.act7(x)
        # x = self.linear8(x)
        # x = self.act8(x)
        # x = self.linear9(x)
        # x = self.act9(x)
        # x = self.linear10(x)
        # x = self.act10(x)
        x = self.linear6(x)
        x = self.act6(x)


                  

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
        self.learning_rate = 0.023
        self.momentum = 0
        self.weight_decay = 0      
        self.batch_norm = True  
        self.act1 = torch.relu
        self.act2 = torch.relu
        self.act3 = torch.relu
        self.act4 = torch.relu
        self.act5 = torch.relu
        # self.act6 = torch.relu
        # self.act7 = torch.relu
        # self.act8 = torch.relu
        # self.act9 = torch.relu
        # self.act10 = torch.relu
        self.act6 = torch.sigmoid
        self.hidden_size = 45 


        self.output_weight_initialization = 'default'
        self.weight_initialization = 'kaiming_uniform'
        
        self.loss = 'BCELoss'
        # Control number of hidden neurons
        
        # self.learning_rate_grid = np.linspace(0.01, 0.04, 10)
        # self.learning_rate_grid = np.linspace(0.005, 0.015, 11)
        self.learning_rate_grid = np.linspace(0.015, 0.025, 11)
        # self.loss_grid = ['logloss']
        self.loss_grid = ['logloss', 'BCELoss']
        self.weight_initialization_grid = ['kaiming_uniform']
        # self.weight_initialization_grid = ['xavier_normal', 'kaiming_uniform', 'xavier_uniform', 'kaiming_normal']
        self.output_weight_initialization_grid = [ 'default']
        # self.act1_grid = [torch.relu, torch.tanh, torch.nn.functional.relu6]
        self.act1_grid = [torch.relu]
        # self.act1_grid = [torch.tanh]
        # self.momentum_grid = [0.88, 0.9, 0.92]
        # self.weight_decay_grid = [0, 0.002]
        # self.hidden_size_grid = [10, 20, 30]
        

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
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        # optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)        
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
        self.size_limit = 10000
        self.test_limit = 1.0

class DataProvider:
    def __init__(self):
        self.number_of_cases = 10

    def create_data(self, input_size, seed):
        random.seed(seed)
        data_size = 1 << input_size
        data = torch.FloatTensor(data_size, input_size)
        target = torch.FloatTensor(data_size)
        for i in range(data_size):
            for j in range(input_size):
                input_bit = (i>>j)&1
                data[i,j] = float(input_bit)
            target[i] = float(random.randint(0, 1))
        return (data, target.view(-1, 1))

    def create_case_data(self, case):
        input_size = min(3+case, 7)
        data, target = self.create_data(input_size, case)
        return sm.CaseData(case, Limits(), (data, target), (data, target)).set_description("{} inputs".format(input_size))


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
