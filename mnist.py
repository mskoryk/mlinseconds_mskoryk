# The MNIST database of handwritten digits. http://yann.lecun.com/exdb/mnist/
#
# In this problem you need to implement model that will learn to recognize
# handwritten digits
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
import argparse

class SolutionModel(nn.Module):
    def __init__(self, input_size, output_size, solution):
        super(SolutionModel, self).__init__()
        self.solution = solution
        self.conv1 = torch.nn.Conv2d(
            in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.act1  = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
       
        self.conv2 = torch.nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=5, padding=0)
        self.act2  = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn0 = torch.nn.BatchNorm1d(5 * 5 * 16, track_running_stats=False)
        self.fc1   = torch.nn.Linear(5 * 5 * 16, self.solution.hidden_size_1)        
        self.act3  = torch.nn.ReLU()
        self.bn1   = torch.nn.BatchNorm1d(self.solution.hidden_size_1, track_running_stats=False)
        
        self.fc2   = torch.nn.Linear(self.solution.hidden_size_1, self.solution.hidden_size_2)
        self.act4  = torch.nn.ReLU()
        self.bn2   = torch.nn.BatchNorm1d(self.solution.hidden_size_2, track_running_stats=False)
        
        self.fc3   = torch.nn.Linear(self.solution.hidden_size_2, output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)
        
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        
        x = self.bn0(x)
        x = self.fc1(x)
        x = self.act3(x)
        x = self.bn1(x)
        x = self.fc2(x)
        x = self.act4(x)
        x = self.bn2(x)
        x = self.fc3(x)        
        return x

    def calc_error(self, output, target):
        # for key in self.solution.target_to_ind.keys().sort().values:
        result = target.clone()#.to(self.solution.device)
        for key in self.solution.target_to_ind.keys():
            result[result == key] = self.solution.target_to_ind[key]
        result = torch.nn.CrossEntropyLoss()(output, result)
        return  result
    
    def calc_predict(self, output):
        result = torch.argmax(output, 1)
        # for key in self.solution.ind_to_target.keys().sort(descending=True).values:
        for key in self.solution.ind_to_target.keys():
            result[result == key] = self.solution.ind_to_target[key]
        
        return result


class Solution():
    def __init__(self):
        # NOTE: Network params
        self.lr = 0.0035
        self.batch_size = 256
        
        self.hidden_size_1 = 49
        self.hidden_size_2 = 29
        
        
        

        self.weight_init = False

        self.loss = nn.CrossEntropyLoss()                                
        self.grid_search = None
        # grid search will initialize this field
        self.iter = 0
        # This fields indicate how many times to run with same arguments
        self.iter_number = 10

    
    def train_model(self, train_data, train_target, context):
        time_left = context.get_timer().get_time_left()
        def init_normal(m):
            if type(m) == nn.Linear:
                nn.init.uniform_(m.weight)

        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.0)
        def weights_init_uniform_rule(m):
            classname = m.__class__.__name__
            # for every Linear layer in a model..
            if classname.find('Linear') != -1:
                # get the number of the inputs
                n = m.in_features
                y = 1.0/np.sqrt(n)

                m.weight.data.uniform_(-y, y)
                m.bias.data.fill_(0.0)

        # Uncommend next line to understand grid search
        if run_grid_search:
            self.grid_search_tutorial()
            
        time_left = context.get_timer().get_time_left()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # device = torch.device('cpu')
        self.device = device        
        
        time_left = context.get_timer().get_time_left()
        
        model = SolutionModel(train_data.size(1), len(train_target.unique()), self).to(device)
        print(device)
#         model = model.to(device)
        time_left = context.get_timer().get_time_left()

        
        # recode the targets to [0, ..., C-1] labels
        target_to_ind = {}      
        ind_to_target = {}

        for i, t in enumerate(train_target.unique().sort().values):
            target_to_ind[t.item()] = i
            ind_to_target[i] = t.item()  
        
        self.target_to_ind = target_to_ind
        self.ind_to_target = ind_to_target
        
        time_left = context.get_timer().get_time_left()
        
        step = 0
        model.train()
        if self.weight_init:
            model.apply(weights_init_uniform_rule)             
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        
        batches = int(train_data.shape[0] // self.batch_size)
        
        good_count = 0
        good_streak = 7
        good_percent=0.98
        while True:
            ind = step % batches
            start_ind = self.batch_size * ind
            end_ind = self.batch_size * (ind + 1)
            data = train_data[start_ind:end_ind]
            target = train_target[start_ind:end_ind]
            data = data.to(device)
            target = target.to(device)

            # model.parameters()...gradient set to zero
            optimizer.zero_grad()
            # evaluate model => model.forward(data)
            output = model(data)
            
            predict = model.calc_predict(output)
            # Number of correct predictions
            correct = predict.eq(target.view_as(predict)).long().sum().item()
            # Total number of needed predictions
            total = target.view(-1).size(0)
            # calculate error
#             if correct / total > good_percent:
#               good_count += 1
#             else:
#               good_count = 0
            
#             if good_count >= good_streak:
#               break
            error = model.calc_error(output, target)            
            # calculate deriviative of model.forward() and put it in model.parameters()...gradient
            error.backward()
            # update model: model.parameters() -= lr * gradient
            optimizer.step()
            step += 1

            time_left = context.get_timer().get_time_left()
            # No more time left, stop training
            if time_left < 0.1:
                print(f'Failed step: {step}, loss: {error.item()} ')
                break

            time_limit = 0.1
            if time_left < time_limit:
                break

            # if step % batches == 0:
            #     self.print_stats(context.step, error, correct, total)

            #     with torch.no_grad():
            #         output = model(train_data)
            #         diff = (output-train_target).abs()                    
            #         if diff.max() <  self.threshold_error:                        
            #             break     
            context.increase_step()                    
                
        if self.grid_search:
            res = context.step if correct == total else 1000000
            self.grid_search.add_result('steps', res)
        
        return model.to(torch.device('cpu'))

    def print_stats(self, step, error, correct, total):
        if step % 10 == 0:
            print("Step = {} Correct = {}/{} Error = {}".format(step, correct, total, error.item()))

    def grid_search_tutorial(self):
        # During grid search every possible combination in field_grid, train_model will be called
        # iter_number times. This can be used for automatic parameters tunning.
        if self.grid_search:
            # print("[HelloXor] learning_rate={} iter={}".format(self.learning_rate, self.iter))
            self.grid_search.add_result('iter', self.iter)
            if self.iter == self.iter_number-1:
                # print("[HelloXor] chose_str={}".format(self.grid_search.choice_str))
                # print("[HelloXor] iters={}".format(self.grid_search.get_results('iter')))
                stats = self.grid_search.get_stats('iter')
                # print("[HelloXor] Mean={} Std={}".format(stats[0], stats[1]))
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
        self.size_limit = 25000
        self.test_limit = 0.95

class DataProvider:
    def __init__(self):
        self.number_of_cases = 10
        print("Start data loading...")
        train_dataset = torchvision.datasets.MNIST(
            './data/data_mnist', train=True, download=True,
            transform=torchvision.transforms.ToTensor()
        )
        trainLoader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset))
        test_dataset = torchvision.datasets.MNIST(
            './data/data_mnist', train=False, download=True,
            transform=torchvision.transforms.ToTensor()
        )
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset))
        self.train_data = next(iter(trainLoader))
        self.test_data = next(iter(test_loader))
        print("Data loaded")

    def select_data(self, data, digits):
        data, target = data
        mask = target == -1
        for digit in digits:
            mask |= target == digit
        indices = torch.arange(0,mask.size(0))[mask].long()
        return (torch.index_select(data, dim=0, index=indices), target[mask])

    def create_case_data(self, case):
        if case == 1:
            digits = [0,1]
        elif case == 2:
            digits = [8, 9]
        else:
            digits = [i for i in range(10)]

        description = "Digits: "
        for ind, i in enumerate(digits):
            if ind > 0:
                description += ","
            description += str(i)
        train_data = self.select_data(self.train_data, digits)
        test_data = self.select_data(self.test_data, digits)
        return sm.CaseData(case, Limits(), train_data, test_data).set_description(description).set_output_size(10)

class Config:
    def __init__(self):
        self.max_samples = 1000

    def get_data_provider(self):
        return DataProvider()

    def get_solution(self):
        return Solution()

run_grid_search = False
# Uncomment next line if you want to run grid search
#run_grid_search = True
if run_grid_search:
    gs.GridSearch().run(Config(), case_number=1, random_order=False, verbose=False)
else:
    # this part is needed for GPU initialization (so that it doesn't take place
    # inside the first test)
    a = torch.tensor(np.ones((1, 10000)), dtype=torch.float32)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    a.to(device)
    # If you want to run specific case, put number here
    sm.SolutionManager().run(Config(), case_number=-1)
