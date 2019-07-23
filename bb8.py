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
import matplotlib.pyplot as plt

class SolutionModel(nn.Module):
    def __init__(self, input_size, output_size, solution):
        super(SolutionModel, self).__init__()
        self.solution = solution
        self.hidden_size = solution.hidden_size

        self.i2h = nn.Linear(input_size + self.hidden_size, self.hidden_size)
        self.bn_i2h = nn.BatchNorm1d(self.hidden_size, track_running_stats=False)
        self.h2h = nn.Linear(self.hidden_size, self.hidden_size)
        self.bn_h2h = nn.BatchNorm1d(self.hidden_size, track_running_stats=False)
        self.i2o = nn.Linear(input_size + self.hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

        torch.nn.init.xavier_normal_(self.i2h.weight) 
        torch.nn.init.xavier_normal_(self.h2h.weight) 
        torch.nn.init.xavier_normal_(self.i2o.weight) 

    def forward_pass(self, input, hidden):
        combined = torch.cat((input.float(), hidden), 1)
        hidden = self.i2h(combined)
        if self.solution.batch_norm:
            hidden = self.bn_i2h(hidden)
        hidden = nn.ReLU()(hidden)
        combined = torch.cat((input.float(), hidden), 1)
        output = self.i2o(combined)
        # output = self.softmax(output)
        return output, hidden
      
    def forward(self, input):
        # input = input.unsqueeze(2)
        hidden = self.initHidden(input.size(0))
        for i in range(input.size(1)):
            output, hidden = self.forward_pass(input[:, i].unsqueeze(1), hidden)
        
        return output

    def initHidden(self, n_samples):
        # return torch.zeros(n_samples, 1, self.hidden_size)
        return torch.zeros(n_samples, self.hidden_size).to(self.solution.device)

    
        

    def calc_error(self, output, target):
        # result = torch.nn.NLLLoss()(output.squeeze(1), target.squeeze(1).long())
        result = torch.nn.CrossEntropyLoss()(output.squeeze(1), target.squeeze(1).long())
        return  result
    
    def calc_predict(self, output):
        return torch.argmax(output, 1).float()


class Solution():
    def __init__(self):
        # NOTE: Network params
        self.lr = 0.001
        self.step_size = 5
        self.gamma = 0.65
        self.momentum = 0.0
        self.optimizer = 'Adam'
        self.batch_size = 128       
        self.hidden_size = 20
        self.batch_norm = False
        
        self.hidden_size_grid = [2, 5, 10, 20, 50]
        self.learning_rate_grid = [0.0001, 0.001, 0.01, 0.1]
        self.gamma_grid = [0.5, 0.7, 0.9]
                                     
        self.grid_search = None
        # grid search will initialize this field
        self.iter = 0
        # This fields indicate how many times to run with same arguments
        self.iter_number = 1

    
    def train_model(self, train_data, train_target, context):  
        if train_data.size(1) < 20:
          self.hidden_size = 40
          self.lr = 0.02
          self.step_size = 10
          self.gamma = 0.8
          self.batch_size = 128
        elif train_data.size(1) < 40:
          self.hidden_size = 40
          self.lr = 0.002
          self.step_size = 10
          self.gamma = 0.8
          self.batch_size = 128
        elif train_data.size(1) < 90:
          self.hidden_size = 40
          self.lr = 0.002
          self.step_size = 10
          self.gamma = 0.1
          self.batch_size = 128
        elif train_data.size(1) < 150:
          self.hidden_size = 20
          self.lr = 0.005
          self.step_size = 10
          self.gamma = 0.65
          self.batch_size = 128
        time_left = context.get_timer().get_time_left()
        # Uncommend next line to understand grid search
        if run_grid_search:
            self.grid_search_tutorial()
            
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#         print(device)
        self.device = device
        
        train_data = train_data.float()
        train_data = train_data.to(device)
        train_target = train_target.to(device)
        
        model = SolutionModel(1, 2, self).to(device)
        time_left = context.get_timer().get_time_left()
        
        

        step = 0
        model.train()
        if self.optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=self.lr)
        else:
            optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
        
       
        batches = int(train_data.shape[0] // self.batch_size)
        
        
        
        # good_count = 0
        # good_streak = 7
        # good_percent=0.98
        train_loss_history = []
        while True:
            ind = step % batches
            # get the results of epoch
            if ind == 0:
                scheduler.step()
                with torch.no_grad():
                    epoch_loss = model.calc_error(model(train_data), train_target)
                    train_loss_history.append(epoch_loss)
                    if epoch_loss < 0.45:
                        break
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
#                 print(f'Failed step: {step}, loss: {error.item()} ')
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
#         print("plotting...")
#         print(train_loss_history)
#         plt.plot(train_loss_history)
#         plt.show()
        model.solution.device = torch.device('cpu')
#         print("model.solution.device: {}". format(model.solution.device))
        return model.to(torch.device('cpu'))
#         return model

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
        self.size_limit = 1000000
        self.test_limit = 0.8

# There are 2 languages.
class Language:
    def __init__(self, states_count, letters_count):
        self.states_count = states_count
        with torch.no_grad():
            self.state_to_state_prob = torch.FloatTensor(states_count, states_count).uniform_()
            self.state_to_state_prob = self.state_to_state_prob/torch.sum(self.state_to_state_prob, dim=1).view(-1,1)
            self.state_to_state_cum_prob = torch.cumsum(self.state_to_state_prob, dim=1)

    def balance(self, sentences):
        letters_id, counts = torch.unique(sentences.view(-1), sorted=True, return_counts=True)
        perm = torch.randperm(letters_id.size(0))
        letters_id = letters_id[perm]
        counts = counts[perm]
        total = counts.sum().item()
        x = torch.ByteTensor(total+1).zero_()
        x[0] = 1
        xs = [x]
        for letter_id, count in zip(letters_id, counts):
            cc = count.item()
            nx = xs[-1].clone()
            nx[cc:][xs[-1][:-cc]] = 1
            xs.append(nx)
        best_balance = total//2
        while xs[-1][best_balance].item() == 0:
            best_balance -= 1
        #if best_balance != total//2:
        #    print("UNBALANCED")
        current_balance = best_balance
        balance_set = [False for _ in range(letters_id.size(0))]
        last_ind = len(xs)-1
        while current_balance != 0:
            while xs[last_ind-1][current_balance].item() == 1:
                last_ind -= 1
            balance_set[last_ind-1] = True
            current_balance -= counts[last_ind-1].item()
        b_sentences = sentences.clone()
        self.state_to_state_letter = self.state_to_state_letter.view(-1)
        for ind, set_id in enumerate(balance_set):
            val = 0
            if set_id:
                val = 1
            b_sentences[sentences == letters_id[ind]] = val
            self.state_to_state_letter[letters_id[ind]] = val
        assert b_sentences.view(-1).sum() == best_balance
        self.state_to_state_letter = self.state_to_state_letter.view(self.states_count, self.states_count)
        return b_sentences

    def gen(self, count, length):
        with torch.no_grad():
            self.state_to_state_letter = torch.arange(self.states_count*self.states_count).view(self.states_count, self.states_count)
            #self.state_to_state_letter.random_(0,2)
            sentences = torch.LongTensor(count, length)
            states = torch.LongTensor(count).random_(0, self.states_count)
            for i in range(length):
                res = torch.FloatTensor(count).uniform_()
                probs = self.state_to_state_cum_prob[states]
                next_states = self.states_count-(res.view(-1,1) < probs).sum(dim=1)
                next_states = next_states.clamp(max=self.states_count-1)
                letters_ind = self.state_to_state_letter[states, next_states]
                sentences[:,i] = letters_ind
                states = next_states
            sentences = self.balance(sentences)
            return sentences

    def calc_probs(self, sentences):
        size = sentences.size(0)
        states_count = self.state_to_state_prob.size(0)
        length = sentences.size(1)
        with torch.no_grad():
            state_to_prob = torch.FloatTensor(size, states_count).double()
            state_to_prob[:,:] = 1.0
            for i in range(length):
                letters = sentences[:,i]
                s1 = self.state_to_state_letter.size()
                s2 = letters.size()
                sf = s2+s1

                t1 = self.state_to_state_letter.view((1,)+s1).expand(sf)
                t2 = letters.view(s2+(1,1)).expand(sf)
                t3 = self.state_to_state_prob.view((1,)+s1).expand(sf).double()
                t4 = (t1 == t2).double()
                t5 = torch.mul(t3, t4)
                t6 = state_to_prob
                next_state_to_prob = torch.matmul(t6.view(t6.size(0), 1, t6.size(1)), t5).view_as(t6)
                state_to_prob = next_state_to_prob
            return state_to_prob.sum(dim=1)

class DataProvider:
    def __init__(self):
        self.number_of_cases = 10

    def create_data(self, data_size, length, states_count, letters_count, seed):
        while True:
            torch.manual_seed(seed)
            languages = [Language(states_count, letters_count), Language(states_count, letters_count)]
            data_size_per_lang = data_size//len(languages)
            datas = []
            targets = []
            for ind, lan in enumerate(languages):
                datas.append(lan.gen(data_size_per_lang, length))
                t = torch.LongTensor(data_size_per_lang)
                t[:] = ind
                targets.append(t)
            bad_count = 0
            good_count = 0
            for ind, data in enumerate(datas):
                probs = [lan.calc_probs(data) for lan in languages]
                bad_count += (probs[ind] <= probs[1-ind]).long().sum().item()
                good_count += (probs[ind] > probs[1-ind]).long().sum().item()
            best_prob = good_count/(bad_count+good_count)
            if best_prob > 0.95:
                break
            print("Low best prob = {}, seed = {}".format(best_prob, seed))
            seed += 1

        data = torch.cat(datas, dim=0)
        target = torch.cat(targets, dim=0)
        perm = torch.randperm(data.size(0))
        data = data[perm]
        target = target[perm]
        return (data, target.view(-1, 1).float(), best_prob)

    def create_case_data(self, case):
        data_size = 256*4
        case_configs = [
                (8, 2, 7),
                (16, 3, 34),
                (32, 4, 132),
                (64, 5, 13),
                (128, 6, 1),
                (256, 7, 5),
                (256, 7, 6),
                (256, 7, 71),
                (256, 7, 19),
                (256, 7, 40)
                ]
        case_config = case_configs[min(case, 10)-1]
        length = case_config[0]
        states_count = case_config[1]
        # seed help generate data faster
        seed = 1000*case + case_config[2]
        letters_count = 2
        data, target, best_prob = self.create_data(2*data_size, length, states_count, letters_count, seed)
        return sm.CaseData(case, Limits(), (data[:data_size], target[:data_size]), (data[data_size:], target[data_size:])).set_description("States = {} Length = {} Seed = {} Best prob = {:.3}".format(states_count, length, seed, best_prob))

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
    gs.GridSearch().run(Config(), case_number=6, random_order=False, verbose=False)
else:
    a = torch.tensor(np.ones((1, 10000)), dtype=torch.float32)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    a.to(device)
    # If you want to run specific case, put number here
    sm.SolutionManager().run(Config(), case_number=-1)
