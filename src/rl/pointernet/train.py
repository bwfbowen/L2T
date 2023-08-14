import os 
import time 
import random 
import matplotlib.pyplot as plt
from IPython.display import clear_output

import torch 
import torch.nn as nn
from torch.optim import Adam
from torch.optim import lr_scheduler
from torch.autograd import Variable

# training
class Train(object):

    def __init__(self, model, train_set, validation_set, batch_size = 128, max_grad_norm = 1., lr = 1e-4, update_steps = 5000, beta = 0.9, max_time: int = 2 * 3600):
        self.model = model
        self.train_set = train_set
        self.validation_set = validation_set
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.max_time = max_time
        self.beta = beta
        self.optimizer_all = Adam(list(model.critic.parameters()) + list(model.pointer_net.parameters()), lr = lr)
        self.optimizer_pointer = Adam(model.pointer_net.parameters(), lr = lr)
        self.lr_scheduler_pointer = lr_scheduler.MultiStepLR(self.optimizer_pointer, 
                                                     list(range(update_steps, update_steps * 1000, update_steps)), gamma=0.96)
        self.lr_scheduler_all = lr_scheduler.MultiStepLR(self.optimizer_all, 
                                                     list(range(update_steps, update_steps * 1000, update_steps)), gamma=0.96)
        self.mse_loss = nn.MSELoss()
        self.input_dim = 2 # points dimension

        self.train_rewards = []
        self.val_rewards = []
        self.best_tour = None 
        self.best_cost = float('inf')

    def train_and_validation(self, n_epoch, training_steps, use_critic = True):
        start_time = time.time()
        moving_average = 0
        for epoch in range(n_epoch):
            for step in range(training_steps):
                
                if time.time() - start_time >= self.max_time:
                    return 
                training_set = random.sample(self.train_set, self.batch_size)
                training_set = Variable(torch.cat(training_set).view(self.batch_size, -1, self.input_dim))
                L, log_probs, pi, index_list, b = self.model(training_set)

                log_probs = log_probs.view(-1)
                log_probs[(log_probs < -1000).detach()] = 0.

                if not use_critic:
                    if step == 0:
                        moving_average = L.mean()
                    else:
                        moving_average = (moving_average * self.beta) + ((1. - self.beta) * L.mean())

                    advantage = L - moving_average
                    actor_loss = (advantage * log_probs).mean()


                    self.optimizer_pointer.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.pointer_net.parameters(), self.max_grad_norm, norm_type=2)
                    self.optimizer_pointer.step()
                    self.lr_scheduler_pointer.step()
                    moving_average = moving_average.detach()

                else:
                    critic_loss = self.mse_loss(b.view(-1), L)
                    advantage = L - b.view(-1)
                    actor_loss = (advantage * log_probs).mean()
                    loss = actor_loss + critic_loss
                    self.optimizer_all.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(list(self.model.critic.parameters()) + \
                                                   list(self.model.pointer_net.parameters()), self.max_grad_norm, norm_type=2)
#                     torch.nn.utils.clip_grad_norm_(, self.max_grad_norm, norm_type=2)
                    self.optimizer_all.step()
                    self.lr_scheduler_all.step()
                
                self.train_rewards.append(L.mean().data.cpu())

                if step % 10 == 0:
                    self.plot(epoch)
                if step % 100 == 0:
                    val_set = Variable(torch.cat(self.validation_set).view(len(self.validation_set), -1, self.input_dim))
                    L, log_probs, pi, index_list, b = self.model(val_set)
                    cost = L.mean().data.cpu()
                    if cost < self.best_cost:
                        self.best_cost = cost
                        List_all = []
                        list1 =[]
                        for j in range(len(val_set)):
                            for k in range(self.model.seq_len):
                                list1.append(index_list[k][j].item())
                            List_all.append(list1)
                            list1 =[]
                        tour = List_all[0] + [0]
                        self.best_tour = tour 

                    self.val_rewards.append(cost)

                # model save
                if step % 1000 == 0:
                    if use_critic:
                        torch.save(self.model, os.path.join('..', 'tmp', 'pointer', 
                                                            'model_tsp{}_critic.pt'.format(self.model.seq_len)))
                    else:
                        torch.save(self.model, os.path.join('..', 'tmp', 'pointer', 
                                                            'model_tsp{}_mvg_avg.pt'.format(self.model.seq_len)))

    def plot(self, epoch):
        clear_output(True)
        plt.figure(figsize=(20,5))
        plt.subplot(131)
        plt.title('train tour length: epoch %s reward %s' % (epoch, self.train_rewards[-1] if len(self.train_rewards) else 'collecting'))
        plt.plot(self.train_rewards)
        plt.grid()
        plt.subplot(132)
        plt.title('val tour length: epoch %s reward %s' % (epoch, self.val_rewards[-1] if len(self.val_rewards) else 'collecting'))
        plt.plot(self.val_rewards)
        plt.grid()
        plt.show()