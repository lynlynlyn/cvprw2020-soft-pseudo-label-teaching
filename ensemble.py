import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel.parallel_apply import parallel_apply

import numpy as np


class TrainingModel(nn.Module):
    def __init__(self, model, classifier, device, p_coef, e_coef, class_p, params):
        super(TrainingModel, self).__init__()

        self.model = model
        self.classifier = classifier
        self.cls_opt = torch.optim.SGD(self.classifier.parameters(), lr=0.0625, momentum=0.9,
                                         dampening=0.9, weight_decay=0.001)
        self.model_opt = torch.optim.SGD(self.model.parameters(), lr=0.0625)

        self.batch_size = 25
        self.device = device
        self.params =params
        self.epoch = 0
        self.total_epoch = params.joint_start_epoch + params.joint_epoch
        self.val_acc = torch.zeros(self.total_epoch).to(self.device)

        self.p_coef = p_coef
        self.e_coef = e_coef
        self.class_p = class_p
        
        self.drop = params.drop

    def init_task(self, x_tr, y_tr, x_val):
        self.n_way = 5
        self.n_query = x_val.shape[0] / self.n_way

        self.x_tr = x_tr
        self.y_tr = y_tr
        self.x_val = x_val
        self.y_val = torch.from_numpy(np.repeat(range(self.n_way), self.n_query)).to(self.device)
        self.scores_history = torch.zeros([self.total_epoch, x_val.shape[0], 5]).to(self.device)

        self.x = self.x_tr
        self.y = self.y_tr

    def forward(self,):
        self.model.train()
        self.classifier.train()
        cur_support_size = self.x.shape[0]
        rand_id = np.random.permutation(cur_support_size)
        for j in range(0, cur_support_size, self.batch_size):
            self.cls_opt.zero_grad()
            self.model_opt.zero_grad()
            selected_id = torch.from_numpy(rand_id[j: min(j + self.batch_size, cur_support_size)]).to(self.device)
            x_batch = self.x[selected_id]
            y_batch = self.y[selected_id]

            output = self.model(x_batch)
            if self.drop is not None:
                output = F.dropout(output, p=self.drop)
            output = self.classifier(output)
            scores = F.softmax(output, dim=1)
            scores_ = torch.mean(scores, dim=0)

            loss_cls = -torch.mean(torch.sum(F.log_softmax(output, dim=1) * y_batch, dim=1))
            loss = loss_cls
            if self.epoch >= self.params.joint_start_epoch:
                loss_p = -torch.sum(torch.log(scores_) * self.class_p)
                loss_e = -torch.mean(torch.sum(F.log_softmax(output, dim=1) * scores, dim=1))
                loss += self.p_coef * loss_p + self.e_coef * loss_e
            loss.backward()
            self.cls_opt.step()
            self.model_opt.step()

        self.model.eval()
        self.classifier.eval()
        with torch.no_grad():
            output = self.model(self.x_val)
            scores = self.classifier(output)
            scores = F.softmax(scores, dim=1)
            self.scores_history[self.epoch] = scores.detach()
            pseudo_scores, pseudo_label = scores.topk(1, 1, True, True)
            pseudo_scores, pseudo_label = pseudo_scores.squeeze(1), pseudo_label.squeeze(1)

            if self.epoch >= self.params.joint_start_epoch-1:
                avg_query_label = torch.mean(self.scores_history[self.epoch + 1 - self.params.use_epoch:self.epoch + 1, :, :], dim=0)
                self.x = torch.cat((self.x_tr, self.x_val), dim=0)
                self.y = torch.cat((self.y_tr, avg_query_label), dim=0)
            else:
                self.x = self.x_tr
                self.y = self.y_tr
            self.val_acc[self.epoch] = (torch.sum(pseudo_label==self.y_val) * 100. / (self.n_way * self.n_query)).detach()

        self.epoch += 1
        return

    
def avg_ensemble(scores, avg_epoch=1):
    n_way = scores.shape[-1]
    n_query = scores.shape[-2]/n_way
    score = np.mean(np.mean(scores[:,-avg_epoch:,:,:], axis=0), axis=0)
    p = score.argmax(-1)
    val_acc = np.sum(p == np.repeat(range(n_way), n_query))*100/(n_way*n_query)
    return val_acc