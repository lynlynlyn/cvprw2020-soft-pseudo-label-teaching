import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob
from itertools import combinations, chain
import random
import copy
import torch.cuda.comm as comm

import backbone
from data.datamgr import SimpleDataManager, SetDataManager
from methods.protonet import ProtoNet

from io_utils import model_dict, parse_args, get_resume_file, get_best_file, get_assigned_file 

from utils import *

from datasets import ISIC_few_shot, EuroSAT_few_shot, CropDisease_few_shot, Chest_few_shot, miniImageNet_few_shot

import torch.multiprocessing as mp

from ensemble import TrainingModel, avg_ensemble
from torch.nn.parallel.parallel_apply import parallel_apply

class Classifier(nn.Module):
    def __init__(self, dim, n_way):
        super(Classifier, self).__init__()
        
        self.fc = nn.Linear(dim, n_way)

    def forward(self, x):
        x = self.fc(x)
        return x


def plot_array(a, n):
    b = ((np.arange(a.shape[0])+1)%n==0)
    print(np.round(a[b]*100)/100.) 
    

def finetune(novel_loader, n_query = 15, pretrained_dataset='miniImageNet', freeze_backbone = False, n_way = 5, n_support = 5): 

    iter_num = len(novel_loader) 

    val_accs = AverageMeter()
    es = AverageMeter()
    
    models_score = []

    for task_i, (x, y) in enumerate(novel_loader):

        ###############################################################################################
        # load pretrained model on miniImageNet

        if task_i == 0:
            state_list = []
            imgisgray = True if torch.abs(torch.mean(x[0,0,0,:,:]*0.229-x[0,0,1,:,:]*0.224+0.485-0.456)) < 1e-5 else False
            for model_name in params.model_list:
                checkpoint_dir = '%s/%s%s_%s.tar' %(params.model_dir, model_name, 'G50' if imgisgray else '', 399)
                print(checkpoint_dir)
                tmp = torch.load(checkpoint_dir)
                state = tmp['state']
                state_keys = list(state.keys())
                for _, key in enumerate(state_keys):
                    if "feature." in key:
                        newkey = key.replace("feature.","")
                        state[newkey] = state.pop(key)
                    else:
                        state.pop(key)
                state_list.append(state)

        pretrained_model_list = []
        classifier_list = []
        model_num = len(params.model_list)
        device_list = list(range(torch.cuda.device_count()))[: model_num] if params.device_list is None else  params.device_list 
        # set params.device_list =[0,0,0,0,0,0,0,0] to run all models on cuda:0
        for model_idx, model_name in enumerate(params.model_list):
            pretrained_model = model_dict[model_name]()
            pretrained_model.load_state_dict(copy.deepcopy(state_list[model_idx]))
            pretrained_model.to(device_list[model_idx])
            classifier = Classifier(pretrained_model.final_feat_dim, n_way)
            classifier.to(device_list[model_idx])
            pretrained_model_list.append(pretrained_model)
            classifier_list.append(classifier)

        class_p = comm.broadcast(torch.ones(1, n_way) / n_way, device_list)
        model_pool = [TrainingModel(mo, cl, de, params.p_coef, params.e_coef, cp, params)
                      for (mo, cl, de, cp) in zip(pretrained_model_list, classifier_list, device_list, class_p)]

        n_query = x.size(1) - n_support
        support_size = n_way * n_support

        y_a_i = torch.from_numpy(np.repeat(range(n_way), n_support).astype(np.int64))
        y_a_i_oh = torch.zeros(support_size, n_way).scatter_(1, y_a_i.view(-1,1),1)

        x_a_i = x[ :, :n_support,:,:,:].contiguous().view(n_way*n_support, *x.size()[2:])
        x_b_i = x[ :, n_support:,:,:,:].contiguous().view(n_way*n_query, *x.size()[2:])

        x_a_list = comm.broadcast(x_a_i, device_list)
        x_b_list = comm.broadcast(x_b_i, device_list)
        y_a_list = comm.broadcast(y_a_i_oh, device_list)
        total_epoch = params.joint_start_epoch + params.joint_epoch
        for mi, m in enumerate(model_pool):
            m.init_task(x_a_list[mi], y_a_list[mi], x_b_list[mi])

        for epoch in range(total_epoch):
            parallel_apply(model_pool, [[] for _ in range(model_num)])
            if params.mml is not None and epoch >= params.joint_start_epoch-1 and epoch < total_epoch-1:
                if params.mml == 'all':
                    cur_score = [m.scores_history[m.epoch-params.use_epoch:m.epoch].to(0) for m in model_pool]
                    #print(cur_score)
                    mean_score = comm.broadcast(torch.mean(torch.cat(cur_score, dim=0), dim=0), device_list)
                    for mi, m in enumerate(model_pool):
                        m.y = torch.cat((m.y_tr, mean_score[mi]), dim=0)
                else:
                    print('Wrong type !')
            
        val_acc_m = []
        for mi, m in enumerate(model_pool):
            val_acc_m.append(m.val_acc.cpu().numpy())
        val_acc_m = np.array(val_acc_m)
        val_accs.update(val_acc_m)

        s = []
        for mi, m in enumerate(model_pool):
            s.append( m.scores_history.cpu().numpy())
        s = np.array(s)
        models_score.append(s)
        
        es.update(avg_ensemble(s))
        print(es.val, es.avg)
        for i in range(model_num):
            print(val_accs.val[i, params.joint_start_epoch-1], val_accs.val[i, -1],
                  val_accs.avg[i, params.joint_start_epoch-1], val_accs.avg[i, -1])
        print('#################################################')
    
    c95 = val_accs.c95()
    for i in range(model_num):
        print('%d Test Acc = %4.2f+%4.2f%%, %4.2f+%4.2f%%, %4.2f+%4.2f%%'
              %(iter_num, val_accs.avg[i, params.joint_start_epoch-1], c95[i, params.joint_start_epoch-1],
                val_accs.avg[i, 299], c95[i, 299], val_accs.avg[i, -1], c95[i, -1]))
        print(val_accs.avg[i])
        
    c95 = es.c95()
    print('Ensemble Avg. Acc = %4.2f+%4.2f%%'%(es.avg, c95))
    
    return np.array(models_score)


if __name__=='__main__':
    manualSeed = random.randint(1, 10000)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    np.random.seed(manualSeed)
    print('Random Seed is:%d'%(manualSeed))
    params = parse_args('train')
    print(params)
    ##################################################################
    model_name_list = ['ResNet10', 'ResNest102s2N', 'ResNest102s1N', 'ResNest101s1N', 'ResNet10CJ', 'ResNest102s2NCJ', 'ResNest102s1NCJ', 'ResNest101s1NCJ']
    params.model_list = [model_name_list[int(mi)] for mi in params.model_list]
    print(params.model_list)
    
    ##################################################################
    image_size = 224
    iter_num = params.iter_num

    n_query = max(1, int(16* params.test_n_way/params.train_n_way)) #if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
    few_shot_params = dict(n_way = params.test_n_way , n_support = params.n_shot) 
   
    freeze_backbone = params.freeze_backbone
    ##################################################################
    pretrained_dataset = "miniImageNet"

    dataset_list = ["ISIC", "EuroSAT", "CropDisease", "ChestX", "miniImagenet_test"]
    dataset_names = []
    for d in params.dataset_names:
        dataset_names.append(dataset_list[int(d)])
    novel_mgrs = []

    if "ISIC" in dataset_names:
        print ("ISIC")
        datamgr             =  ISIC_few_shot.SetDataManager(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
        novel_mgrs.append(datamgr)

    if "EuroSAT" in dataset_names:
        print ("EuroSAT")
        datamgr             =  EuroSAT_few_shot.SetDataManager(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
        novel_mgrs.append(datamgr)

    if "CropDisease" in dataset_names:
        print ("CropDisease")
        datamgr             =  CropDisease_few_shot.SetDataManager(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
        novel_mgrs.append(datamgr)

    if "ChestX" in dataset_names:
        print ("ChestX")
        datamgr             =  Chest_few_shot.SetDataManager(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
        novel_mgrs.append(datamgr)
        
    if "miniImagenet_test" in dataset_names:
        print ("miniImagenet_test")
        datamgr             =  miniImageNet_few_shot.SetDataManager(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
        novel_mgrs.append(datamgr)
    
    #########################################################################
    import gc
    
    for idx, novel_mgr in enumerate(novel_mgrs):

        start_epoch = params.start_epoch
        stop_epoch = params.stop_epoch
        print (freeze_backbone)
        print('Loading %s'%(dataset_names[idx]))
        novel_loader = novel_mgr.get_data_loader(False)
        
        # replace finetine() with your own method
        ss = finetune(novel_loader, n_query = 15, pretrained_dataset=pretrained_dataset, freeze_backbone=freeze_backbone, **few_shot_params)
        # np.save('/output/%s.npy'%(dataset_names[idx]), ss)
        
        del novel_loader
        gc.collect()