import torch
import numpy as np

def adjust_learning_rate(optimizer, epoch, lr=0.01, step1=30, step2=60, step3=90):
    """Sets the learning rate to the initial LR decayed by 10 every X epochs"""
    if epoch >= step3:
        lr = lr * 0.001
    elif epoch >= step2:
        lr = lr * 0.01
    elif epoch >= step1:
        lr = lr * 0.1
    else:
        lr = lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0      
        self.sum = 0
        self.count = 0
        self.hist = []

    def update(self, val, n=1):
        self.val = val
        self.hist.append(val)
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
    def std(self):
        return np.std(np.array(self.hist), axis=0)
    
    def c95(self):
        return 1.96* self.std()/np.sqrt(len(self.hist))
        

def one_hot(y, num_class):         
    return torch.zeros((len(y), num_class)).scatter_(1, y.unsqueeze(1), 1)

def sparsity(cl_data_file):
    class_list = cl_data_file.keys()
    cl_sparsity = []
    for cl in class_list:
        cl_sparsity.append(np.mean([np.sum(x!=0) for x in cl_data_file[cl] ])  ) 

    return np.mean(cl_sparsity)


def freeze_blocks_resnet(net, use_block):
    net.train()
    if 'B1' not in use_block:
        for m in net.trunk[:5].parameters():
            m.requires_grad = False
        for module in net.trunk[:5].children():
            module.eval()
    if 'B2' not in use_block:
        for m in net.trunk[5].parameters():
            m.requires_grad = False
        net.trunk[5].eval()
    if 'B3' not in use_block:
        for m in net.trunk[6].parameters():
            m.requires_grad = False
        net.trunk[6].eval()
    if 'B4' not in use_block:
        for m in net.trunk[7].parameters():
            m.requires_grad = False
        net.eval()

