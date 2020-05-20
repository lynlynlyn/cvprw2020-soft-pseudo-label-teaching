import numpy as np
import os
import glob
import argparse
import backbone
import timm


model_dict = dict(
            ResNet10 = backbone.ResNet10,
            ResNest102s2N = timm.resnest10_2s2,
            ResNest102s1N = timm.resnest10_2s1,
            ResNest101s1N = timm.resnest10_1s1,
            ResNet10CJ = backbone.ResNet10,
            ResNest101s1NCJ = timm.resnest10_1s1,
            ResNest102s1NCJ = timm.resnest10_2s1,
            ResNest102s2NCJ = timm.resnest10_2s2,
            )

def parse_args(script):
    parser = argparse.ArgumentParser(description= 'few-shot script %s' %(script))
    parser.add_argument('--dataset', default='miniImageNet', help='training base model')
    parser.add_argument('--model', default='ResNet10', help='backbone architecture')
    parser.add_argument('--model_list', nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7], help='backbone architecture') 
    parser.add_argument('--model_dir', default='', help='model dir')
    parser.add_argument('--device_list', nargs='+', default=None, help='cuda device list for each sub network')
    
    parser.add_argument('--mml', type=str, default='all', help='backbone architecture')

    parser.add_argument('--method'      , default='baseline',   help='baseline/protonet')
    parser.add_argument('--iter_num'    , default=600, type=int,  help='class num to classify for training')
    parser.add_argument('--train_n_way' , default=5, type=int,  help='class num to classify for training')
    parser.add_argument('--test_n_way'  , default=5, type=int,  help='class num to classify for testing (validation) ')
    parser.add_argument('--n_shot'      , default=5, type=int,  help='number of labeled data in each class, same as n_support') 
    parser.add_argument('--train_aug'   , default=True,  help='perform data augmentation or not during training ')

    parser.add_argument('--drop', default=None, type=float, help='drop')
    parser.add_argument('--imagejitter', default=0.4, type=float, help='image jitter')

    parser.add_argument('--dataset_names', nargs='+', default=[0, 1, 2, 3], help="ISIC: 0, EuroSAT: 1, CropDisease: 2, ChestX: 3")
    parser.add_argument('--joint_start_epoch', type= int, default=100, help='epoch of fine-tune')
    parser.add_argument('--e_coef', type= float, default=0.4, help='alpha')
    parser.add_argument('--p_coef', type= float, default=0.8, help='beta')
    parser.add_argument('--use_epoch', type=int, default=10, help='average epoch')
    parser.add_argument('--joint_epoch', type=int, default=200, help='epoch of transductive fine-tune')

    parser.add_argument('--models_to_use', '--names-list', nargs='+', default=None, help='invalid')
    parser.add_argument('--fine_tune_all_models'   , action='store_true',  help='invalid')
    parser.add_argument('--freeze_backbone'   , action='store_true', help='invalid')
    parser.add_argument('--use_block', nargs='+', default=None, help='invalid')

    if script == 'train':
        parser.add_argument('--grayscale', action='store_true', default=False, help='Use Grayscale')
        parser.add_argument('--num_classes' , default=200, type=int, help='total number of classes in softmax, only used in baseline') #make it larger than the maximum label value in base class
        parser.add_argument('--save_freq'   , default=50, type=int, help='Save frequency')
        parser.add_argument('--start_epoch' , default=0, type=int,help ='Starting epoch')
        parser.add_argument('--stop_epoch'  , default=400, type=int, help ='Stopping epoch') # for meta-learning methods, each epoch contains 100 episodes
        parser.add_argument('--save_dir', default='', type=str, help='save the pre-training model')

    elif script == 'save_features':
        parser.add_argument('--split'       , default='novel', help='base/val/novel') #default novel, but you can also test base/val class accuracy if you want 
        parser.add_argument('--save_iter', default=-1, type=int,help ='save feature from the model trained in x epoch, use the best model if x is -1')
    elif script == 'test':
        parser.add_argument('--split'       , default='novel', help='base/val/novel') #default novel, but you can also test base/val class accuracy if you want 
        parser.add_argument('--save_iter', default=-1, type=int,help ='saved feature from the model trained in x epoch, use the best model if x is -1')
        parser.add_argument('--adaptation'  , action='store_true', help='further adaptation in test time or not')
    else:
       raise ValueError('Unknown script')
        
    return parser.parse_args()

def get_assigned_file(checkpoint_dir,num):
    assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
    return assign_file

def get_resume_file(checkpoint_dir):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None

    filelist =  [ x  for x in filelist if os.path.basename(x) != 'best_model.tar' ]
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
    return resume_file

def get_best_file(checkpoint_dir):    
    best_file = os.path.join(checkpoint_dir, 'best_model.tar')
    if os.path.isfile(best_file):
        return best_file
    else:
        return get_resume_file(checkpoint_dir)

