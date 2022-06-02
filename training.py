from cProfile import label
from ctypes import c_void_p
from email.mime import image
import os
from pickletools import long1
from typing import Type
import warnings
from tqdm import tqdm
import argparse
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
import itertools
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms, datasets
from torchinfo import summary
from datasets.load_data import load_pickle
from torchvision.datasets import ImageFolder
from sklearn.metrics import balanced_accuracy_score, confusion_matrix ,f1_score
import torch.nn.functional as F
from torch.autograd import Variable
from networks.expr import Expr
import networks.resnet as ResNet
from collections import OrderedDict
def warn(*args, **kwargs):
    pass
warnings.warn = warn

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=700, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate for adam.')
    parser.add_argument('--workers', default=2, type=int, help='Number of data loading workers.')
    parser.add_argument('--epochs', type=int, default=5, help='Total training epochs.')
    parser.add_argument('--num_head', type=int, default=4, help='Number of attention head.')
    parser.add_argument('--opt', type=str)
    parser.add_argument('--freeze', type=str)
    parser.add_argument('--sch', type=str)
    parser.add_argument('--pretrained', type=str)

    return parser.parse_args()

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class AffinityLoss(nn.Module):
    def __init__(self, device, num_class=8, feat_dim=512):
        super(AffinityLoss, self).__init__()
        self.num_class = num_class
        self.feat_dim = feat_dim
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.device = device

        self.centers = nn.Parameter(torch.randn(self.num_class, self.feat_dim).to(device))

    def forward(self, x, labels):
        x = self.gap(x).view(x.size(0), -1)

        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_class) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_class, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)
        #print(batch_size)
        #print(self.num_class)
        classes = torch.arange(self.num_class).long().to(self.device)
        labels = labels.expand(batch_size, self.num_class)
        mask = labels.eq(classes.expand(batch_size, self.num_class))

        dist = distmat * mask.float()
        dist = dist / self.centers.var(dim=0).sum()

        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss

class PartitionLoss(nn.Module):
    def __init__(self, ):
        super(PartitionLoss, self).__init__()
    
    def forward(self, x):
        num_head = x.size(1)

        if num_head > 1:
            var = x.var(dim=1).mean()
            loss = torch.log(1+num_head/var)
        else:
            loss = 0
            
        return loss
class ImbalancedDatasetSampler(data.sampler.Sampler):
    def __init__(self, dataset, indices: list = None, num_samples: int = None):
        self.indices = list(range(len(dataset))) if indices is None else indices
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset)
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()

        weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())

        # self.weights = self.weights.clamp(min=1e-5)

    def _get_labels(self, dataset):
        if isinstance(dataset, datasets.ImageFolder):
            return [x[1] for x in dataset.imgs]
        elif isinstance(dataset, torch.utils.data.Subset):
            return [dataset.dataset.imgs[i][1] for i in dataset.indices]
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
def plt_score(plt_name,train_acc,train_loss,train_f1,val_acc,val_loss,val_f1):
    plt.figure(figsize=(8, 6))
    plt.plot(train_acc,'r',label="train")
    plt.plot(val_acc,'b',label="validation")
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.legend()
    plt.show()
    plt.savefig("../result/Numhead_ensemble/plt/"+plt_name+"_acc.png")
    plt.close()
    
    plt.figure(figsize=(8, 6))
    plt.plot(train_loss,'r',label="train")
    plt.plot(val_loss,'b',label="validation")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.savefig("../result/Numhead_ensemble/plt/"+plt_name+"_loss.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(train_f1,'r',label="train")
    plt.plot(val_f1,'b',label="validation")
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.legend()
    plt.show()
    plt.savefig("../result/Numhead_ensemble/plt/"+plt_name+"_f1.png")
    plt.close()
    
    

def run_training():
    args = parse_args()
    print(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    cuda = torch.cuda.is_available()
    if cuda:
        print("torch.backends.cudnn.version: {}".format(torch.backends.cudnn.version()))
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
    model_name = str(args.num_head)+"_num_head_"+"NumHead_Ensemble_PRETRAINED"+args.pretrained+"_LR_"+str(args.lr)+"_FREEZE_"+str(args.freeze)+"_OPT_"+str(args.opt)+"_schedule_"+str(args.sch)
    
    
    model = Expr(args.pretrained, num_head=args.num_head, num_class=8)
    if ((device.type == 'cuda') and (torch.cuda.device_count()>1)): # GPU 개수 2개이상이면 병렬처리 진행
        print('Multi GPU activate')
        model = nn.DataParallel(model)
        model = model.cuda()

    
    ct=0
    ct_=0
    for param in model.parameters():
        #print
        # param.requires_grad = False
        ct+=1
    print("Model Layer Num : ", ct)
    for param in model.parameters():
        ct_+=1

        if args.freeze == "100" :
            #param.requires_grad = True
            ct_ = 0
        elif args.freeze == "50" :
            if ct_ < ct*0.5 :
                param.requires_grad = False
            else: break
        elif args.freeze == "0" :
             if ct_ < ct-3 :
                param.requires_grad = False

    print("freeze Layer Num : ", ct_)
    model.to(device)

    data_transforms = transforms.Compose([
        transforms.Resize((224,224)),                               # image resize
        transforms.ToTensor(),                                      # image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],            # image normalize 
                                std=[0.229, 0.224, 0.225]),
        
        ])

    train_dataset = ImageFolder("./datasets/fulldatasets/train",transform =data_transforms) # train_dataset load
        
    print('Whole train set size:', train_dataset.__len__())
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               sampler =ImbalancedDatasetSampler(train_dataset), # data sampling
                                               shuffle = False,  
                                               pin_memory = True)

    data_transforms_val = transforms.Compose([
        #transforms.ToPILImage(mode=None),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])   

    val_dataset = ImageFolder("./datasets/fulldatasets/val",transform =data_transforms_val) # val dataset load
    
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               shuffle = False,  
                                               pin_memory = True)
    print("Set Optimizer .. ")

    #criterion_cls = torch.nn.CrossEntropyLoss(weight=torch.Tensor([0.365, 0.975, 0.986, 0.988, 0.837, 0.891, 0.958, 0.365]))
    criterion_cls = FocalLoss()
    criterion_af = AffinityLoss(device)
    criterion_pt = PartitionLoss()
    criterion_cls = criterion_cls.cuda()
    params = list(model.parameters()) + list(criterion_af.parameters())

    if args.opt == "ADAM" :
        optimizer = torch.optim.Adam(params,args.lr,weight_decay = 1e-4)
    elif args.opt == "ADAMW" :
        optimizer = torch.optim.AdamW(params,args.lr,weight_decay = 0)
    elif args.opt == "SGD" :
        optimizer = torch.optim.SGD(params,lr=args.lr, weight_decay = 0, momentum=0.9)
    
    if args.sch == "True":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.6)
    
   
    

    best_acc = 0
    print("Model to device ..")
    print(count_parameters(model))
    #model.to(device)
    train_f1 = []
    val_f1 = []
    train_acc= []
    val_acc = []
    train_loss = []
    val_loss = []
    bf1 = 0
    for epoch in tqdm(range(1, args.epochs + 1)):
        running_loss = 0.0
        correct_sum = 0
        iter_cnt = 0
        model.train()
        temp_exp_target = []
        temp_exp_pred = []
        running_f1=0.0
        
        for (imgs, targets) in tqdm(train_loader): #training part 
            iter_cnt += 1 #step 진행 개수 저장
            optimizer.zero_grad() # gradient 초기화
            
            imgs = imgs.to(device)              #  
            imgs = imgs.float()                 # Image Target GPU에 올리기 
            targets_ =targets                   #
            targets_=targets_.to(device)        # Loss 계산용 target

            targets = torch.eye(8)[targets]     # target One Hot vector 변환
            targets = targets.to(device)        # Affinity loss 계산용 target
            
            out,feat,heads = model(imgs)        # training model 
            
            loss = criterion_cls(out,targets_) + 1* criterion_af(feat,targets) + 1*criterion_pt(heads) #89.3 89.4 #Loss 계산
            
            loss.backward()
            #overfitting 방지용 
            nn.utils.clip_grad_norm_(params, max_norm=1) #max_norm 적절한 값을 넣어야함 0.1 5 

            optimizer.step() 
            
            running_loss += loss # step loss 합치기
            
            _, predicts = torch.max(out, 1)
            
            for i in range(predicts.shape[0]):                
                temp_exp_pred.append(predicts[i].cpu().numpy())
                temp_exp_target.append(targets[i].argmax().cpu().numpy())
           
            correct_num = torch.eq(predicts, targets.argmax(axis=1)).sum()      #
            correct_sum += correct_num                                          # accuracy 계산하기 위해 맞춘 개수 저장
        
        f1=[]

        temp_exp_pred = np.array(temp_exp_pred)             #  
        temp_exp_target = np.array(temp_exp_target)         #
        temp_exp_pred = torch.eye(8)[temp_exp_pred]         # pred target f1 score 계산하기위해 변환   
        temp_exp_target = torch.eye(8)[temp_exp_target]     #
        
        for i in range(0,8):
            exp_pred = temp_exp_pred[:,i]                   #
            exp_target = temp_exp_target[:,i]               # class별 f1 score 계산
            f1.append(f1_score(exp_pred,exp_target))        #
        
        acc = correct_sum.float() / float(train_dataset.__len__())  # 맞춘 개수 / 전체 개수 = accuracy
        running_loss = running_loss/iter_cnt                        # 전체 loss 계산
        running_f1 =np.mean(f1)                                     # f1 average 계산
        train_acc.append(acc.cpu().numpy())
        train_loss.append(running_loss.cpu().detach().numpy())
        train_f1.append(running_f1)
        
        tqdm.write('[Epoch %d] Training accuracy: %.4f. Loss: %.3f. F1: %.5f. LR %.6f. ' % (epoch, acc, running_loss,running_f1,optimizer.param_groups[0]['lr']))
        
        with torch.no_grad(): #validation part
            running_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            sample_cnt = 0
            baccs = []
            running_f1 = 0.0
            p_=[]
            t_=[]
            model.eval()
            
            for (imgs, targets) in tqdm(val_loader):
                imgs = imgs.to(device)
                imgs = imgs.float()
                targets_ =targets
                targets_=targets_.to(device)
                targets = torch.eye(8)[targets]
                targets = targets.to(device)
                
        
                out,feat,heads = model(imgs)
                loss = criterion_cls(out,targets_) + criterion_af(feat,targets) + criterion_pt(heads)

                running_loss += loss
                iter_cnt+=1
                _, predicts = torch.max(out, 1)
            
                correct_num  = torch.eq(predicts,targets.argmax(axis=1))
                bingo_cnt += correct_num.sum().cpu()
                sample_cnt += out.size(0)
                for p, t in zip(predicts, targets) :
                    p_.append(p.cpu())
                    t_.append(t.argmax().cpu())

                
                baccs.append(balanced_accuracy_score(targets.cpu().numpy().argmax(axis=1),predicts.cpu().numpy()))
            running_loss = running_loss/iter_cnt   
            if args.sch == "True":
                scheduler.step()
            
            f1=[]

            temp_exp_pred = np.array(p_)
            temp_exp_target = np.array(t_)
            temp_exp_pred = torch.eye(8)[temp_exp_pred]
            temp_exp_target = torch.eye(8)[temp_exp_target]
            for i in range(0,8):
                #print(temp_exp_pred)
                exp_pred = temp_exp_pred[:,i]
                exp_target = temp_exp_target[:,i]
                
                f1.append(f1_score(exp_pred,exp_target))


            acc = bingo_cnt.float()/float(sample_cnt)
            acc = np.around(acc.numpy(),4)
            best_acc = max(acc,best_acc)
            running_f1=np.mean(f1)
            val_acc.append(acc)
            val_loss.append(running_loss.cpu().detach().numpy())
            val_f1.append(running_f1)
            bacc = np.around(np.mean(baccs),4)
            tqdm.write("[Epoch %d] Validation accuracy:%.4f. bacc:%.4f. Loss:%.3f f1 :%.5f " % (epoch, acc, bacc, running_loss,running_f1))
            tqdm.write("best_acc:" + str(best_acc))
            #cm = confusion_matrix(t_, p_)
            
    #         if running_f1 > bf1 :
    #             bf1 = running_f1
    #             torch.save({'iter': epoch,
    #                         'model_state_dict': model.state_dict(),
    #                          'optimizer_state_dict': optimizer.state_dict(),},
    #                         os.path.join('1_numhead_ensemble/scratch', "epoch"+str(epoch)+"_f1_"+str(bf1)+"_"+model_name+"_.pth"))
    #             plot_confusion_matrix(cm, model_name+str(epoch)+"_cm", normalize = False, target_names = ['neutral', 'anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'other'])
    #             tqdm.write('Model saved.')
    #             to_csv(model_name, val_acc, val_loss, val_f1)
    
    # plt_score(model_name, train_acc,train_loss,train_f1,val_acc,val_loss,val_f1)

def to_csv(model_name, single_score, train_loss, f1):
    data = pd.read_csv("../result/Numhead_ensemble/csv/result.csv", delimiter=',')
    data=data.append({'Model': model_name,
                'Loss' : train_loss[0], 
                'Accuracy':single_score[-1]*100,
                     'F1':f1[-1]*100},ignore_index=True)
    data.to_csv("../result/Numhead_ensemble/csv/result.csv", mode='w',index=False)
def plot_confusion_matrix(cm, plt_name, target_names=None, cmap=None, normalize=True, labels=True, title="DAN" +' confusion matrix'):
    fig1 = plt.gcf()
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if normalize:
        cm = cm / cm.sum(axis=1)[:, np.newaxis]
        
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names)
        plt.yticks(tick_marks, target_names)
    
    if labels:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig("../result/Numhead_ensemble/cm/cm_"+str(plt_name)+".png")
    print("PLT save ..")
    plt.close()
if __name__=="__main__":
    run_training()