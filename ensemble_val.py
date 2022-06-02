import os
import argparse
import pandas as pd
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, confusion_matrix ,f1_score
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision import transforms, datasets
from collections import OrderedDict
from networks.dan_yg import DAN_ab
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import itertools
import glob

path = "./pretrained/" # pretrained model path
weight_list=glob.glob(path+'*.pth') # model path load 
print(weight_list)

class Model() :
    def __init__(self) :
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.all_models = []
        self.load_all_models(weight_list,4)
    def load_all_models(self, weight_list,num_head):
        
        for model_name in weight_list: # make model load list 
            model = DAN_ab(num_head=num_head, num_class=8,pretrained=True)

            weight = model_name
            checkpoint = torch.load(weight)
            if isinstance(model, nn.DataParallel): # GPU 병렬사용 적용 
                model.load_state_dict(checkpoint['model_state_dict'], strict=False) 
            else: # GPU 병렬사용을 안할 경우 
                state_dict = checkpoint['model_state_dict']
                new_state_dict = OrderedDict() 
                for k, v in state_dict.items(): 
                    name = k[7:] # remove `module.` ## module 키 제거 
                    new_state_dict[name] = v 
                model.load_state_dict(new_state_dict, strict = False)

            if ((self.device.type == 'cuda') and (torch.cuda.device_count()>1)):
                print('Multi GPU activate')
                model = nn.DataParallel(model)
                model = model.cuda()
                
            model.to(self.device)
            model.eval()    
            model_ = model
            
            self.all_models.append(model_)
            print('>loaded %s' % model_name)
            print('with num head of ', num_head)

        return self.all_models

    def fit(self, img):
        

        with torch.set_grad_enabled(False):
            img = img.to(self.device)
            outs = None
            for i in self.all_models: # soft voting 구현 
                out, _, _ = i(img)
                if(outs == None):
                    outs=out
                else:
                    outs+=out  

            _, pred = torch.max(outs,1)
            index = pred.cpu()
                

            return index , out.size(0)

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
    #plt.savefig("../cm/base/cm_"+str(plt_name)+".png")
    plt.savefig("../result/cm_"+str(plt_name)+".png")
    print("PLT save ..")
    plt.close()

if __name__ == "__main__":

    model = Model()

    data_transforms_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])   
    print("Generate val data set")
    val_dataset = ImageFolder("./datasets/fulldatasets/val",transform = data_transforms_val)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size =  500,
                                               num_workers = 1,
                                               shuffle = False,  
                                               pin_memory = True)
    with torch.no_grad():
        running_loss = 0.0
        iter_cnt = 0
        bingo_cnt = 0
        sample_cnt = 0
        baccs = []
        running_f1 = 0.0
        p_=[]
        t_=[]
        
        for (imgs, targets) in tqdm(val_loader):
        
            imgs = imgs.float()
            targets_ =targets
            targets = torch.eye(8)[targets]
    
            predicts, size = model.fit(imgs)
            sample_cnt += size
            iter_cnt+=1
            correct_num  = torch.eq(predicts.cpu(),targets.argmax(axis=1))
            bingo_cnt += correct_num.sum().cpu()
            
            for p, t in zip(predicts, targets) :
                p_.append(p.cpu())
                t_.append(t.argmax().cpu())            
            baccs.append(balanced_accuracy_score(targets.cpu().numpy().argmax(axis=1),predicts.cpu().numpy()))
        running_loss = running_loss/iter_cnt   
        
        f1=[]

        temp_exp_pred = np.array(p_)
        temp_exp_target = np.array(t_)
        temp_exp_pred = torch.eye(8)[temp_exp_pred]
        temp_exp_target = torch.eye(8)[temp_exp_target]
        for i in range(0,8):
            exp_pred = temp_exp_pred[:,i]
            exp_target = temp_exp_target[:,i]
            f1.append(f1_score(exp_pred,exp_target))


        acc = bingo_cnt.float()/float(sample_cnt)
        acc = np.around(acc.numpy(),4)
        running_f1=np.mean(f1)

        bacc = np.around(np.mean(baccs),4)
        tqdm.write("Validation accuracy:%.4f. bacc:%.4f. Loss:%.3f f1 :%.5f " % ( acc, bacc, running_loss,running_f1))

        print(f1)
        #cm = confusion_matrix(t_, p_)
        #plot_confusion_matrix(cm, "TL_weak__single_numhead_cm", normalize = False, target_names = ['neutral', 'anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'other'])

