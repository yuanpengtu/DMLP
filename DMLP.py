import os
import math
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import solve
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as lr_scheduler
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
def kl_soft(pred, soft_targets,reduce=True):
    kl=F.kl_div(F.log_softmax(pred, dim=1), soft_targets, reduce=False)
    if reduce:
       return torch.mean(torch.sum(kl,dim=1))
    else:
       return torch.sum(kl,1)
def adjust_learning_rate(optimizer, epoch):
    lr = 0.006
    if False:
        eta_min = lr * (0.1 ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / v2)) / 2
    else:
        steps = np.sum(epoch > np.asarray(3))#3
        if steps > 0:
            lr = lr * (0.1 ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr



num_classes=10
train_num=49000#Training set size 
val_num=1000#Validation set size
test_num=10000
batchsize=7000#Batchsize for IPC
iteration=train_num//batchsize
batchsize_fc=500##Batchsize for EAC
iteration_fc=batchsize//batchsize_fc
maxacc=0
epoch=100#Training epochs
weight_conf=0.4#Loss weight for confidence penalty loss
weight_prior=2200#Loss weight for class balance loss
epoch_T=20#Period for EAC update

path='./labels_cifar10/symmetric20/'#Path for features and labels
features_v1=np.load(path+'trainfeatures.npy')#Training set features
test_features=np.load(path+'testfeatures.npy')#Test set features
features_v2=np.load(path+'trainfeatures.npy')
labels_correct=torch.from_numpy(np.load(path+'correctlabels.npy')).cuda()#Correct training set labels

idx_shuffle = torch.randperm(train_num+val_num)#Divide training set and validation set
idx_to_train=idx_shuffle[:train_num]
idx_to_meta=idx_shuffle[train_num:]

data_list_val = {}
for j in range(num_classes):
    data_list_val[j] = [i for i, label in enumerate(labels_correct[idx_to_meta]) if label == j]
    print("ratio class", j, ":", len(data_list_val[j])/1000*100)

train_features_v1=torch.from_numpy(features_v1[idx_to_train])
val_features_v1=torch.from_numpy(features_v1[idx_to_meta]).cuda()
test_features_v1=torch.from_numpy(test_features)
train_features_v2=torch.from_numpy(features_v2[idx_to_train])
val_features_v2=torch.from_numpy(features_v2[idx_to_meta]).cuda()
test_features_v2=torch.from_numpy(test_features)


test_labels_v2=torch.from_numpy(np.load(path+'testlabels.npy'))
train_labels_v2=torch.from_numpy(np.load(path+'trainlabels.npy')[idx_to_train]).long().cuda()

train_all=torch.from_numpy(np.load(path+'trainlabels.npy')).long()#.cuda()
correct_all=labels_correct.cpu()
noise_or_not=[train_all[idx]==correct_all[idx] for idx in range(50000)]
print("ratio:",np.sum(np.array(noise_or_not))/50000)

train_labels_v2=F.one_hot(train_labels_v2,num_classes)

val_labels_v2=torch.from_numpy(np.load(path+'correctlabels.npy')[idx_to_meta]).long().cuda()
vval_labels_maxv2=val_labels_v2
val_labels_v2=F.one_hot(val_labels_v2,num_classes)
train_labels_v2=Variable(train_labels_v2.float(),requires_grad=True)
optimizer_v2 = torch.optim.Adam([train_labels_v2], 0.01, betas=(0.5, 0.999), weight_decay=3e-4)


test_labels=torch.from_numpy(np.load(path+'testlabels.npy'))
train_labels_v1=torch.from_numpy(np.load(path+'trainlabels.npy')[idx_to_train]).long().cuda()
train_labels_v1=F.one_hot(train_labels_v1,num_classes)

val_labels_v1=torch.from_numpy(np.load(path+'correctlabels.npy')[idx_to_meta]).long().cuda()
val_labels_argv1=val_labels_v1
val_labels_v1=F.one_hot(val_labels_v1,num_classes)
train_labels_v1=Variable(train_labels_v1.float(),requires_grad=True)
optimizer_v1 = torch.optim.Adam([train_labels_v1], 0.01, betas=(0.5, 0.999), weight_decay=3e-4)#0.03

loss_mse=torch.nn.MSELoss(reduce=False, size_average=False)
feat_dim=train_features_v2.shape[1]
model_fcv2=nn.Sequential(nn.Linear(feat_dim, num_classes),nn.Softmax()).cuda()
loss_fc=nn.CrossEntropyLoss()
loss_w2=nn.CrossEntropyLoss()
optimizer_fcv2 = torch.optim.Adam(model_fcv2.parameters(), 0.03, betas=(0.5, 0.999), weight_decay=3e-4)
lr_schedulerv2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_fcv2, 20, eta_min=0.0002)

feat_dim=train_features_v1.shape[1]
model_fcv1=nn.Sequential(nn.Linear(feat_dim, num_classes),nn.Softmax()).cuda()
optimizer_fcv1 = torch.optim.Adam(model_fcv1.parameters(), 0.03, betas=(0.5, 0.999), weight_decay=3e-4)
lr_schedulerv1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_fcv1, 20, eta_min=0.0002)
corrected_labels=torch.zeros([50000])
val_features_v1=val_features_v1.detach()
val_labels_v1=val_labels_v1.detach()
val_features_v2=val_features_v2.detach()
val_labels_v2=val_labels_v2.detach()

for i in range(epoch):
    lrv1 = adjust_learning_rate(optimizer_fcv1, i)
    lrv2 = adjust_learning_rate(optimizer_fcv2, i)
    for j in range(iteration): 
        if batchsize*(j+1)>train_num:
           end=train_num
        else:
           end=batchsize*(j+1)
        
        batchfeat_v1=train_features_v1[batchsize*j:end]
        batchfeat_v1=batchfeat_v1.cuda()
        batchlabel_v1=train_labels_v1[batchsize*j:end].detach()

        batchfeat_v2=train_features_v2[batchsize*j:end]
        batchfeat_v2=batchfeat_v2.cuda()
        batchlabel_v2=train_labels_v2[batchsize*j:end].detach()

        model_fcv1=model_fcv1.cuda().train()
        optimizer_v1.zero_grad()
        #IPC Training
        W1_v1=torch.mm(torch.mm(torch.inverse(torch.mm(batchfeat_v1.transpose(1,0), batchfeat_v1)),batchfeat_v1.transpose(1,0)), train_labels_v1[batchsize*j:end])
        loss_v1=loss_w2(torch.mm(val_features_v1,W1_v1),val_labels_argv1.detach().long().cuda())+torch.sum(loss_mse(torch.mm(val_features_v1,W1_v1),val_labels_v1.detach().float()))
        prior = torch.ones(num_classes)/num_classes
        prior = prior.cuda()
        pred_mean = torch.softmax(torch.mm(batchfeat_v1,W1_v1), dim=1).mean(0)
        penalty_wv1 = torch.sum(prior*torch.log(prior/pred_mean))
        loss_v1+=penalty_wv1
        loss_v1.backward()
        optimizer_v1.step()
        
        model_fcv2=model_fcv2.cuda().train()
        optimizer_v2.zero_grad()
        W1_v2=torch.mm(torch.mm(torch.inverse(torch.mm(batchfeat_v2.transpose(1,0), batchfeat_v2)),batchfeat_v2.transpose(1,0)), train_labels_v2[batchsize*j:end])
        loss_v2=loss_w2(torch.mm(val_features_v2,W1_v2),vval_labels_maxv2.detach().long().cuda())+torch.sum(loss_mse(torch.mm(val_features_v2,W1_v2),val_labels_v2.detach().float()))
        prior = torch.ones(num_classes)/num_classes
        prior = prior.cuda()
        pred_mean = torch.softmax(torch.mm(batchfeat_v2,W1_v2), dim=1).mean(0)
        penalty_wv2 = torch.sum(prior*torch.log(prior/pred_mean))
        loss_v2+=penalty_wv2
        loss_v2.backward()
        optimizer_v2.step()


        correct=labels_correct[idx_to_train[batchsize*j:end]]
        pure_or_not=[torch.argmax(batchlabel_v2[idx])==correct[idx] for idx in range(end-batchsize*j)]
        pure_or_not=torch.Tensor(pure_or_not)#.shape)
        acc_ori_v2=torch.sum(pure_or_not)/(end-batchsize*j)*100

        pure_or_not=[torch.argmax(batchlabel_v1[idx])==correct[idx] for idx in range(end-batchsize*j)]
        pure_or_not=torch.Tensor(pure_or_not)#.shape)
        acc_ori_v1=torch.sum(pure_or_not)/(end-batchsize*j)*100

        newlabel_v1=train_labels_v1[batchsize*j:end].detach()
        newlabel_v2=train_labels_v2[batchsize*j:end].detach()
        #EAC Training
        for p in range(iteration_fc):
            if batchsize_fc*(p+1)>batchsize:
               continue
            else:
               end_fc=batchsize_fc*(p+1)

            newlabel_fcv1=newlabel_v1[batchsize_fc*p:end_fc]
            newlabel_fcv2=newlabel_v2[batchsize_fc*p:end_fc]
            batchfeat_fcv1=batchfeat_v1[batchsize_fc*p:end_fc].detach()
            batchfeat_fcv2=batchfeat_v2[batchsize_fc*p:end_fc].detach()

            optimizer_fcv1.zero_grad()
            predict_fcv1=model_fcv1(batchfeat_fcv1)
            labels_v1=torch.Tensor([torch.argmax(newlabel_fcv1[idx]) for idx in range(end_fc-batchsize_fc*p)])
            loss_newv1=loss_fc(predict_fcv1, labels_v1.cuda().detach().long())
            predict_fcv1 = torch.softmax(predict_fcv1, dim=1)
            conf_penalty = torch.mean(torch.sum(predict_fcv1.log() * predict_fcv1, dim=1))
            pred_mean = torch.softmax(predict_fcv1, dim=1).mean(0)
            penalty_wv1 = torch.sum(prior*torch.log(prior/pred_mean))
            loss_newv1+=weight_prior*penalty_wv1


            loss_newv1+=conf_penalty*weight_conf
            loss_newv1.backward()
            optimizer_fcv1.step()


            optimizer_fcv2.zero_grad()
            predict_fcv2=model_fcv2(batchfeat_fcv2)
            labels_v2=torch.Tensor([torch.argmax(newlabel_fcv2[idx]) for idx in range(end_fc-batchsize_fc*p)])
            loss_newv2=loss_fc(predict_fcv2, labels_v2.cuda().detach().long())
            predict_fcv2 = torch.softmax(predict_fcv2, dim=1)
            conf_penalty = torch.mean(torch.sum(predict_fcv2.log() * predict_fcv2, dim=1))
            pred_mean = torch.softmax(predict_fcv2, dim=1).mean(0)
            penalty_wv2 = torch.sum(prior*torch.log(prior/pred_mean))
            loss_newv2+=weight_prior*penalty_wv2
            loss_newv2+=conf_penalty*weight_conf
            loss_newv2.backward()
            optimizer_fcv2.step()
        predict_newfcv1=model_fcv1(batchfeat_v1)
        predict_newfcv2=model_fcv2(batchfeat_v2)

        pure_or_not=[torch.argmax(predict_newfcv1[idx])==correct[idx] for idx in range(end-batchsize*j)]
        pure_or_not=torch.Tensor(pure_or_not)#.shape)
        acc_fcv1=torch.sum(pure_or_not)/(end-batchsize*j)*100


        pure_or_not=[torch.argmax(predict_newfcv2[idx])==correct[idx] for idx in range(end-batchsize*j)]
        pure_or_not=torch.Tensor(pure_or_not)#.shape)
        acc_fcv2=torch.sum(pure_or_not)/(end-batchsize*j)*100

        if i%epoch_T==0 and i!=0 and i==epoch-1:
           with torch.no_grad():
                w_labels_v1=train_labels_v1[batchsize*j:end]
                w_labels_v2=train_labels_v2[batchsize*j:end]

                train_labels_v1[batchsize*j:end]=predict_newfcv2.detach()
                train_labels_v2[batchsize*j:end]=predict_newfcv1.detach()

                outputs = (predict_newfcv2+predict_newfcv1)/2.
                outputs = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)

                outputs_w = (w_labels_v1+w_labels_v2)/2.
                outputs_w = F.softmax(outputs_w, dim=1)
                _, predicted_w = torch.max(outputs_w.data, 1)
               
        print(datetime.datetime.now(), '------- Epoch [%d/%d] label Accuracy: accv1 %.4f, accv2 %.4f, acc fcv1 %.4f, acc fcv2 %.4f, lossv1 %.4f, lossv2 %.4f, loss_fcv1 %.4f, loss_fcv2 %.4f  %%' %
              (i, j, acc_ori_v1, acc_ori_v2, acc_fcv1, acc_fcv2, loss_v1, loss_v2, loss_newv1, loss_newv2))

    with torch.no_grad():
         test_predictv1=torch.mm(test_features_v1, W1_v1.cpu())
         predict_labelv1=[torch.argmax(test_predictv1[idx])==test_labels[idx] for idx in range(test_num)]
         predict_labelv1=torch.Tensor(predict_labelv1)#.shape)
         acc_testv1=torch.sum(predict_labelv1)/test_num*100

         model_fcv1.eval()
         model_fcv1=model_fcv1.cpu()
         test_predict_fcv1=model_fcv1(test_features_v1)
         test_predict_fcv1=test_predict_fcv1.detach().cpu()
         predict_label_fcv1=[torch.argmax(test_predict_fcv1[idx])==test_labels[idx] for idx in range(test_num)]
         predict_label_fcv1=torch.Tensor(predict_label_fcv1)#.shape)
         acc_test_fcv1=torch.sum(predict_label_fcv1)/test_num*100


         test_predictv2=torch.mm(test_features_v2, W1_v2.cpu())
         predict_labelv2=[torch.argmax(test_predictv2[idx])==test_labels[idx] for idx in range(test_num)]
         predict_labelv2=torch.Tensor(predict_labelv2)#.shape)
         acc_testv2=torch.sum(predict_labelv2)/test_num*100

         model_fcv2.eval()
         model_fcv2=model_fcv2.cpu()
         test_predict_fcv2=model_fcv2(test_features_v2)
         test_predict_fcv2=test_predict_fcv2.detach().cpu()
         predict_label_fcv2=[torch.argmax(test_predict_fcv2[idx])==test_labels[idx] for idx in range(test_num)]
         predict_label_fcv2=torch.Tensor(predict_label_fcv2)#.shape)
         acc_test_fcv2=torch.sum(predict_label_fcv2)/test_num*100

         outputs = (test_predict_fcv1+test_predict_fcv2)/2.
         outputs = F.softmax(outputs, dim=1)
         _, predicted = torch.max(outputs.data, 1)   
         ensemacc=[predicted[idx]==test_labels[idx] for idx in range(test_num)]
         ensemacc=torch.sum(torch.Tensor(ensemacc))/test_num*100
         

         maxacc=max(maxacc, ensemacc)
         print("===================================================")
         print("max accuracy:", maxacc ,"ensemble accuracy:", ensemacc, "test accuracyv1:", acc_testv1, "test accuracyv2:", acc_testv2, "test accuracy linearv1:", acc_test_fcv1, "test accuracy linearv2:", acc_test_fcv2)
         print("===================================================")


