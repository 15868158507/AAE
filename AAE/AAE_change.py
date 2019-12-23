import torch
from torch import mean, log, rand
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V
import torchvision.datasets as dsets
import torchvision.transforms as transforms

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# q(z|x)    编码器
class Q_net(nn.Module):  
    def __init__(self,X_dim,N,z_dim):
        super(Q_net, self).__init__()
        self.lin1 = nn.Linear(X_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3_gauss = nn.Linear(N, z_dim)
    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.25, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.25, training=self.training)
        x = F.relu(x)
        z_gauss = self.lin3_gauss(x)
        return z_gauss

# p(x|z)    解码器
class P_net(nn.Module):  
    def __init__(self,X_dim,N,z_dim):
        super(P_net, self).__init__()
        self.lin1 = nn.Linear(z_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, X_dim)
    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.25, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.25, training=self.training)
        x = self.lin3(x)
        return F.sigmoid(x)

# D()       判别器
class D_net_gauss(nn.Module):  
    def __init__(self,N,z_dim):
        super(D_net_gauss, self).__init__()
        self.lin1 = nn.Linear(z_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, 1)
    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.2, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.2, training=self.training)
        x = F.relu(x)
        return F.sigmoid(self.lin3(x))    

def to_np(x):
    return x.data.cpu().numpy()

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return V(x)  

# MNIST Dataset 
# dataset = dsets.MNIST(root='./data', 
#                       train=True, 
#                       transform=transforms.ToTensor(),  
#                       download=True)

# # Data Loader (Input Pipeline)
# data_loader = torch.utils.data.DataLoader(dataset=dataset, 
#                                           batch_size=100, 
#                                           shuffle=True)
trx_start = 0
trx_datastep  = 700

tex_start = 1000
tex_datastep  = 50

fileData = pd.read_csv(r'E:\pythoncode\meltadata2.txt', sep="\t", dtype=np.float32, header=None)
wholeData = fileData.values

trX, teX = wholeData[trx_start:trx_start + trx_datastep , :13], wholeData[tex_start: tex_start + tex_datastep, :13]

Xtrain = trX.astype(np.float32)
Xtrain = torch.Tensor(Xtrain)

Xtest = teX.astype(np.float32)
Xtest = torch.Tensor(Xtest)

trX_mean, trX_std = trX.mean(), trX.std(ddof=1)

EPS = 1e-15
# 学习率
gen_lr = 0.0001
reg_lr = 0.00005
# 隐变量的维度
z_red_dims = 256
# encoder
Q = Q_net(13,128,z_red_dims).cpu()
# decoder
P = P_net(13,128,z_red_dims).cpu()
# discriminator
D_gauss = D_net_gauss(500,z_red_dims).cpu()


#encode/decode 优化器
optim_P = torch.optim.Adam(P.parameters(), lr=gen_lr)
optim_Q_enc = torch.optim.Adam(Q.parameters(), lr=gen_lr)
# GAN部分优化器
optim_Q_gen = torch.optim.Adam(Q.parameters(), lr=reg_lr)
optim_D = torch.optim.Adam(D_gauss.parameters(), lr=reg_lr)





# 数据迭代器
data_iter = Xtrain
iter_per_epoch = 700
# total_step = 50000
total_step = 1000

for step in range(total_step):

    # 从MNSIT数据集中拿样本
    images = data_iter
    images = to_var(images.view(images.size(0), -1))

    # 把这三个模型的累积梯度清空
    P.zero_grad()
    Q.zero_grad()
    D_gauss.zero_grad()
    ################ Autoencoder部分 ######################
    # encoder 编码x, 生成z
    z_sample = Q(images)
    # decoder 解码z, 生成x'
    X_sample = P(z_sample)
    # 这里计算下autoencoder 的重建误差|x' - x|
    recon_loss = F.binary_cross_entropy(X_sample + EPS, images + EPS)

    # 优化autoencoder
    recon_loss.backward()
    optim_P.step()
    optim_Q_enc.step()

    ################ GAN 部分 #############################

    # 从正太分布中, 采样real gauss(真-高斯分布样本点)
    # z_real_gauss = V(torch.randn(images.size()[0], z_red_dims) * 5.).cpu()
    z_real_gauss = V(torch.normal(trX_mean, trX_std, (images.size()[0], z_red_dims))).cpu()
    # 判别器判别一下真的样本, 得到loss
    D_real_gauss = D_gauss(z_real_gauss)

    # 用encoder 生成假样本
    Q.eval()  # 切到测试形态, 这时候, Q(即encoder)不参与优化
    z_fake_gauss = Q(images)
    # 用判别器判别假样本, 得到loss
    D_fake_gauss = D_gauss(z_fake_gauss)

    # 判别器总误差
    D_loss = -mean(log(D_real_gauss + EPS) + log(1 - D_fake_gauss + EPS))
    print("epoch:", step, 
        "|recon_loss:", recon_loss.detach().numpy(), 
        "|D_loss:", D_loss.detach().numpy())
    
    
    # 优化判别器
    D_loss.backward()
    optim_D.step()

    # encoder充当生成器
    Q.train()  # 切换训练形态, Q(即encoder)参与优化
    z_fake_gauss = Q(images)
    D_fake_gauss = D_gauss(z_fake_gauss)

    G_loss = -mean(log(D_fake_gauss + EPS))

    G_loss.backward()
    # 仅优化Q
    optim_Q_gen.step()

# 训练结束后, 存一下encoder的参数
torch.save(Q.state_dict(), 'Q_encoder_weights.pt')

######################################################
#########               数据打印              #########
######################################################
train_show       = Xtrain.cpu().detach().numpy()             #测试集原始数据
decode_show_trX = P(Q(images)).cpu().detach().numpy()        #测试集解码数据

test_show       = Xtest.cpu().detach().numpy()             #测试集原始数据
decode_show_teX = P(Q(Xtest)).cpu().detach().numpy()        #测试集解码数据


plt.figure(1)
plt.subplot(7,2,1), plt.plot(train_show[:,0]), plt.plot(decode_show_trX[:,0] , color = 'red', linewidth = 1.0, linestyle = '--')
plt.title("数据0",),plt.ylim(0, 1)
plt.subplot(7,2,2), plt.plot(train_show[:,1]), plt.plot(decode_show_trX[:,1] , color = 'red', linewidth = 1.0, linestyle = '--')
plt.title("数据1"),plt.ylim(0, 1)
plt.subplot(7,2,3), plt.plot(train_show[:,2]), plt.plot(decode_show_trX[:,2] , color = 'red', linewidth = 1.0, linestyle = '--')
plt.title("数据2"),plt.ylim(0, 1)
plt.subplot(7,2,4), plt.plot(train_show[:,3]), plt.plot(decode_show_trX[:,3] , color = 'red', linewidth = 1.0, linestyle = '--')
plt.title("数据3"),plt.ylim(0, 1)
plt.subplot(7,2,5), plt.plot(train_show[:,4]), plt.plot(decode_show_trX[:,4] , color = 'red', linewidth = 1.0, linestyle = '--')
plt.title("数据4"),plt.ylim(0, 1)
plt.subplot(7,2,6), plt.plot(train_show[:,5]), plt.plot(decode_show_trX[:,5] , color = 'red', linewidth = 1.0, linestyle = '--'),
plt.title("数据5"),plt.ylim(0, 1)
plt.subplot(7,2,7), plt.plot(train_show[:,6]), plt.plot(decode_show_trX[:,6] , color = 'red', linewidth = 1.0, linestyle = '--')
plt.title("数据6"),plt.ylim(0, 1)

plt.subplot(7,2,8), plt.plot(train_show[:,7]), plt.plot(decode_show_trX[:,7] , color = 'red', linewidth = 1.0, linestyle = '--')
plt.title("数据7"),plt.ylim(0, 1)
plt.subplot(7,2,9), plt.plot(train_show[:,8]), plt.plot(decode_show_trX[:,8] , color = 'red', linewidth = 1.0, linestyle = '--')
plt.title("数据8"),plt.ylim(0, 1)
plt.subplot(7,2,10), plt.plot(train_show[:,9]), plt.plot(decode_show_trX[:,9] , color = 'red', linewidth = 1.0, linestyle = '--')
plt.title("数据9"),plt.ylim(0, 1)
plt.subplot(7,2,11), plt.plot(train_show[:,10]), plt.plot(decode_show_trX[:,10] , color = 'red', linewidth = 1.0, linestyle = '--')
plt.title("数据10"),plt.ylim(0, 1)
plt.subplot(7,2,12), plt.plot(train_show[:,11]), plt.plot(decode_show_trX[:,11] , color = 'red', linewidth = 1.0, linestyle = '--')
plt.title("数据11"),plt.ylim(0, 1)
plt.subplot(7,2,13), plt.plot(train_show[:,12]), plt.plot(decode_show_trX[:,12] , color = 'red', linewidth = 1.0, linestyle = '--')
plt.title("数据12"),plt.ylim(0, 1)

plt.figure(2)
plt.subplot(7,2,1), plt.plot(test_show[:,0]), plt.plot(decode_show_teX[:,0] , color = 'red', linewidth = 1.0, linestyle = '--')
plt.title("数据0",),plt.ylim(0, 1)
plt.subplot(7,2,2), plt.plot(test_show[:,1]), plt.plot(decode_show_teX[:,1] , color = 'red', linewidth = 1.0, linestyle = '--')
plt.title("数据1"),plt.ylim(0, 1)
plt.subplot(7,2,3), plt.plot(test_show[:,2]), plt.plot(decode_show_teX[:,2] , color = 'red', linewidth = 1.0, linestyle = '--')
plt.title("数据2"),plt.ylim(0, 1)
plt.subplot(7,2,4), plt.plot(test_show[:,3]), plt.plot(decode_show_teX[:,3] , color = 'red', linewidth = 1.0, linestyle = '--')
plt.title("数据3"),plt.ylim(0, 1)
plt.subplot(7,2,5), plt.plot(test_show[:,4]), plt.plot(decode_show_teX[:,4] , color = 'red', linewidth = 1.0, linestyle = '--')
plt.title("数据4"),plt.ylim(0, 1)
plt.subplot(7,2,6), plt.plot(test_show[:,5]), plt.plot(decode_show_teX[:,5] , color = 'red', linewidth = 1.0, linestyle = '--'),
plt.title("数据5"),plt.ylim(0, 1)
plt.subplot(7,2,7), plt.plot(test_show[:,6]), plt.plot(decode_show_teX[:,6] , color = 'red', linewidth = 1.0, linestyle = '--')
plt.title("数据6"),plt.ylim(0, 1)

plt.subplot(7,2,8), plt.plot(test_show[:,7]), plt.plot(decode_show_teX[:,7] , color = 'red', linewidth = 1.0, linestyle = '--')
plt.title("数据7"),plt.ylim(0, 1)
plt.subplot(7,2,9), plt.plot(test_show[:,8]), plt.plot(decode_show_teX[:,8] , color = 'red', linewidth = 1.0, linestyle = '--')
plt.title("数据8"),plt.ylim(0, 1)
plt.subplot(7,2,10), plt.plot(test_show[:,9]), plt.plot(decode_show_teX[:,9] , color = 'red', linewidth = 1.0, linestyle = '--')
plt.title("数据9"),plt.ylim(0, 1)
plt.subplot(7,2,11), plt.plot(test_show[:,10]), plt.plot(decode_show_teX[:,10] , color = 'red', linewidth = 1.0, linestyle = '--')
plt.title("数据10"),plt.ylim(0, 1)
plt.subplot(7,2,12), plt.plot(test_show[:,11]), plt.plot(decode_show_teX[:,11] , color = 'red', linewidth = 1.0, linestyle = '--')
plt.title("数据11"),plt.ylim(0, 1)
plt.subplot(7,2,13), plt.plot(test_show[:,12]), plt.plot(decode_show_teX[:,12] , color = 'red', linewidth = 1.0, linestyle = '--')
plt.title("数据12"),plt.ylim(0, 1)

plt.show()
