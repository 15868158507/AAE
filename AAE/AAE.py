import torch
from torch import mean, log, rand
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V
import torchvision.datasets as dsets
import torchvision.transforms as transforms

# q(z|x)
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

# p(x|z)
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

# D()
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
dataset = dsets.MNIST(root='./data', 
                      train=True, 
                      transform=transforms.ToTensor(),  
                      download=True)

# Data Loader (Input Pipeline)
data_loader = torch.utils.data.DataLoader(dataset=dataset, 
                                          batch_size=100, 
                                          shuffle=True)


EPS = 1e-15
# 学习率
gen_lr = 0.0001
reg_lr = 0.00005
# 隐变量的维度
z_red_dims = 120
# encoder
Q = Q_net(784,1000,z_red_dims).cpu()
# decoder
P = P_net(784,1000,z_red_dims).cpu()
# discriminator
D_gauss = D_net_gauss(500,z_red_dims).cpu()


#encode/decode 优化器
optim_P = torch.optim.Adam(P.parameters(), lr=gen_lr)
optim_Q_enc = torch.optim.Adam(Q.parameters(), lr=gen_lr)
# GAN部分优化器
optim_Q_gen = torch.optim.Adam(Q.parameters(), lr=reg_lr)
optim_D = torch.optim.Adam(D_gauss.parameters(), lr=reg_lr)





# 数据迭代器
data_iter = iter(data_loader)
iter_per_epoch = len(data_loader)
# total_step = 50000
total_step = 5000

for step in range(total_step):

    if (step+1) % iter_per_epoch == 0:
        data_iter = iter(data_loader)

    # 从MNSIT数据集中拿样本
    images, labels = next(data_iter)
    images, labels = to_var(images.view(images.size(0), -1)), to_var(labels)

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
    z_real_gauss = V(torch.randn(images.size()[0], z_red_dims) * 5.).cpu()
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
