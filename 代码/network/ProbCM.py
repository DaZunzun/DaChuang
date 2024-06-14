import torch
import torch.nn as nn
import math
from torch import distributions
import matplotlib.pyplot as plt
from .unet_cm import *
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class Conv1x1Decoder(nn.Module):        #对Unet的输出加入潜在表示Z（认为是噪声），然后进行若干次1*1卷积操作

    def __init__(self,
                 latent_dim,
                 init_features,
                 num_classes=1,
                 num_1x1_convs=3):
        super().__init__()

        self._num_1x1_convs = num_1x1_convs     #共进行了几次1*1卷积
        self._latent_dim = latent_dim           #单个潜在表示z向量的长度
        self._num_classes = num_classes         #输出结果的通道数
        self._features = init_features          #单个特征矩阵的通道数

        self.net = self._build()                #_num_1x1_convs次1*1卷积构成的网络


    def forward(self, z, unet_features):
        # Unet的输出加入了潜在表示Z（认为是噪声），然后进行若干次1*1卷积操作
        # z的尺寸: [Batch size, latent_dim]: 批次大小 * 单个潜在表示z的长度
        # unet_feature尺寸: [Batch size, input_channels, H, W]: 批次大小 * 每张图片的通道数 * 图片平面的尺寸

        *_, h, w = unet_features.shape         #特征矩阵的平面尺寸
        # 将Z扩展后连接在UNet的输出上，也就是在所有特征矩阵平面每个点上都连接一个向量Z
        out = torch.cat([unet_features, z[..., None, None].tile(dims=(1, 1, h, w))], dim=1)
        logits = self.net(out)                 #通过若干次1*1卷积，输出结果，结果尺寸为_num_classes * h * w

        return logits

    def _build(self):   #若干1*1卷积操作
        layers = []     #layers作为一个封装小模块的大模块
        in_channels = self._latent_dim + self._features      #输入通道数 =  单个潜在表示z向量的长度 + 单个特征矩阵的通道数
        for i in range(self._num_1x1_convs - 1):    #进行（_num_1x1_convs - 1）次卷积（输入与输出通道数一致,卷积核尺寸：1*1,激活函数LeakyReLU）
            layers += [nn.Conv2d(in_channels, in_channels, (1, 1)),
                       nn.LeakyReLU(0.1)]

        layers += [nn.Conv2d(in_channels, self._num_classes, (1, 1))]   # 最后一次1*1卷积，输出通道数为_num_classes


        return nn.Sequential(*layers)


class AxisAlignedConvGaussian(nn.Module):
    # 一种卷积网络，参数化高斯分布
    # 输入一个图像和其真实分割结果
    # 输出高斯分布的均值与log协方差矩阵

    def __init__(self,
                 latent_dim,        #潜在表示的长度
                 in_channels=3,     #输入通道数
                 init_features=32,  #初始的特征矩阵的通道数
                 ):

        super().__init__()
        self._latent_dim = latent_dim
        features = init_features
        self.encoder1 = AxisAlignedConvGaussian._block(in_channels, features)    #两次卷积
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)       #二维平均池化操作，池化核2*2，以上重复5次
        self.encoder2 = AxisAlignedConvGaussian._block(features, 2 * features)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.encoder3 = AxisAlignedConvGaussian._block(features * 2, features * 4)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.encoder4 = AxisAlignedConvGaussian._block(features * 4, features * 8)
        self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.encoder5 = AxisAlignedConvGaussian._block(features * 8, features * 8)
        self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.bottleneck = AxisAlignedConvGaussian._block(features * 8, features * 8)   #底层，进行两次卷积
        self.avg_pool = nn.AdaptiveAvgPool2d(1)                 #自适应平均池化
        self._mu_log_sigma = nn.Conv2d(8 * features, 2 * self._latent_dim, (1, 1))     #1*1卷积,输出结果的通道数为2 * _latent_dim

    def forward(self, x):    #前向传播
        enc1 = self.encoder1(x)                 #两次二维卷积
        enc2 = self.encoder2(self.pool1(enc1))  #池化+两次二维卷积
        enc3 = self.encoder3(self.pool2(enc2))  #池化+两次二维卷积
        enc4 = self.encoder4(self.pool3(enc3))  #池化+两次二维卷积
        enc5 = self.encoder5(self.pool4(enc4))  #池化+两次二维卷积
        bottleneck = self.bottleneck(self.pool5(enc5))   #池化+两次二维卷积

        mu_log_sigma = self._mu_log_sigma(self.avg_pool(bottleneck))   #1*1卷积,输出结果的通道数为2 * _latent_dim
        mu = mu_log_sigma[:, :self._latent_dim, 0, 0]          #高斯分布的均值
        log_sigma = mu_log_sigma[:, self._latent_dim:, 0, 0]   #高斯分布的协方差矩阵取对数
# distributions.Independent(distributions.Normal(loc=mu, scale=torch.exp(log_sigma)), 1)
        return mu, log_sigma


    @staticmethod
    def _block(in_channels, features):     #两次卷积操作
        return nn.Sequential(    #把以下小模块封装成大模块
            nn.Conv2d(                     #二维卷积
                in_channels=in_channels,   #输入张量的channels数
                out_channels=features,     #输出张量的channels数
                kernel_size=(3, 3),        #卷积核大小
                padding=(1, 1),            #图像填充
                bias=True),                #添加偏置参数
            nn.GroupNorm(4, features),     #将channel切分成许多组进行归一化
            nn.LeakyReLU(0.1),             #激活函数
            nn.Conv2d(                     #二维卷积
                in_channels=features,
                out_channels=features,
                kernel_size=(3, 3),
                padding=(1, 1),
                bias=True,
            ),
            nn.GroupNorm(4, features),     #分组做归一化
            nn.LeakyReLU(0.1)              #激活函数
        )

class UNet(nn.Module):             # 一个准标准的Unet

    def __init__(self, in_channels=3, init_features=32):
        super(UNet, self).__init__()

        features = init_features

        self.encoder1 = UNet._block(in_channels, features)       #两次二维卷积（编码）
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)       #平均池化
        self.encoder2 = UNet._block(features, features * 2)      #两次二维卷积
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)       #平均池化
        self.encoder3 = UNet._block(features * 2, features * 4)  #两次二维卷积
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)       #平均池化
        self.encoder4 = UNet._block(features * 4, features * 8)  #两次二维卷积
        self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)       #平均池化
        self.encoder5 = UNet._block(features * 8, features * 8)  #两次二维卷积
        self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)       #平均池化

        self.bottleneck = UNet._block(features * 8, features * 8)    #两次二维卷积

        self.upconv5 = nn.ConvTranspose2d(                                 #反卷积
            features * 8, features * 8, kernel_size=(2, 2), stride=(2, 2)
        )
        self.decoder5 = UNet._block(features * 8 * 2, features * 8)        #两次二维卷积（解码）

        self.upconv4 = nn.ConvTranspose2d(                                 #反卷积
            features * 8, features * 8, kernel_size=(2, 2), stride=(2, 2)
        )
        self.decoder4 = UNet._block(features * 8 * 2, features * 8)        #两次二维卷积
        self.upconv3 = nn.ConvTranspose2d(                                 #反卷积
            features * 8, features * 4, kernel_size=(2, 2), stride=(2, 2)
        )
        self.decoder3 = UNet._block(features * 4 * 2, features * 4)        #两次二维卷积
        self.upconv2 = nn.ConvTranspose2d(                                 #反卷积
            features * 4, features * 2, kernel_size=(2, 2), stride=(2, 2)
        )
        self.decoder2 = UNet._block(features * 2 * 2, features * 2)        #两次二维卷积
        self.upconv1 = nn.ConvTranspose2d(                                 #反卷积
            features * 2, features, kernel_size=(2, 2), stride=(2, 2)
        )
        self.decoder1 = UNet._block(features * 2, features)                #两次二维卷积

    def forward(self, x):
        #x=x.resize_(1,1,256,256)

        enc1 = self.encoder1(x)                 #两次二维卷积        （编码）
        enc2 = self.encoder2(self.pool1(enc1))  #池化+两次二维卷积
        enc3 = self.encoder3(self.pool2(enc2))  #池化+两次二维卷积
        enc4 = self.encoder4(self.pool3(enc3))  #池化+两次二维卷积
        enc5 = self.encoder5(self.pool4(enc4))  #池化+两次二维卷积

        bottleneck = self.bottleneck(self.pool5(enc5))   #两次二维卷积

        dec5 = self.upconv5(bottleneck)         #反卷积          （解码）
        #print(dec5.size(),enc5.size())
        dec5 = torch.cat((dec5, enc5), dim=1)   #特征拼接操作
        dec5 = self.decoder5(dec5)              #两次二维卷积

        dec4 = self.upconv4(dec5)               #反卷积
        dec4 = torch.cat((dec4, enc4), dim=1)   #特征拼接操作
        dec4 = self.decoder4(dec4)              #两次二维卷积

        dec3 = self.upconv3(dec4)               #反卷积
        dec3 = torch.cat((dec3, enc3), dim=1)   #特征拼接操作
        dec3 = self.decoder3(dec3)              #两次二维卷积

        dec2 = self.upconv2(dec3)               #反卷积
        dec2 = torch.cat((dec2, enc2), dim=1)   #特征拼接操作
        dec2 = self.decoder2(dec2)              #两次二维卷积

        dec1 = self.upconv1(dec2)               #反卷积
        dec1 = torch.cat((dec1, enc1), dim=1)   #特征拼接操作
        dec1 = self.decoder1(dec1)              #两次二维卷积    通道数64

        return dec1

    @staticmethod
    def _block(in_channels, features):  #同上
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=(3, 3),
                padding=(1, 1),
                bias=True),
            nn.GroupNorm(4, features),
            nn.LeakyReLU(0.01),
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=(3, 3),
                padding=(1, 1),
                bias=True,
            ),
            nn.GroupNorm(4, features),
            nn.LeakyReLU(0.01)
        )


class ProbCMNet(nn.Module):    #概率Unet（Unet+条件变分自编码器）+ DA网络（cm_layer 得到混淆矩阵confusion matrices）


    def __init__(self,
                 latent_dim,
                 in_channels,
                 num_classes,
                 low_rank=True,
                 num_1x1_convs=3,
                 init_features=32,
                 lq=0.7):
        super().__init__()
        self._latent_dim = latent_dim     #潜在表示z的长度
        self._unet = UNet(in_channels, init_features)    #准标准的unet
        self._f_comb = Conv1x1Decoder(latent_dim, init_features, num_classes, num_1x1_convs)   #拼接z并进行1*1卷积
        # self._prior = AxisAlignedConvGaussian(latent_dim, in_channels, init_features)  # RGB image
        self._posterior = AxisAlignedConvGaussian(latent_dim, in_channels + 1, init_features)  # Image + ground truth  后验编码器,得到高斯分布的参数
        self.low_rank = low_rank  #是否低秩
        self.validation = False   #数据是来自于局部还是全局
        self.lq = lq
        # DA网络
        if self.low_rank is False:   #是否低秩
            self.decoders_noisy_layers = cm_layers(in_channels=latent_dim, norm='in', class_no=num_classes)
            #self.decoders_noisy_layers = gcm_layers(num_classes,256,256)
        else:
            self.decoders_noisy_layers=low_rank_cm_layers(in_channels=latent_dim, norm='in', class_no=num_classes, rank=1)


    def forward(self, *args):

        if self.training:    #训练
                
                img, mask = args
                self.mu, self.log_sigma = self._posterior(torch.cat([img, mask], dim=1))  # 得到高斯分布的 mu均值   log_sigma对数协方差矩阵
                self.q = distributions.Normal(self.mu, torch.exp(self.log_sigma) + 1e-3)  # 建立正态分布
                z_q = self.q.sample()              # 对正态分布采样，得到潜在表示zq
                unet_features = self._unet(img)    # 将图像输入unet,得到64通道的特征矩阵unet_features
                logits = self._f_comb(z_q, unet_features)   #扩展z_q并与unet_features拼接并进行1*1卷积
                h,w = img.shape[-2:]     #图像的平面的尺寸
                y_noisy = self.decoders_noisy_layers(z_q[..., None, None].tile(dims=(1, 1, h, w)).detach())    #z_q扩展之后输入DA网络后得到的值

                return logits,y_noisy

        else:      #测试
            img = args[0]
            batch_size = img.shape[0]    #一批img的数量
            mean = torch.zeros(batch_size, self._latent_dim, device=img.device) # 多元高斯分布的均值 (batch_size * _latent_dim 的0矩阵)
            cov = torch.eye(self._latent_dim, device=img.device)     # 协方差矩阵 (_latent_dim * _latent_dim 的对角矩阵)
            prior = distributions.MultivariateNormal(mean, cov)      # 建立多元高斯分布，即前验编码器
            z_p = prior.sample()         #采样得到潜在表示z_p
            unet_features = self._unet(img)    #img输入unet得到unet_features
            logits = self._f_comb(z_p, unet_features)    #扩展z_p并与各个unet_feature拼接并进行1*1卷积
            if self.validation:       #局部测试
                h,w = img.shape[-2:]
                cm = self.decoders_noisy_layers(z_p[..., None, None].tile(dims=(1, 1, h, w)).detach())   #得到自适应矩阵
                #pred_noisy = self.pred_noise_low_rank(logits, cm）
                pred_noisy = self.pred_noisy(logits, cm)   #计算噪声预测 (predicted noisy segmentation)
                return pred_noisy    #输出预测结果
            else:                    #全局
                return logits        #输出预测结果

    @staticmethod
    def init_weight(m):   #初始化参数

        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):   #如果m是二维卷积 或 二维转置卷积
            torch.nn.init.kaiming_normal_(m.weight)          #初始化参数，服从高斯分布（大概）

            if hasattr(m, 'bias'):                           #m如果包含属性bias
                torch.nn.init.constant_(m.bias, 0)           #用0填充m.bias

        if isinstance(m, nn.GroupNorm):                      #如果m是分组正则化函数
            torch.nn.init.normal_(m.weight)                  #初始化参数，并且初始化参数值符合正态分布
            torch.nn.init.constant_(m.bias, 0)               #用0填充m.bias

    def kl(self):    # L_KL
        kld = torch.mean(-0.5 * torch.sum(1 + 2 * self.log_sigma - self.mu ** 2 - self.log_sigma.exp()**2, dim=1))
        return kld
    
    @staticmethod
    def reconstruction_loss(pred, target):    # L_CE
        return nn.CrossEntropyLoss()(pred, target)
    
    def warmup1(self, pred, target, beta=0.01):  #  L_CE + beta*L_KL  (热身时损失第一个部分)

        kl = self.kl()
        recon_loss = self.reconstruction_loss(pred, target)
        
        return beta * kl + recon_loss
    
    def warmup2(self, cms):     # -L_TR   （热身时损失第二个部分）
        
        if self.low_rank:
            return -trace_low_rank(cms)
        else:
            return -trace_reg(cms)
    
    def elbo1(self, pred, cms, target, alpha=0.7, beta=0.05):  # 损失 : beta * L_KL + L_CE + alpha * L_TR （alpha=1时为热身阶段损失）

        kl = self.kl()                                       # L_KL
        recon_loss = self.reconstruction_loss(pred, target)  # L_CE
        if self.low_rank:                                    # alpha * L_TR
            _,_,TR = noisy_label_loss_low_rank(pred, [cms], [target], alpha)      #低秩版
        else:
            _,_,TR = noisy_label_loss(pred, [cms], [target], alpha)

        #print("Kl  ", kl.item(), "Recon loss  ", recon_loss.item())

        return beta * kl + recon_loss - TR   # beta * L_KL + L_CE + alpha * L_TR
    
    def elbo(self, pred, cms, target, alpha=1, beta=0.05):  # 正式训练阶段总的损失 : beta * L_KL + L_CE + alpha * L_TR + L_NR

        kl = self.kl()
        if self.low_rank:
            recon_loss,CE,TR = noisy_label_loss_low_rank(pred, [cms], [target], alpha)
        else:
            recon_loss,CE,TR = noisy_label_loss(pred, [cms], [target], alpha)
        
        #print("Kl  ", kl.item(), "Recon loss  ", recon_loss.item())

        return beta * kl + recon_loss + TR + Lq_loss(pred,target,self.lq)
    
    def pred_noisy(self, pred, cm):    #pred是全局模型输出的结果, cm是混淆矩阵, 使用矩阵乘积计算预测噪声输出（the predicted noisy segmentation）

        b, c, h, w = pred.size()    #b （batch）图像数量   c （channel）每张图片的通道数     h w  图像平面的尺寸

        #  沿着pred的第一维度 ,对 每个分割输出张量 依次用softmax函数进行归一化
        pred_norm = nn.Softmax(dim=1)(pred)

        # 重新调整pred_norm的形状
        #    b x c x h x w   --->  b x c x h*w  --->  b x h*w x c    ---> b*h*w x c x 1
        pred_norm = pred_norm.view(b, c, h*w).permute(0, 2, 1).contiguous().view(b*h*w, c, 1)

        # cm: 学习了每个噪声标签的混淆矩阵, b x c**2 x h x w
        # label_noisy: 噪声标签, b x h x w

        # 重新调整cm的形状
        # b x c**2 x h x w ---> b*h*w x c x c
        cm = cm.view(b, c ** 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, c * c).view(b * h * w, c, c)

        # 逐行进行归一化:
        cm = cm / cm.sum(1, keepdim=True)

        # 对cm与pred_norm进行矩阵乘法,从而得到the predicted noisy segmentation:
        # cm: b*h*w x c x c
        # pred_norm: b*h*w x c x 1
        pred_noisy = torch.bmm(cm, pred_norm).view(b*h*w, c)
        pred_noisy = pred_noisy.view(b, h*w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)   #调整形状为b x c x h x w

        return pred_noisy
    
    def pred_noise_low_rank(self, pred, cm):  #低秩版 计算预测噪声输出, 与上一个函数不同的是混淆矩阵cm需要重构

        b, c, h, w = pred.size()
        # pred: b x c x h x w
        pred_norm = nn.Softmax(dim=1)(pred)
        # pred_norm: b x c x h x w
        pred_norm = pred_norm.view(b, c, h*w)
        # pred_norm: b x c x h*w
        pred_norm = pred_norm.permute(0, 2, 1).contiguous()
        # pred_norm: b x h*w x c
        pred_norm = pred_norm.view(b*h*w, c)
        # pred_norm: b*h*w x c
        pred_norm = pred_norm.view(b*h*w, c, 1)
        # pred_norm: b*h*w x c x 1


        b, c_r_d, h, w = cm.size()
        r = c_r_d // c // 2

        # reconstruct the full-rank confusion matrix from low-rank approximations:
        cm1 = cm[:, 0:r * c, :, :]
        cm2 = cm[:, r * c:c_r_d-1, :, :]
        scaling_factor = cm[:, c_r_d-1, :, :].view(b, 1, h, w).view(b, 1, h*w).permute(0, 2, 1).contiguous().view(b*h*w, 1, 1)
        cm1_reshape = cm1.view(b, c_r_d // 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, r * c).view(b * h * w, r, c)
        cm2_reshape = cm2.view(b, c_r_d // 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, r * c).view(b * h * w, c, r)
        cm_reconstruct = torch.bmm(cm2_reshape, cm1_reshape)

        # add an identity residual to make approximation easier
        identity_residual = torch.cat(b*h*w*[torch.eye(c, c)]).reshape(b*h*w, c, c).to(device=pred.device, dtype=torch.float32)
        cm_reconstruct_approx = cm_reconstruct + identity_residual*scaling_factor
        cm_reconstruct_approx = cm_reconstruct_approx / cm_reconstruct_approx.sum(1, keepdim=True)

        # calculate noisy prediction from confusion matrix and prediction
        pred_noisy = torch.bmm(cm_reconstruct_approx, pred_norm).view(b*h*w, c)
        pred_noisy = pred_noisy.view(b, h * w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)
        
        return pred_noisy


def trace_reg(cm):   # L_TR  （热身时使用）
    
    b, c, h, w = cm.size()
    c = int(math.sqrt(c))
    cm = cm.view(b, c**2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, c * c).view(b * h * w, c, c)

    # 逐行归一化:
    cm = cm / cm.sum(1, keepdim=True)

    regularisation = torch.trace(torch.transpose(torch.sum(cm, dim=0), 0, 1)).sum() / (b * h * w)

    return regularisation

def trace_low_rank(cm):   # 低秩版的 L_TR  （cm需要重构）（热身时使用）
    
    b, c_r_d, h, w = cm.size()
    c = int(math.sqrt(c_r_d-1))
    r = c_r_d // c // 2

    # reconstruct the full-rank confusion matrix from low-rank approximations:
    cm1 = cm[:, 0:r * c, :, :]
    cm2 = cm[:, r * c:c_r_d-1, :, :]
    scaling_factor = cm[:, c_r_d-1, :, :].view(b, 1, h, w).view(b, 1, h*w).permute(0, 2, 1).contiguous().view(b*h*w, 1, 1)
    cm1_reshape = cm1.view(b, c_r_d // 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, r * c).view(b * h * w, r, c)
    cm2_reshape = cm2.view(b, c_r_d // 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, r * c).view(b * h * w, c, r)
    cm_reconstruct = torch.bmm(cm2_reshape, cm1_reshape)

    # add an identity residual to make approximation easier
    identity_residual = torch.cat(b*h*w*[torch.eye(c, c)]).reshape(b*h*w, c, c).to(device='cuda', dtype=torch.float32)
    cm_reconstruct_approx = cm_reconstruct + identity_residual*scaling_factor
    cm_reconstruct_approx = cm_reconstruct_approx / cm_reconstruct_approx.sum(1, keepdim=True)

    regularisation_ = torch.trace(torch.transpose(torch.sum(cm_reconstruct_approx, dim=0), 0, 1)).sum() / (b * h * w)
    
    return regularisation_


def noisy_label_loss(pred, cms, labels, alpha=0.1):   # L_CE + alpha * L_TR   （正式训练时使用）
    #函数定义了所提出的 迹正则损失函数trace regularised loss function,
    #适用于二进制或多类分割任务。本质上，每个像素都有一个混淆矩阵
    # 输入参数:
        # pred (torch.tensor): 分割网络最后一层的输出张量 without Sigmoid or Softmax
        # cms (list): 混淆矩阵  a list of output tensors for each noisy label, each item contains all of the modelled confusion matrix for each spatial location
        # labels (torch.tensor): 标签
        # alpha (double): 决定正则化强度的超参数
    # 返回值:
        # loss : 总损失 = 主要分割损失 + 正则化项 = L_CE + alpha * L_TR
        # main_loss : 主要分割损失  L_CE
        # regularisation : 正则化项  alpha * L_TR

    main_loss = 0.0
    regularisation = 0.0
    b, c, h, w = pred.size()

    # 沿维度1归一化分割输出张量
    pred_norm = nn.Softmax(dim=1)(pred)

    # b x c x h x w ---> b*h*w x c x 1
    pred_norm = pred_norm.view(b, c, h*w).permute(0, 2, 1).contiguous().view(b*h*w, c, 1)

    for cm, label_noisy in zip(cms, labels):
        # cm: 学习了每个噪声标签的混淆矩阵, b x c**2 x h x w
        # label_noisy: 噪声标签, b x h x w

        # b x c**2 x h x w ---> b*h*w x c x c
        cm = cm.view(b, c ** 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, c * c).view(b * h * w, c, c)

        # 逐行归一化:
        cm = cm / cm.sum(1, keepdim=True)

        # 用矩阵乘法计算噪声预测 (predicted noisy segmentation)
        # cm: b*h*w x c x c
        # pred_norm: b*h*w x c x 1
        pred_noisy = torch.bmm(cm, pred_norm).view(b*h*w, c)
        pred_noisy = pred_noisy.view(b, h*w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)
        loss_current = nn.CrossEntropyLoss(reduction='mean')(pred_noisy, label_noisy.view(b, h, w).long())   # L_CE
        main_loss += loss_current
        regularisation += torch.trace(torch.transpose(torch.sum(cm, dim=0), 0, 1)).sum() / (b * h * w)    #L_TR

    regularisation = alpha*regularisation
    loss = main_loss + regularisation       #L_CE + alpha * L_TR

    return loss, main_loss, regularisation
    
    
def noisy_label_loss_low_rank(pred, cms, labels, alpha):  # 低秩版的 L_CE + alpha * L_TR   （正式训练时使用）
    #该函数定义了所提出的low-rank trace正则化损失函数
    #适用于二进制或多类分割任务。本质上，每个像素都有一个混淆矩阵

    #输入参数:
        #pred (torch.tensor): 分割网络最后一层的输出张量 without Sigmoid or Softmax
        #cms (list): 混淆矩阵  a list of output tensors for each noisy label, each item contains all of the modelled confusion matrix for each spatial location
        #labels (torch.tensor): 标签
        #alpha (double): 决定正则化强度的超参数
    #返回值:
        # loss : 总损失 = 主要分割损失 + 正则化项 = L_CE + alpha * L_TR
        # main_loss : 主要分割损失  L_CE
        # regularisation : 正则化项  alpha * L_TR


    main_loss = 0.0
    regularisation = 0.0
    b, c, h, w = pred.size()
    # pred: b x c x h x w
    pred_norm = nn.Softmax(dim=1)(pred)
    # pred_norm: b x c x h x w
    pred_norm = pred_norm.view(b, c, h*w)
    # pred_norm: b x c x h*w
    pred_norm = pred_norm.permute(0, 2, 1).contiguous()
    # pred_norm: b x h*w x c
    pred_norm = pred_norm.view(b*h*w, c)
    # pred_norm: b*h*w x c
    pred_norm = pred_norm.view(b*h*w, c, 1)
    # pred_norm: b*h*w x c x 1
    #
    for j, (cm, label_noisy) in enumerate(zip(cms, labels)):
        # cm: 学习了每个噪声标签的混淆矩阵, b x c_r_d x h x w, where c_r_d < c
        # label_noisy: 噪声标签, b x h x w

        b, c_r_d, h, w = cm.size()
        r = c_r_d // c // 2

        # 从低秩近似重构全秩混淆矩阵:
        cm1 = cm[:, 0:r * c, :, :]
        cm2 = cm[:, r * c:c_r_d-1, :, :]
        scaling_factor = cm[:, c_r_d-1, :, :].view(b, 1, h, w).view(b, 1, h*w).permute(0, 2, 1).contiguous().view(b*h*w, 1, 1)
        cm1_reshape = cm1.view(b, c_r_d // 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, r * c).view(b * h * w, r, c)
        cm2_reshape = cm2.view(b, c_r_d // 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, r * c).view(b * h * w, c, r)
        cm_reconstruct = torch.bmm(cm2_reshape, cm1_reshape)

        # 添加恒等残差（an identity residual）使近似更容易
        identity_residual = torch.cat(b*h*w*[torch.eye(c, c)]).reshape(b*h*w, c, c).to(device='cuda', dtype=torch.float32)
        cm_reconstruct_approx = cm_reconstruct + identity_residual*scaling_factor
        cm_reconstruct_approx = cm_reconstruct_approx / cm_reconstruct_approx.sum(1, keepdim=True)    ###近似的重构结果

        # 由混淆矩阵（cm_reconstruct_approx）和预测（pred_norm）计算噪声预测（noisy prediction）
        pred_noisy = torch.bmm(cm_reconstruct_approx, pred_norm).view(b*h*w, c)
        pred_noisy = pred_noisy.view(b, h * w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)

        regularisation_ = torch.trace(torch.transpose(torch.sum(cm_reconstruct_approx, dim=0), 0, 1)).sum() / (b * h * w)  #正则化项  L_TR

        loss_current = nn.CrossEntropyLoss(reduction='mean')(pred_noisy, label_noisy.view(b, h, w).long())

        regularisation += regularisation_

        main_loss += loss_current

    regularisation = alpha*regularisation  # 正则化项L_TR * 超参数alpha

    loss = main_loss + regularisation   #主要分割损失 L_CE + alpha * L_TR

    return loss, main_loss, regularisation


             
def Lq_loss(logits, targets, q=0.7):   # L_NR
    p = F.softmax(logits, dim=1)

    Yg = torch.gather(p, 1, targets.unsqueeze(1))

    loss = (1-(Yg**q))/q
    loss = torch.mean(loss)

    return loss