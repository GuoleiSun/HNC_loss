import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

class ChannelGate_dense(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'],block_id=0):
        super(ChannelGate_dense, self).__init__()
        self.gate_channels = gate_channels
        if block_id!=0:
            in_channels=2*gate_channels
            # print("bloch_id: %d in_channels:%d ",(block_id,in_channels))
        else:
            in_channels=gate_channels
        print("bloch_id: %d in_channels:%d gate_channels: %d",(block_id,in_channels,gate_channels))
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
        self.block_id=block_id
    def forward(self, x, atten):
        channel_att_sum = None
        channel_ave=None
        channel_max=None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                avg_pool=avg_pool.squeeze(3).squeeze(2)
                channel_ave=avg_pool.clone()
                if self.block_id==0:
                    channel_att_raw = self.mlp( avg_pool )
                elif self.block_id!=0:
                    avg_pool_combine=torch.cat((avg_pool,atten[:,:,0]),dim=1)
                    channel_att_raw = self.mlp( avg_pool_combine )
                # print(channel_att_raw.size())
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                max_pool=max_pool.squeeze(3).squeeze(2)
                channel_max=max_pool.clone()
                if self.block_id==0:
                    channel_att_raw = self.mlp( max_pool )
                elif self.block_id!=0:
                    max_pool_combine=torch.cat((max_pool,atten[:,:,1]),dim=1)
                    channel_att_raw = self.mlp( max_pool_combine )                    
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        channel_ave=channel_ave.unsqueeze(2)
        channel_max=channel_max.unsqueeze(2)
        # print(torch.cat((channel_ave,channel_max),dim=2).size())
        return x * scale, torch.cat((channel_ave,channel_max),dim=2)

class ChannelGate_dense2(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'],block_id=0):
        super(ChannelGate_dense2, self).__init__()
        self.gate_channels = gate_channels
        if block_id!=0:
            in_channels=2*gate_channels
            # print("bloch_id: %d in_channels:%d ",(block_id,in_channels))
        else:
            in_channels=gate_channels
        print("bloch_id: %d in_channels:%d gate_channels: %d",(block_id,in_channels,gate_channels))
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.BatchNorm1d(in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
        self.block_id=block_id
    def forward(self, x, atten):
        channel_att_sum = None
        channel_ave=None
        channel_max=None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                avg_pool=avg_pool.squeeze(3).squeeze(2)
                if self.block_id==0:
                    channel_att_raw = self.mlp( avg_pool )
                elif self.block_id!=0:
                    avg_pool_combine=torch.cat((avg_pool,atten[:,:,0]),dim=1)
                    channel_att_raw = self.mlp( avg_pool_combine )
                channel_ave=channel_att_raw.clone()
                # print(channel_att_raw.size())
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                max_pool=max_pool.squeeze(3).squeeze(2)
                if self.block_id==0:
                    channel_att_raw = self.mlp( max_pool )
                elif self.block_id!=0:
                    max_pool_combine=torch.cat((max_pool,atten[:,:,1]),dim=1)
                    channel_att_raw = self.mlp( max_pool_combine ) 
                channel_max=channel_att_raw.clone()                   
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        channel_ave=channel_ave.unsqueeze(2)
        channel_max=channel_max.unsqueeze(2)
        # print(torch.cat((channel_ave,channel_max),dim=2).size())
        return x * scale, torch.cat((channel_ave,channel_max),dim=2)

class ChannelGate2(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate2, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
        self.conv=nn.Conv2d(2, 1, kernel_size=1, bias=False)
    def forward(self, x):
        channel_att_sum = None
        # for pool_type in self.pool_types:
            # if pool_type=='avg':
        avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_raw_avg = self.mlp( avg_pool )
            # elif pool_type=='max':
        max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_raw_max = self.mlp( max_pool )
        channel_att_raw_avg=channel_att_raw_avg.unsqueeze(1).unsqueeze(3)
        channel_att_raw_max=channel_att_raw_max.unsqueeze(1).unsqueeze(3)

        # print(channel_att_raw_max.size(),channel_att_raw_avg.size())
        channel_atts=torch.cat((channel_att_raw_avg,channel_att_raw_max),1)
        print(channel_atts.size())
        channel_att_sum=self.conv(channel_atts)
        channel_att_sum=channel_att_sum.squeeze()
        # print(channel_att_sum.size())
            # elif pool_type=='lp':
                # lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                # channel_att_raw = self.mlp( lp_pool )
            # elif pool_type=='lse':
                # LSE pool only
                # lse_pool = logsumexp_2d(x)
                # channel_att_raw = self.mlp( lse_pool )

            # if channel_att_sum is None:
            #     channel_att_sum = channel_att_raw
            # else:
            #     channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

class ChannelGate_group(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=2, pool_types=['avg', 'max'],groups=32):
        super(ChannelGate_group, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(groups, groups // reduction_ratio),
            nn.ReLU(),
            nn.Linear(groups // reduction_ratio, groups)
            )
        self.pool_types = pool_types
        self.groups=groups
    def forward(self, x):
        b,c,h,w=x.size()
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                y=x.unsqueeze(0)
                avg_pool = F.adaptive_avg_pool3d( y, (self.groups, 1,1))
                avg_pool=avg_pool.squeeze(0)
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                y=x.unsqueeze(0)
                max_pool = F.adaptive_max_pool3d( y, (self.groups, 1,1))
                max_pool=max_pool.squeeze(0)
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
        channel_att_sum=channel_att_sum.unsqueeze(1)
        channel_att_sum=F.upsample(channel_att_sum,c)
        channel_att_sum=channel_att_sum.squeeze(1)
        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

class ChannelGate_group_no_upsample(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=2, pool_types=['avg', 'max'],groups=32):
        super(ChannelGate_group_no_upsample, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(groups, groups // reduction_ratio),
            nn.ReLU(),
            nn.Linear(groups // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
        self.groups=groups
    def forward(self, x):
        b,c,h,w=x.size()
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                y=x.unsqueeze(0)
                avg_pool = F.adaptive_avg_pool3d( y, (self.groups, 1,1))
                avg_pool=avg_pool.squeeze(0)
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                y=x.unsqueeze(0)
                max_pool = F.adaptive_max_pool3d( y, (self.groups, 1,1))
                max_pool=max_pool.squeeze(0)
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
        # channel_att_sum=channel_att_sum.unsqueeze(1)
        # channel_att_sum=F.upsample(channel_att_sum,c)
        # channel_att_sum=channel_att_sum.squeeze(1)
        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


class CBAM_dense(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, block_id=0,pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM_dense, self).__init__()
        self.ChannelGate = ChannelGate_dense(gate_channels, reduction_ratio, pool_types,block_id=block_id)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x,atten):
        x_out,atten = self.ChannelGate(x,atten)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out,atten


class CBAM_dense2(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, block_id=0,pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM_dense2, self).__init__()
        self.ChannelGate = ChannelGate_dense2(gate_channels, 8, pool_types,block_id=block_id)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x,atten):
        x_out,atten = self.ChannelGate(x,atten)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out,atten


class CBAM_channel_group_no_upsample(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM_channel_group_no_upsample, self).__init__()
        self.ChannelGate = ChannelGate_group_no_upsample(gate_channels, 16*256//gate_channels, pool_types,256*256//gate_channels)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

class CBAM_channel_group3(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM_channel_group3, self).__init__()
        # if gate_channels==256 or gate_channels==512 or gate_channels==1024:
        reduction_ratio=16
        groups=gate_channels
        # elif gate_channels==2048:
        #     reduction_ratio=16
        #     groups=256
        # else:
        #     print("error")
        print("grups: ", groups)
        print("reduction_ratio: ",reduction_ratio)
        self.ChannelGate = ChannelGate_group(gate_channels, reduction_ratio, pool_types,groups)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

class CBAM2(nn.Module):
    ## using ChannelGate2
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM2, self).__init__()
        self.ChannelGate = ChannelGate2(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

class CBAM_new(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM_new, self).__init__()
        # self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        # self.no_spatial=no_spatial
        # if not no_spatial:
        #     self.SpatialGate = SpatialGate()

        # self.avg_pool = nn.AdaptiveAvgPool3d(self.ave_size)
        self.conv1=nn.Conv2d(gate_channels, gate_channels, kernel_size=1, bias=False)
        self.conv2=nn.Conv2d(int(gate_channels/4), int(gate_channels/4), kernel_size=1, bias=False)
        # self.conv3=nn.Conv2d(gate_channels, gate_channels, kernel_size=1, bias=False)
    def forward(self, x):
        _,c,h,w=x.size()
        c2,h2,w2=int(c/4),int(h/4),int(w/4)
        y=self.conv1(x)
        y=y.unsqueeze(0)
        y=F.adaptive_avg_pool3d(y,(c2,h2,w2))
        y=y.squeeze(0)
        y=self.conv2(y)
        y=y.unsqueeze(0)
        y=F.upsample(y,size=(c,h,w))
        y=y.squeeze(0)
        # y=self.conv3(y)
        y=F.sigmoid(y)

        return y*x


class CBAM_new_oneconv(nn.Module):
    ## one conv
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM_new_oneconv, self).__init__()
        # self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        # self.no_spatial=no_spatial
        # if not no_spatial:
        #     self.SpatialGate = SpatialGate()

        # self.avg_pool = nn.AdaptiveAvgPool3d(self.ave_size)
        # self.conv1=nn.Conv2d(gate_channels, gate_channels, kernel_size=1, bias=False)
        self.conv2=nn.Conv2d(int(gate_channels/4), int(gate_channels/4), kernel_size=1, bias=False)
        # self.conv3=nn.Conv2d(gate_channels, gate_channels, kernel_size=1, bias=False)
    def forward(self, x):
        _,c,h,w=x.size()
        c2,h2,w2=int(c/4),int(h/4),int(w/4)
        y=x
        y=y.unsqueeze(0)
        y=F.adaptive_avg_pool3d(y,(c2,h2,w2))
        y=y.squeeze(0)
        y=self.conv2(y)
        y=y.unsqueeze(0)
        y=F.upsample(y,size=(c,h,w))
        y=y.squeeze(0)
        # y=self.conv3(y)
        y=F.sigmoid(y)

        return y*x

import math
class CBAM_new_multiscale_oneconv2(nn.Module):
    def __init__(self, gate_channels,reduction_ratio=16, group=32, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM_new_multiscale_oneconv2, self).__init__()
        # self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        # self.no_spatial=no_spatial
        # if not no_spatial:
        #     self.SpatialGate = SpatialGate()

        # self.avg_pool = nn.AdaptiveAvgPool3d(self.ave_size)
        # group=32
        
        self.group=80-16*int(math.log(gate_channels/256,2))
        print(self.group)
        self.spa_ratio=int(gate_channels/256)

        # self.conv1=nn.Conv2d(gate_channels, gate_channels, kernel_size=1, bias=False)
        self.conv2=nn.Conv2d(self.group, self.group, kernel_size=1, bias=False)
        # self.conv3=nn.Conv2d(gate_channels, gate_channels, kernel_size=1, bias=False)
    def forward(self, x):
        _,c,h,w=x.size()
        # c2,h2,w2=int(c/4),int(h/4),int(w/4)
        c2,h2,w2=self.group,min([self.spa_ratio,h]),min([self.spa_ratio,w])
        # y=self.conv1(x)
        y=x
        y=y.unsqueeze(0)
        y=F.adaptive_avg_pool3d(y,(c2,h2,w2))
        y=y.squeeze(0)
        y=self.conv2(y)
        y=y.unsqueeze(0)
        y=F.upsample(y,size=(c,h,w))
        y=y.squeeze(0)
        # y=self.conv3(y)
        y=F.sigmoid(y)

        return y*x


class CBAM_new_group32(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, group=32, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM_new_group32, self).__init__()
        # self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        # self.no_spatial=no_spatial
        # if not no_spatial:
        #     self.SpatialGate = SpatialGate()

        # self.avg_pool = nn.AdaptiveAvgPool3d(self.ave_size)
        # group=32
        print(group)
        self.group=group
        # self.conv1=nn.Conv2d(gate_channels, gate_channels, kernel_size=1, bias=False)
        self.conv2=nn.Conv2d(group, group, kernel_size=1, bias=False)
        # self.conv3=nn.Conv2d(gate_channels, gate_channels, kernel_size=1, bias=False)
    def forward(self, x):
        _,c,h,w=x.size()
        # c2,h2,w2=int(c/4),int(h/4),int(w/4)
        c2,h2,w2=self.group,int(h/4),int(w/4)
        # y=self.conv1(x)
        y=x
        y=y.unsqueeze(0)
        y=F.adaptive_avg_pool3d(y,(c2,h2,w2))
        y=y.squeeze(0)
        y=self.conv2(y)
        y=y.unsqueeze(0)
        y=F.upsample(y,size=(c,h,w))
        y=y.squeeze(0)
        # y=self.conv3(y)
        y=F.sigmoid(y)

        return y*x

class CBAM_new_group32_no_upsample(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, group=32, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM_new_group32_no_upsample, self).__init__()
        print(group)
        self.group=group
        # self.conv1=nn.Conv2d(gate_channels, gate_channels, kernel_size=1, bias=False)
        # self.conv2=nn.Conv2d(group, group, kernel_size=1, bias=False)
        spatial_size=int(56*256/gate_channels)
        if spatial_size==28 or spatial_size==56:
            self.conv2=nn.Sequential(
                        nn.ConvTranspose2d(group,gate_channels,4,4,0,bias=False),
                        nn.BatchNorm2d(gate_channels))
        elif spatial_size==14:
            self.conv2=nn.Sequential(
                        nn.ConvTranspose2d(group,gate_channels,4,4,1,bias=False),
                        nn.BatchNorm2d(gate_channels))
        elif spatial_size==7:
            self.conv2=nn.Sequential(
                        nn.ConvTranspose2d(group,gate_channels,4,4,1,1,bias=False),
                        nn.BatchNorm2d(gate_channels))
        # self.conv3=nn.Conv2d(gate_channels, gate_channels, kernel_size=1, bias=False)
    def forward(self, x):
        _,c,h,w=x.size()
        c2,h2,w2=self.group,int(round(h/4)),int(round(w/4))
        y=x.clone()
        y=y.unsqueeze(0)
        y=F.adaptive_avg_pool3d(y,(c2,h2,w2))
        y=y.squeeze(0)
        y=self.conv2(y)
        # print(y.size())
        y=F.sigmoid(y)

        return y*x

class CBAM_new_group32_ave_max(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, group=32, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM_new_group32_ave_max, self).__init__()
        # self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        # self.no_spatial=no_spatial
        # if not no_spatial:
        #     self.SpatialGate = SpatialGate()

        # self.avg_pool = nn.AdaptiveAvgPool3d(self.ave_size)
        # group=32
        print(group)
        self.group=group
        # self.conv1=nn.Conv2d(gate_channels, gate_channels, kernel_size=1, bias=False)
        self.conv2_1=nn.Conv2d(group, group, kernel_size=1, bias=False)
        self.conv2_2=nn.Conv2d(group, group, kernel_size=1, bias=False)
        self.conv=nn.Conv3d(2,1,kernel_size=(1,1,1),bias=False)
        # self.conv3=nn.Conv2d(gate_channels, gate_channels, kernel_size=1, bias=False)
    def forward(self, x):
        _,c,h,w=x.size()
        # c2,h2,w2=int(c/4),int(h/4),int(w/4)
        c2,h2,w2=self.group,int(h/4),int(w/4)
        # y=self.conv1(x)
        y=x.clone()
        y=y.unsqueeze(0)
        y1=F.adaptive_avg_pool3d(y,(c2,h2,w2))
        y1=y1.squeeze(0)
        y1=self.conv2_1(y1)

        # y=y.unsqueeze(0)
        y2=F.adaptive_max_pool3d(y,(c2,h2,w2))
        y2=y2.squeeze(0)
        y2=self.conv2_2(y2)

        y1=y1.unsqueeze(1)
        y2=y2.unsqueeze(1)
        y3=torch.cat((y1,y2),dim=1)
        y3=self.conv(y3)
        y3=y3.squeeze(1)

        y3=y3.unsqueeze(0)
        y3=F.upsample(y3,size=(c,h,w))
        y3=y3.squeeze(0)
        # y=self.conv3(y)
        y3=F.sigmoid(y3)
        # print(y3.size())

        return y3*x


class CBAM_new_group32_ave_max2(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, group=32, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM_new_group32_ave_max2, self).__init__()
        # self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        # self.no_spatial=no_spatial
        # if not no_spatial:
        #     self.SpatialGate = SpatialGate()

        # self.avg_pool = nn.AdaptiveAvgPool3d(self.ave_size)
        # group=32
        print(group)
        self.group=group
        # self.conv1=nn.Conv2d(gate_channels, gate_channels, kernel_size=1, bias=False)
        self.conv2_1=nn.Sequential(nn.Conv2d(group, group, kernel_size=1, bias=False),nn.BatchNorm2d(group),nn.ReLU(inplace=True))
        self.conv2_2=nn.Sequential(nn.Conv2d(group, group, kernel_size=1, bias=False),nn.BatchNorm2d(group),nn.ReLU(inplace=True))
        # self.conv=nn.Conv3d(2,1,kernel_size=(1,1,1),bias=False)
        # self.conv3=nn.Conv2d(gate_channels, gate_channels, kernel_size=1, bias=False)
    def forward(self, x):
        _,c,h,w=x.size()
        # c2,h2,w2=int(c/4),int(h/4),int(w/4)
        c2,h2,w2=self.group,int(h/4),int(w/4)
        # y=self.conv1(x)
        y=x.clone()
        y=y.unsqueeze(0)
        y1=F.adaptive_avg_pool3d(y,(c2,h2,w2))
        y1=y1.squeeze(0)
        y1=self.conv2_1(y1)

        # y=y.unsqueeze(0)
        y2=F.adaptive_max_pool3d(y,(c2,h2,w2))
        y2=y2.squeeze(0)
        y2=self.conv2_2(y2)

        # y1=y1.unsqueeze(1)
        # y2=y2.unsqueeze(1)
        # y3=torch.cat((y1,y2),dim=1)
        # y3=self.conv(y3)
        # y3=y3.squeeze(1)
        y3=y1+y2

        y3=y3.unsqueeze(0)
        y3=F.upsample(y3,size=(c,h,w))
        y3=y3.squeeze(0)
        # y=self.conv3(y)
        y3=F.sigmoid(y3)
        # print(y3.size())

        return y3*x



class CBAM_new_group32_groupconv1(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, group=32, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM_new_group32_groupconv1, self).__init__()
        # self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        # self.no_spatial=no_spatial
        # if not no_spatial:
        #     self.SpatialGate = SpatialGate()

        # self.avg_pool = nn.AdaptiveAvgPool3d(self.ave_size)
        # group=32
        print(group)
        self.group=group
        self.conv1=nn.Conv2d(gate_channels, gate_channels, kernel_size=1,  stride=1, padding=0, groups=group, bias=False)
        self.conv2=nn.Conv2d(group, group, kernel_size=1, bias=False)
        # self.conv3=nn.Conv2d(gate_channels, gate_channels, kernel_size=1, bias=False)
    def forward(self, x):
        _,c,h,w=x.size()
        x=self.conv1(x)
        # c2,h2,w2=int(c/4),int(h/4),int(w/4)
        c2,h2,w2=self.group,int(h/4),int(w/4)
        # y=self.conv1(x)
        y=x.clone()
        y=y.unsqueeze(0)
        y=F.adaptive_avg_pool3d(y,(c2,h2,w2))
        y=y.squeeze(0)
        y=self.conv2(y)
        y=y.unsqueeze(0)
        y=F.upsample(y,size=(c,h,w))
        y=y.squeeze(0)
        # y=self.conv3(y)
        y=F.sigmoid(y)

        return y*x

class CBAM_new_group32_groupconv2(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, group=32, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM_new_group32_groupconv2, self).__init__()
        # self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        # self.no_spatial=no_spatial
        # if not no_spatial:
        #     self.SpatialGate = SpatialGate()

        # self.avg_pool = nn.AdaptiveAvgPool3d(self.ave_size)
        # group=32
        print(group)
        self.group=group
        self.conv1=nn.Conv2d(gate_channels, gate_channels, kernel_size=1,  stride=1, padding=0, groups=group, bias=False)
        self.conv2=nn.Conv2d(group, group, kernel_size=1, bias=False)
        # self.conv3=nn.Conv2d(gate_channels, gate_channels, kernel_size=1, bias=False)
    def forward(self, x):
        _,c,h,w=x.size()
        y=x.clone()
        y=self.conv1(y)
        # c2,h2,w2=int(c/4),int(h/4),int(w/4)
        c2,h2,w2=self.group,int(h/4),int(w/4)
        # y=self.conv1(x)
        
        y=y.unsqueeze(0)
        y=F.adaptive_avg_pool3d(y,(c2,h2,w2))
        y=y.squeeze(0)
        y=self.conv2(y)
        y=y.unsqueeze(0)
        y=F.upsample(y,size=(c,h,w))
        y=y.squeeze(0)
        # y=self.conv3(y)
        y=F.sigmoid(y)

        return y*x

class CBAM_new_group_multicardi(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, group=32, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM_new_group_multicardi, self).__init__()
        # self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        # self.no_spatial=no_spatial
        # if not no_spatial:
        #     self.SpatialGate = SpatialGate()

        # self.avg_pool = nn.AdaptiveAvgPool3d(self.ave_size)
        # group=32
        # print(group)
        self.group=64
        # self.conv1=nn.Conv2d(gate_channels, gate_channels, kernel_size=1, bias=False)
        self.conv2=nn.Conv2d(group, group, kernel_size=1, bias=False)
        # self.conv3=nn.Conv2d(gate_channels, gate_channels, kernel_size=1, bias=False)
    def forward(self, x):
        _,c,h,w=x.size()
        # c2,h2,w2=int(c/4),int(h/4),int(w/4)
        c2,h2,w2=self.group,int(h/4),int(w/4)
        # y=self.conv1(x)
        y=x
        y=y.unsqueeze(0)
        y=F.adaptive_avg_pool3d(y,(c2,h2,w2))
        y=y.squeeze(0)
        y=self.conv2(y)
        y=y.unsqueeze(0)
        y=F.upsample(y,size=(c,h,w))
        y=y.squeeze(0)
        # y=self.conv3(y)
        y=F.sigmoid(y)

        return y*x


class CBAM_new_group32_multi3(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, group=32, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM_new_group32_multi3, self).__init__()
        print(group)
        self.group=group
        self.conv2=nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.conv3=nn.Conv2d(48, 48, kernel_size=1, bias=False)
        self.conv4=nn.Conv2d(32, 32, kernel_size=1, bias=False)
        # self.conv=nn.Conv2d(3, 1, kernel_size=1, bias=False)
        # self.conv3=nn.Conv2d(gate_channels, gate_channels, kernel_size=1, bias=False)
    def forward(self, x):
        _,c,h,w=x.size()
        c2,h2,w2=64,int(round(h/4)),int(round(w/4))
        c3,h3,w3=48,int(round(h/4)),int(round(w/4))
        c4,h4,w4=32,int(round(h/4)),int(round(w/4))
        y2=x.clone()
        y2=y2.unsqueeze(0)
        y2=F.adaptive_avg_pool3d(y2,(c2,h2,w2))
        y2=y2.squeeze(0)
        y2=self.conv2(y2)
        y2=y2.unsqueeze(0)
        y2=F.upsample(y2,size=(c,h,w))
        y2=y2.squeeze(0)
        y2=F.sigmoid(y2)

        y3=y2*x+x
        y3=y3.unsqueeze(0)
        y3=F.adaptive_avg_pool3d(y3,(c3,h3,w4))
        y3=y3.squeeze(0)
        y3=self.conv3(y3)
        y3=y3.unsqueeze(0)
        y3=F.upsample(y3,size=(c,h,w))
        y3=y3.squeeze(0)
        # y=self.conv3(y)
        y3=F.sigmoid(y3)

        y4=y3*x+x
        y4=y4.unsqueeze(0)
        y4=F.adaptive_avg_pool3d(y4,(c4,h4,w4))
        y4=y4.squeeze(0)
        y4=self.conv4(y4)
        y4=y4.unsqueeze(0)
        y4=F.upsample(y4,size=(c,h,w))
        y4=y4.squeeze(0)
        # y=self.conv3(y)
        y4=F.sigmoid(y4)

        return y4*x

class CBAM_new_oneconv2(nn.Module):
    ## one conv
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM_new_oneconv2, self).__init__()
        # self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        # self.no_spatial=no_spatial
        # if not no_spatial:
        #     self.SpatialGate = SpatialGate()

        # self.avg_pool = nn.AdaptiveAvgPool3d(self.ave_size)
        # self.conv1=nn.Conv2d(gate_channels, gate_channels, kernel_size=1, bias=False)
        self.conv2=nn.Conv2d(int(gate_channels/4), int(gate_channels/4), kernel_size=1, bias=False)
        # self.conv3=nn.Conv2d(gate_channels, gate_channels, kernel_size=1, bias=False)
    def forward(self, x):
        _,c,h,w=x.size()
        c2,h2,w2=int(c/4),int(h/4),int(w/4)
        y=x
        y=y.unsqueeze(0)
        y1=F.adaptive_avg_pool3d(y,(c2,h2,w2))
        y1=y1.squeeze(0)
        y1=self.conv2(y1)
        y1=y1.unsqueeze(0)
        y1=F.upsample(y1,size=(c,h,w))
        y1=y1.squeeze(0)
        # y=self.conv3(y)
        y1=F.sigmoid(y1)

        y2=F.adaptive_max_pool3d(y,(c2,h2,w2))
        y2=y2.squeeze(0)
        y2=self.conv2(y2)
        y2=y2.unsqueeze(0)
        y2=F.upsample(y2,size=(c,h,w))
        y2=y2.squeeze(0)
        # y=self.conv3(y)
        y2=F.sigmoid(y2)

        return (y1+y2)*x

class CBAM_new_oneconv3(nn.Module):
    ## one conv
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM_new_oneconv2, self).__init__()
        # self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        # self.no_spatial=no_spatial
        # if not no_spatial:
        #     self.SpatialGate = SpatialGate()

        # self.avg_pool = nn.AdaptiveAvgPool3d(self.ave_size)
        # self.conv1=nn.Conv2d(gate_channels, gate_channels, kernel_size=1, bias=False)
        self.conv2=nn.Conv2d(int(gate_channels/2), int(gate_channels/4), kernel_size=1, bias=False)
        # self.conv3=nn.Conv2d(gate_channels, gate_channels, kernel_size=1, bias=False)
    def forward(self, x):
        _,c,h,w=x.size()
        c2,h2,w2=int(c/4),int(h/4),int(w/4)
        y=x
        y=y.unsqueeze(0)
        y1=F.adaptive_avg_pool3d(y,(c2,h2,w2))
        y1=y1.squeeze(0)
        y2=F.adaptive_max_pool3d(y,(c2,h2,w2))
        y2=y2.squeeze(0)
        y3=torch.cat((y1,y2),1)
        print(y3.size())

        y3=self.conv2(y3)
        y3=y3.unsqueeze(0)
        y3=F.upsample(y3,size=(c,h,w))
        y3=y3.squeeze(0)
        # y=self.conv3(y)
        y3=F.sigmoid(y3)

        # y2=F.adaptive_max_pool3d(y,(c2,h2,w2))
        # y2=y2.squeeze(0)
        # y2=self.conv2(y2)
        # y2=y2.unsqueeze(0)
        # y2=F.upsample(y2,size=(c,h,w))
        # y2=y2.squeeze(0)
        # # y=self.conv3(y)
        # y2=F.sigmoid(y2)

        return y3*x

class CBAM_new_noconv(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM_new_noconv, self).__init__()
        # self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        # self.no_spatial=no_spatial
        # if not no_spatial:
        #     self.SpatialGate = SpatialGate()

        # self.avg_pool = nn.AdaptiveAvgPool3d(self.ave_size)
        # self.conv1=nn.Conv2d(gate_channels, gate_channels, kernel_size=1, bias=False)
        # self.conv2=nn.Conv2d(int(gate_channels/4), int(gate_channels/4), kernel_size=1, bias=False)
        # self.conv3=nn.Conv2d(gate_channels, gate_channels, kernel_size=1, bias=False)
    def forward(self, x):
        _,c,h,w=x.size()
        c2,h2,w2=int(c/4),int(h/4),int(w/4)
        # y=self.conv1(x)
        y=x.unsqueeze(0)
        y=F.adaptive_avg_pool3d(y,(c2,h2,w2))
        y=y.squeeze(0)
        # y=self.conv2(y)
        y=y.unsqueeze(0)
        y=F.upsample(y,size=(c,h,w))
        y=y.squeeze(0)
        # y=self.conv3(y)
        y=F.sigmoid(y)

        return y*x


from .peak_stimulation import PeakStimulation_ori4_6_3
import random
class CBAM_new_group32_hs2(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM_new_group32_hs2, self).__init__()
        group=32
        self.group=group
        print(self.group)
        self.conv2=nn.Conv2d(group, group, kernel_size=1, bias=False)
        self.prm_ours=PeakStimulation_ori4_6_3(0.2)

    def forward(self, x):
        _,c,h,w=x.size()
        c2,h2,w2=self.group,int(h/4),int(w/4)
        # print("x1:",x[0,0,0,:])
        y=x
        y=y.unsqueeze(0)
        y=F.adaptive_avg_pool3d(y,(c2,h2,w2))
        y=y.squeeze(0)
        y=self.conv2(y)
        y=y.unsqueeze(0)
        y=F.interpolate(y,size=(c,h,w))
        y=y.squeeze(0)
        #second_time=time.time()
        #print("time1",second_time-start_time)
        # y=self.conv3(y)
        # print(random.random())
        if random.random()<0.5:
            # print("here")
            y=self.prm_ours(y)
        #print("time2",time.time()-second_time)
        y=F.sigmoid(y)
        #exit()
        # print("y:",y[0,0,0,:])
        # print("x2:",x[0,0,0,:])
        # exit()

        return y*x