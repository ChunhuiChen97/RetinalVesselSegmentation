import torch
import torch.nn as nn
import torch.nn.functional as F

class CBR(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dilation_rate:str, stride = 1, drop=False):
        super(CBR,self).__init__()
        self.drop = drop
        self.conv0 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size,
                                padding=(dilation_rate[0]*(kernel_size-1) + 1)//2, dilation=dilation_rate[0], stride=stride)
        self.bn0 = nn.BatchNorm2d(num_features = out_channel)
        self.conv1 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=kernel_size, 
                                padding=(dilation_rate[1]*(kernel_size-1) + 1)//2, dilation=dilation_rate[1], stride=stride)
        self.bn1 = nn.BatchNorm2d(num_features = out_channel)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=kernel_size, 
                                padding=(dilation_rate[2]*(kernel_size-1) + 1)//2, dilation=dilation_rate[2], stride=stride)
        self.bn2 = nn.BatchNorm2d(num_features = out_channel)
        self.drop = nn.Dropout2d(0.25)

    def forward(self,inputs):
        x = self.conv0(inputs)
        x = F.relu(self.bn0(x))
        if self.drop:
            x = self.drop(x)
        res = x
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        if self.drop:
            x = self.drop(x)
        x = self.conv2(x)
        x = x + res
        x = F.relu(self.bn2(x))
        
        return x
class BottleNeck(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride =1):
        super(BottleNeck, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, padding=(kernel_size-1)//2, stride=stride)
        self.bn0 = nn.BatchNorm2d(num_features=out_channel)
        self.conv1 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=kernel_size, padding=(kernel_size-1)//2, stride=stride)
        self.bn1 = nn.BatchNorm2d(num_features=out_channel)
    def forward(self,inputs):
        x = self.conv0(inputs)
        x = F.relu(self.bn0(x))
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        return x
class Bridge(nn.Module):
    def __init__(self, in_channel, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channels, kernel_size=1,stride=1)
        self.bn = nn.BatchNorm2d(num_features=1)
    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = nn.Sigmoid()(x)
        return x

class First_Unet(nn.Module):
    def __init__(self, out_features,kernel_size, dilation_rate):
        super(First_Unet, self).__init__()
        self.depth0_0 = CBR(in_channel=1, out_channel=out_features[0], 
                            kernel_size= kernel_size, dilation_rate= dilation_rate, drop=False)
        self.depth0_1 = CBR(in_channel=out_features[0], out_channel=out_features[1], 
                            kernel_size= kernel_size, dilation_rate= dilation_rate, drop=False)
        self.depth0_2 = CBR(in_channel=out_features[1], out_channel=out_features[2], 
                            kernel_size= kernel_size, dilation_rate= dilation_rate, drop=False)
        self.depth0_3 = CBR(in_channel=out_features[2], out_channel=out_features[3], 
                            kernel_size= kernel_size, dilation_rate= dilation_rate, drop=False)

        self.bottle1 = BottleNeck(in_channel= out_features[3], out_channel= 2 * out_features[3], kernel_size=kernel_size)

        self.depth1_3 = CBR(in_channel=2*out_features[3] + out_features[3], 
                            out_channel=out_features[3], kernel_size= kernel_size, dilation_rate= dilation_rate, drop=False)
        self.depth1_2 = CBR(in_channel=out_features[3] + out_features[2], 
                            out_channel=out_features[2], kernel_size= kernel_size, dilation_rate= dilation_rate, drop=False)
        self.depth1_1 = CBR(in_channel=out_features[2] + out_features[1], 
                            out_channel=out_features[1], kernel_size= kernel_size, dilation_rate= dilation_rate, drop=False)
        self.depth1_0 = CBR(in_channel=out_features[1] + out_features[0], 
                            out_channel=out_features[0], kernel_size= kernel_size, dilation_rate= dilation_rate, drop=False)
        self.output = Bridge(in_channel=out_features[0], out_channels=1)
        
        self.pooling = nn.MaxPool2d(kernel_size=2,stride=2)
        self.upsampling = nn.Upsample(scale_factor=2,mode="bilinear")

    def forward(self, inputs):

        x0_0 = self.pooling(self.depth0_0(inputs))
        x0_1 = self.pooling(self.depth0_1(x0_0))
        x0_2 = self.pooling(self.depth0_2(x0_1))
        x0_3 = self.pooling(self.depth0_3(x0_2))
        x_neck = self.bottle1(x0_3)
        x1_3 = self.upsampling(self.depth1_3(torch.cat((x_neck,x0_3), dim=1)))
        x1_2 = self.upsampling(self.depth1_2(torch.cat((x1_3,x0_2),dim=1)))
        x1_1 = self.upsampling(self.depth1_1(torch.cat((x1_2,x0_1), dim=1)))
        x1_0 = self.upsampling(self.depth1_0(torch.cat((x1_1,x0_0), dim=1)))
        out_1 = self.output(x1_0)
        return out_1, x1_0, x1_1, x1_2, x1_3
class Second_Unet(nn.Module):
    def __init__(self, out_features,kernel_size, dilation_rate):
        super(Second_Unet,self).__init__()

        self.depth0_0 = CBR(in_channel=1+1, out_channel=out_features[0], kernel_size= kernel_size, dilation_rate= dilation_rate, drop=True)
        self.depth0_1 = CBR(in_channel=out_features[1]+out_features[0], 
                            out_channel=out_features[1], kernel_size= kernel_size, dilation_rate= dilation_rate, drop=True)
        self.depth0_2 = CBR(in_channel=out_features[2]+out_features[1], 
                            out_channel=out_features[2], kernel_size= kernel_size, dilation_rate= dilation_rate, drop=True)
        self.depth0_3 = CBR(in_channel=out_features[3]+out_features[2], 
                            out_channel=out_features[3], kernel_size= kernel_size, dilation_rate= dilation_rate, drop=True)

        self.bottle1 = BottleNeck(in_channel= out_features[3], out_channel= 2 * out_features[3], kernel_size=kernel_size)

        self.depth1_3 = CBR(in_channel=2*out_features[3] + out_features[3], 
                            out_channel=out_features[3], kernel_size= kernel_size, dilation_rate= dilation_rate, drop=True)
        self.depth1_2 = CBR(in_channel=out_features[3] + out_features[2], 
                            out_channel=out_features[2], kernel_size= kernel_size, dilation_rate= dilation_rate, drop=True)
        self.depth1_1 = CBR(in_channel=out_features[2] + out_features[1], 
                            out_channel=out_features[1], kernel_size= kernel_size, dilation_rate= dilation_rate, drop=True)
        self.depth1_0 = CBR(in_channel=out_features[1] + out_features[0], 
                            out_channel=out_features[0], kernel_size= kernel_size, dilation_rate= dilation_rate, drop=True)
        self.output = Bridge(in_channel=out_features[0], out_channels=1)

        self.pooling = nn.MaxPool2d(kernel_size=2,stride=2)
        self.upsampling = nn.Upsample(scale_factor=2,mode="bilinear")
        
    def forward(self, inputs, x, x1_0, x1_1, x1_2, x1_3):

        x2_0 = self.pooling(self.depth0_0(torch.cat((inputs, x), dim = 1)))
        x2_1 = self.pooling(self.depth0_1(torch.cat((x2_0, x1_1), dim = 1)))
        x2_2 = self.pooling(self.depth0_2(torch.cat((x2_1, x1_2), dim = 1)))
        x2_3 = self.pooling(self.depth0_3(torch.cat((x2_2, x1_3), dim = 1)))
        x_neck = self.bottle1(x2_3)
        x3_3 = self.upsampling(self.depth1_3(torch.cat((x_neck,x2_3), dim=1)))
        x3_2 = self.upsampling(self.depth1_2(torch.cat((x3_3,x2_2),dim=1)))
        x3_1 = self.upsampling(self.depth1_1(torch.cat((x3_2,x2_1), dim=1)))
        x3_0 = self.upsampling(self.depth1_0(torch.cat((x3_1,x2_0), dim=1)))
        out_2 = self.output(x3_0)

        return out_2

class get_model(nn.Module):
    def __init__(self, out_features, kernel_size, dilation_rate, mode):
        super(get_model,self).__init__()
        self.mode = mode
        self.first_network = First_Unet(out_features=out_features, kernel_size = kernel_size, dilation_rate = dilation_rate)
        if mode == "double":
            self.second_network = Second_Unet(out_features=out_features, kernel_size = kernel_size, dilation_rate = dilation_rate)
            self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1,stride=1)
            self.bn = nn.BatchNorm2d(num_features=1)
            self.sig = nn.Sigmoid()
    def forward(self,inputs):
        
        if self.mode == "single":
            out_1, x1_0, x1_1, x1_2, x1_3 = self.first_network(inputs)
            return out_1
        elif self.mode == "double":
            out_1, x1_0, x1_1, x1_2, x1_3 = self.first_network(inputs)
            out_2 = self.second_network(inputs, out_1, x1_0, x1_1, x1_2, x1_3)
            out = self.sig(self.bn(self.conv(torch.cat((out_1, out_2),dim=1))))
            return out
        else:
            raise ValueError("wrong mode!!")

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
    def forward(self, pred, target):
        # weight = torch.where(target>0.5,2,1)
        weight = target+1
        loss = nn.BCELoss(weight=weight)(pred, target)

        return loss


if __name__=="__main__":
    sample = torch.rand(8,1,48,48)
    model = get_model(out_features=[16,32,64,128], kernel_size=3,dilation_rate=[1,2,3],mode="xxx")
    num = 0
    for para in model.parameters():
        num += para.numel()
    print(f"number of parameters:{num}")
    out = model(sample)
    print(out.shape)

