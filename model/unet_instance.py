import sys
#sys.modules.pop("model.model_components")
from model.model_components import *
from model.unet_super import UnetSuper
from model.utils import weights_init
import sys


class Unet(UnetSuper):
    """Unet

    Basic Unet which is used for medical image segmentation and classification
    original paper: https://arxiv.org/pdf/1505.04597
    """
    def __init__(self, hparams, input_channels, min_filter=16, is_deconv=False, is_batchnorm=True, on_gpu=False, **kwargs):
        super().__init__(hparams=hparams, **kwargs)
        self.in_channels = input_channels
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.input = input_channels
        filters = [32, 64, 128, 256]
        self.conv1 = UnetConv(self.in_channels, filters[0], is_batchnorm, gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.conv2 = UnetConv(filters[0], filters[1], is_batchnorm, gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.conv3 = UnetConv(filters[1], filters[2], is_batchnorm, gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.center = UnetConv(filters[2], filters[3], is_batchnorm, gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        # upsampling
        self.up_concat3 = UnetUp(filters[3], filters[2], gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.up_concat2 = UnetUp(filters[2], filters[1], gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.up_concat1 = UnetUp(filters[1], filters[0], gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], kwargs["num_classes"], 1)
        if on_gpu:
            self.conv1.cuda()
            self.conv2.cuda()
            self.conv3.cuda()
            self.center.cuda()
            self.up_concat3.cuda()
            self.up_concat2.cuda()
            self.up_concat1.cuda()
            self.final.cuda()
        self.apply(weights_init)

    def forward(self, inputs):
        maxpool = nn.MaxPool2d(kernel_size=2)
        conv1 = self.conv1(inputs)  
        maxpool1 = maxpool(conv1)  

        conv2 = self.conv2(maxpool1)  
        maxpool2 = maxpool(conv2)  

        conv3 = self.conv3(maxpool2)  
        maxpool3 = maxpool(conv3) 

        center = self.center(maxpool3)

        up3 = self.up_concat3(center, conv3)  
        up2 = self.up_concat2(up3, conv2) 
        up1 = self.up_concat1(up2, conv1)  

        final = self.final(up1)
        finalize = nn.functional.softmax(final, dim=1)
        return finalize


    def print(self, args: torch.Tensor) -> None:
        print(args)


class UneXt(UnetSuper):
    """UneXt

    U-Net architecture with alterations inspired from swin transformer derived from
    http://arxiv.org/abs/2201.03545 (describes a ResNet block inspired by the swin stransformer)
    We have adapted the idea to fit U-net architecture
    """
    def __init__(self, hparams, input_channels, min_filter=16, on_gpu=False, **kwargs):
        super().__init__(hparams=hparams, **kwargs)
        self.in_channels = input_channels
        self.input = input_channels
        filters = [32, 64, 128, 256]

        # encoder
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, filters[0], kernel_size=3, stride=1, padding=1))
        self.conv1 = UneXtConv(filters[0],gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.conv2 = UneXtConv(filters[1], gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.conv3 = nn.Sequential(
            UneXtConv(filters[2], gpus=on_gpu, dropout_val=kwargs["dropout_val"]),
            UneXtConv(filters[2], gpus=on_gpu, dropout_val=kwargs["dropout_val"]),
            UneXtConv(filters[2], gpus=on_gpu, dropout_val=kwargs["dropout_val"]))

        # downsampling
        self.down1 = UneXtDown(filters[0], filters[1], gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.down2 = UneXtDown(filters[1], filters[2], gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.down3 = UneXtDown(filters[2], filters[3], gpus=on_gpu, dropout_val=kwargs["dropout_val"])

        # inverted bottleneck
        self.center = UneXtConv(filters[3], gpus=on_gpu, dropout_val=kwargs["dropout_val"])

        # upsampling
        self.up_concat3 = UneXtUp(filters[3], filters[2], gpus=on_gpu, is_third=True, dropout_val=kwargs["dropout_val"])
        self.up_concat2 = UneXtUp(filters[2], filters[1], gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.up_concat1 = UneXtUp(filters[1], filters[0], gpus=on_gpu, dropout_val=kwargs["dropout_val"])

        # final conv (without any concat)
        self.final = nn.Sequential(
            nn.GroupNorm(1, filters[0]),
            nn.Conv2d(filters[0], kwargs["num_classes"], 1))

        if on_gpu:
            self.stem.cuda()
            self.conv1.cuda()
            self.conv2.cuda()
            self.conv3.cuda()
            self.down1.cuda()
            self.down2.cuda()
            self.down3.cuda()
            self.center.cuda()
            self.up_concat3.cuda()
            self.up_concat2.cuda()
            self.up_concat1.cuda()
            self.final.cuda()

        self.apply(weights_init)

    def forward(self, inputs):

        stem = self.stem(inputs) 
        conv1 = self.conv1(stem) 
        down1 = self.down1(conv1)  

        conv2 = self.conv2(down1)  
        down2 = self.down2(conv2)  

        conv3 = self.conv3(down2) 
        down3 = self.down3(conv3)  

        center = self.center(down3) 

        up3 = self.up_concat3(center, conv3)  
        up2 = self.up_concat2(up3, conv2)  
        up1 = self.up_concat1(up2, conv1)  

        final = self.final(up1)
        finalize = nn.functional.softmax(final, dim=1)

        return finalize


    def print(self, args: torch.Tensor) -> None:
        print(args)



#### ==== Context Unet ==== ####
class ContextUnet(UnetSuper):
    """Context Unet is a U-Net with added context modules and localization modules and a different way of generating
    the higher dimension feature maps. Additionally deep_supervision elements are present, however not meaningfully
    better

    """
    def __init__(self, hparams, input_channels, is_deconv=True, is_batchnorm=True, on_gpu=False,
                 deep_supervision=True, **kwargs):
        super().__init__(hparams=hparams, **kwargs)
        self.deep_supervision = deep_supervision
        self.in_channels = input_channels
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.input = input_channels
        filters = [16, 32, 64, 128, 256]
        self.conv1 = SimpleUnetConv(self.in_channels, filters[0], stride=1, gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.context1 = ContextModule(filters[0], filters[0],gpus=on_gpu)
        self.ttt2 = SimpleUnetConv(filters[0], filters[1], gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.context2 = ContextModule(filters[1], filters[1], gpus=on_gpu)
        self.ttt3 = SimpleUnetConv(filters[1], filters[2], gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.context3 = ContextModule(filters[2], filters[2],  gpus=on_gpu)
        self.ttt4 = SimpleUnetConv(filters[2], filters[3], gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.context4 = ContextModule(filters[3], filters[3], gpus=on_gpu)
        self.up_center = SimpleUnetUp(filters[3], filters[2], gpus=on_gpu)
        self.local1 = Localization(filters[3], filters[2], gpus=on_gpu)
        self.up1 = SimpleUnetUp(filters[2], filters[1], gpus=on_gpu)
        self.local2 = Localization(filters[2], filters[1], gpus=on_gpu)
        self.up2 = SimpleUnetUp(filters[1], filters[0], gpus=on_gpu)
        self.final = nn.Conv2d(filters[1], kwargs["num_classes"], 1)
        self.seg = SegmentationLayer(64, 32, 7, gpus=on_gpu)
        self.apply(weights_init)
        if on_gpu:
            self.conv1.cuda()
            self.context1.cuda()
            self.context2.cuda()
            self.context3.cuda()
            self.context4.cuda()
            self.ttt2.cuda()
            self.ttt3.cuda()
            self.ttt4.cuda()
            self.up_center.cuda()
            self.local1.cuda()
            self.local2.cuda()
            self.up1.cuda()
            self.up2.cuda()
            self.final.cuda()

    def forward(self, x):
        con1 = self.conv1(x) # 16*256*256
        son1 = self.context1(con1)
        plus1 = con1 + son1

        con2 = self.ttt2(plus1) # 32*128*128
        son2 = self.context2(con2)
        plus2 = con2 + son2

        con3 = self.ttt3(plus2) # 64*64*64
        son3 = self.context3(con3)
        plus3 = con3+son3

        con4 = self.ttt4(plus3) # 128*32*32
        son4 = self.context4(con4)
        plus4 = con4 + son4

        up_center = self.up_center(plus4) #64*64*64

        comb = torch.cat([plus3, up_center], dim=1) #128*64*64
        local1 = self.local1(comb) #64*64*64
        up1 = self.up1(local1) #32*128*128

        comb = torch.cat([plus2, up1], dim=1) #64*128*128
        local2 = self.local2(comb) #32*128*128
        up2 = self.up2(local2) #16*256*256

        comb = torch.cat([plus1, up2], dim=1) #32*256*256
        final = self.final(comb)  #7*256*256

        if self.deep_supervision:
            final = self.seg(local1, local2, final)

        return nn.functional.softmax(final, dim=1) #1*256*256


