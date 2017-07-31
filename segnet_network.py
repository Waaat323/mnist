# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 19:56:50 2017

@author: kawalab
"""
import chainer
import chainer.functions as F
import chainer.links as L


class SegNet(chainer.Chain):
    def __init__(self, in_channel=3, out_channel=10,  c1=64, c2=64, c3=64,
                 c4=64, c5=64, filter_size=7):
        super(SegNet, self).__init__(
                
            conv1_1=L.Convolution2D(in_channel, c1, ksize=filter_size),
            conv1_1_bn = L.BatchNormalization(c1, initial_beta=0.001),
            conv1_2=L.Convolution2D(c1, c1, filter_size),
            conv1_2_bn = L.BatchNormalization(c1, initial_beta=0.001),
                   
            conv2_1=L.Convolution2D(c1, c2, ksize=filter_size),
            conv2_1_bn = L.BatchNormalization(c2, initial_beta=0.001),
            conv2_2=L.Convolution2D(c2, c2, ksize=filter_size),            
            conv2_2_bn = L.BatchNormalization(c2, initial_beta=0.001),
            
            conv3_1=L.Convolution2D(c2, c3, ksize=filter_size),
            conv3_1_bn = L.BatchNormalization(c3, initial_beta=0.001),
            conv3_2=L.Convolution2D(c3, c3, ksize=filter_size),            
            conv3_2_bn = L.BatchNormalization(c3, initial_beta=0.001),  
            conv3_3=L.Convolution2D(c3, c3, ksize=filter_size),            
            conv3_3_bn = L.BatchNormalization(c3, initial_beta=0.001),  
            
            conv4_1=L.Convolution2D(c3, c4, ksize=filter_size),
            conv4_1_bn = L.BatchNormalization(c4, initial_beta=0.001),
            conv4_2=L.Convolution2D(c4, c4, ksize=filter_size),            
            conv4_2_bn = L.BatchNormalization(c4, initial_beta=0.001),             
            conv4_3=L.Convolution2D(c4, c4, ksize=filter_size),            
            conv4_3_bn = L.BatchNormalization(c4, initial_beta=0.001),   
    
            conv5_1=L.Convolution2D(c4, c5, ksize=filter_size),
            conv5_1_bn = L.BatchNormalization(c5, initial_beta=0.001),
            conv5_2=L.Convolution2D(c5, c5, ksize=filter_size),            
            conv5_2_bn = L.BatchNormalization(c5, initial_beta=0.001),       
            conv5_3=L.Convolution2D(c5, c5, ksize=filter_size),            
            conv5_3_bn = L.BatchNormalization(c5, initial_beta=0.001),   
            
            dconv6_1=L.Deconvolution2D(c5, c4, ksize=filter_size, pad=3),
            conv6_1_bn = L.BatchNormalization(c4, initial_beta=0.001),
            dconv6_2=L.Deconvolution2D(c4, c4, ksize=filter_size, pad=3),
            conv6_2_bn = L.BatchNormalization(c4, initial_beta=0.001),
            dconv6_3=L.Deconvolution2D(c5, c4, ksize=filter_size, pad=3),
            conv6_3_bn = L.BatchNormalization(c4, initial_beta=0.001),

            dconv7_1=L.Deconvolution2D(c4, c3, ksize=filter_size, pad=3),
            conv7_1_bn = L.BatchNormalization(c3, initial_beta=0.001),
            dconv7_2=L.Deconvolution2D(c3, c3, ksize=filter_size, pad=3),
            conv7_2_bn = L.BatchNormalization(c3, initial_beta=0.001),
            dconv7_3=L.Deconvolution2D(c3, c3, ksize=filter_size, pad=3),
            conv7_3_bn = L.BatchNormalization(c3, initial_beta=0.001),

            dconv8_1=L.Deconvolution2D(c3, c2, ksize=filter_size, pad=3),
            conv8_1_bn = L.BatchNormalization(c2, initial_beta=0.001),
            dconv8_2=L.Deconvolution2D(c2, c2, ksize=filter_size, pad=3),
            conv8_2_bn = L.BatchNormalization(c2, initial_beta=0.001),
            dconv8_3=L.Deconvolution2D(c2, c2, ksize=filter_size, pad=3),
            conv8_3_bn = L.BatchNormalization(c2, initial_beta=0.001),
            
            dconv9_1=L.Deconvolution2D(c2, c1, ksize=filter_size, pad=3),
            conv9_1_bn = L.BatchNormalization(c1, initial_beta=0.001),
            dconv9_2=L.Deconvolution2D(c1, c1, ksize=filter_size, pad=3),
            conv9_2_bn = L.BatchNormalization(c1, initial_beta=0.001),
            
            dconv10_1=L.Deconvolution2D(c2, c1, ksize=filter_size, pad=3),
            conv10_1_bn = L.BatchNormalization(c1, initial_beta=0.001),
            dconv10_2=L.Deconvolution2D(c1, c1, ksize=filter_size, pad=3),
            conv10_2_bn = L.BatchNormalization(c1, initial_beta=0.001),


            
            )

    def __call__(self, x):
 
        outsize1 = x.shape[-2:] #[-1]から
        h = F.relu(self.conv1_1_bn(self.conv1_1(x)))
        h = F.relu(self.conv1_2_bn(self.conv1_2(x)))
        h = F.max_pooling_2d(h, 2)
        
        outsize2 = x.shape[-2:]
        h = F.relu(self.conv2_1_bn(self.conv2_1(x)))
        h = F.relu(self.conv2_2_bn(self.conv2_2(x)))
        h = F.max_pooling_2d(h, 2)      
        
        outsize3 = x.shape[-2:]
        h = F.relu(self.conv3_1_bn(self.conv3_1(x)))
        h = F.relu(self.conv3_2_bn(self.conv3_2(x)))
        h = F.relu(self.conv3_3_bn(self.conv3_3(x)))
        h = F.max_pooling_2d(h, 2)       
        
        outsize4 = x.shape[-2:]
        h = F.relu(self.conv4_1_bn(self.conv4_1(x)))
        h = F.relu(self.conv4_2_bn(self.conv4_2(x)))
        h = F.relu(self.conv4_3_bn(self.conv4_3(x)))
        h = F.max_pooling_2d(h, 2)
        
        outsize5 = x.shape[-2:]
        h = F.relu(self.conv5_1_bn(self.conv5_1(x)))
        h = F.relu(self.conv5_2_bn(self.conv5_2(x)))
        h = F.relu(self.conv5_3_bn(self.conv5_3(x)))
        h = F.max_pooling_2d(h, 2)
        
        h = F.unpooling_2d(h, 2, outsize=outsize5)
        h = F.relu(self.conv6_1_bn(self.dconv6_1(x)))
        h = F.relu(self.conv6_2_bn(self.dconv6_2(x)))
        h = F.relu(self.conv6_3_bn(self.dconv6_3(x)))
        
        h = F.unpooling_2d(h, 2, outsize=outsize4)
        h = F.relu(self.conv7_1_bn(self.dconv7_1(x)))
        h = F.relu(self.conv7_2_bn(self.dconv7_2(x)))
        h = F.relu(self.conv7_3_bn(self.dconv7_3(x)))

        h = F.unpooling_2d(h, 2, outsize=outsize3)
        h = F.relu(self.conv8_1_bn(self.dconv8_1(x)))
        h = F.relu(self.conv8_2_bn(self.dconv8_2(x)))
        h = F.relu(self.conv8_3_bn(self.dconv8_3(x)))

        h = F.unpooling_2d(h, 2, outsize=outsize2)
        h = F.relu(self.conv9_1_bn(self.dconv9_1(x)))
        h = F.relu(self.conv9_2_bn(self.dconv9_2(x)))

        h = F.unpooling_2d(h, 2, outsize=outsize1)
        h = F.relu(self.conv10_1_bn(self.dconv10_1(x)))
        y = F.relu(self.conv10_2_bn(self.dconv10_2(x)))     
        
        return y


if __name__ == '__main__':
    SegNet()