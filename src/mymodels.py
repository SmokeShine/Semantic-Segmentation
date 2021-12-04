import torch
import torch.nn as nn

class Vanilla_SegNet(nn.Module):
    def __init__(self,num_output_channels):
        super(Vanilla_SegNet,self).__init__()
        self.num_output_channels=num_output_channels
        # https://production-media.paperswithcode.com/methods/segnet_Vorazx7.png

        # No fully connected layers - Only Convolutional Layers
        # 
        # 5 Stages of Encoding
        # Blue - Conv+Batch Normalization + Relu
        ## Conv - shape changes only after change in stage
        self.encoder_stage_1_conv1 = nn.Conv2d(3,64,kernel_size=3,padding=1)
        self.encoder_stage_1_conv2 = nn.Conv2d(64,64,kernel_size=3,padding=1)

        self.encoder_stage_2_conv1 = nn.Conv2d(64,128,kernel_size=3,padding=1)
        self.encoder_stage_2_conv2 = nn.Conv2d(128,128,kernel_size=3,padding=1)

        self.encoder_stage_3_conv1 = nn.Conv2d(128,256,kernel_size=3,padding=1)
        self.encoder_stage_3_conv2 = nn.Conv2d(256,256,kernel_size=3,padding=1)

        self.encoder_stage_4_conv1 = nn.Conv2d(256,512,kernel_size=3,padding=1)
        self.encoder_stage_4_conv2 = nn.Conv2d(512,512,kernel_size=3,padding=1)

        # No change in shape here
        self.encoder_stage_5_conv1 = nn.Conv2d(512,512,kernel_size=3,padding=1)
        self.encoder_stage_5_conv2 = nn.Conv2d(512,512,kernel_size=3,padding=1)

        ## Batch Normalization - one for each conv in a stage
        self.encoder_stage_1_batch_normalization1=nn.BatchNorm2d(64)
        self.encoder_stage_2_batch_normalization1=nn.BatchNorm2d(128)
        self.encoder_stage_3_batch_normalization1=nn.BatchNorm2d(256)
        self.encoder_stage_4_batch_normalization1=nn.BatchNorm2d(512)
        self.encoder_stage_5_batch_normalization1=nn.BatchNorm2d(512)

        self.encoder_stage_1_batch_normalization2=nn.BatchNorm2d(64)
        self.encoder_stage_2_batch_normalization2=nn.BatchNorm2d(128)
        self.encoder_stage_3_batch_normalization2=nn.BatchNorm2d(256)
        self.encoder_stage_4_batch_normalization2=nn.BatchNorm2d(512)
        self.encoder_stage_5_batch_normalization2=nn.BatchNorm2d(512)
        ## Relu
        self.relu = nn.ReLU()

        # Green - Pooling
        # Save indices for decoder
        self.encoder_max_pool = nn.MaxPool2d(2,2,return_indices=True)

        # 5 Stages of Decoding - Upsampling with transferred pool indices
        # The feature map is still sparse and need to perform convolution with a trainable filter bank 
        # to densify the output map

        # Red - Unsampling via max unpool - one across stages because of no parameters
        self.decoder_unpool = nn.MaxUnpool2d(2,stride=2)

        # Blue - 3 for stage 1-3, 2 for stage 4 and 5
        self.decoder_stage_1_conv1 = nn.Conv2d(512,512,kernel_size=3,padding=1)
        self.decoder_stage_1_conv2 = nn.Conv2d(512,512,kernel_size=3,padding=1)
        self.decoder_stage_1_conv3 = nn.Conv2d(512,512,kernel_size=3,padding=1)

        self.decoder_stage_2_conv1 = nn.Conv2d(512,512,kernel_size=3,padding=1)
        self.decoder_stage_2_conv2 = nn.Conv2d(512,512,kernel_size=3,padding=1)
        self.decoder_stage_2_conv3 = nn.Conv2d(512,256,kernel_size=3,padding=1)

        self.decoder_stage_3_conv1 = nn.Conv2d(256,256,kernel_size=3,padding=1)
        self.decoder_stage_3_conv2 = nn.Conv2d(256,256,kernel_size=3,padding=1)
        self.decoder_stage_3_conv3 = nn.Conv2d(256,128,kernel_size=3,padding=1)

        self.decoder_stage_4_conv1 = nn.Conv2d(128,128,kernel_size=3,padding=1)
        self.decoder_stage_4_conv2 = nn.Conv2d(128,64,kernel_size=3,padding=1)

        self.decoder_stage_5_conv1 = nn.Conv2d(64,64,kernel_size=3,padding=1)
        self.decoder_stage_5_conv2 = nn.Conv2d(64,self.num_output_channels,kernel_size=3,padding=1)

        ## Batch Normalization - one for each conv in a stage
        self.decoder_stage_1_batch_normalization1=nn.BatchNorm2d(512)
        self.decoder_stage_2_batch_normalization1=nn.BatchNorm2d(512)
        self.decoder_stage_3_batch_normalization1=nn.BatchNorm2d(256)
        self.decoder_stage_4_batch_normalization1=nn.BatchNorm2d(128)
        self.decoder_stage_5_batch_normalization1=nn.BatchNorm2d(64)

        self.decoder_stage_1_batch_normalization2=nn.BatchNorm2d(512)
        self.decoder_stage_2_batch_normalization2=nn.BatchNorm2d(512)
        self.decoder_stage_3_batch_normalization2=nn.BatchNorm2d(256)
        self.decoder_stage_4_batch_normalization2=nn.BatchNorm2d(64)
        self.decoder_stage_5_batch_normalization2=nn.BatchNorm2d(self.num_output_channels)

        self.decoder_stage_1_batch_normalization3=nn.BatchNorm2d(512)
        self.decoder_stage_2_batch_normalization3=nn.BatchNorm2d(256)
        self.decoder_stage_3_batch_normalization3=nn.BatchNorm2d(128)

        # Yellow
        # Final Decoder fed to softmax for pixel wise classification
        # Loss function will take care of this. Numerical Stability

    def forward(self,input):
        ###########################
        #    Start of Encoder     #
        ###########################
        ## Encoding

        # Stage 1
        #########

        # conv1
        encoder_stage_1=self.encoder_stage_1_conv1(input)
        encoder_stage_1=self.encoder_stage_1_batch_normalization1(encoder_stage_1)
        encoder_stage_1=self.relu(encoder_stage_1)
        # conv2
        encoder_stage_1=self.encoder_stage_1_conv2(encoder_stage_1)
        encoder_stage_1=self.encoder_stage_1_batch_normalization2(encoder_stage_1)
        encoder_stage_1=self.relu(encoder_stage_1)
        # max pool
        encoder_stage_1,encoder_stage_1_indices=self.encoder_max_pool(encoder_stage_1)

        # Stage 2
        #########

        # conv1
        encoder_stage_2=self.encoder_stage_2_conv1(encoder_stage_1)
        encoder_stage_2=self.encoder_stage_2_batch_normalization1(encoder_stage_2)
        encoder_stage_2=self.relu(encoder_stage_2)
        # conv2
        encoder_stage_2=self.encoder_stage_2_conv2(encoder_stage_2)
        encoder_stage_2=self.encoder_stage_2_batch_normalization2(encoder_stage_2)
        encoder_stage_2=self.relu(encoder_stage_2)
        # max pool
        encoder_stage_2,encoder_stage_2_indices=self.encoder_max_pool(encoder_stage_2)

        # Stage 3
        #########

        # conv1
        encoder_stage_3=self.encoder_stage_3_conv1(encoder_stage_2)
        encoder_stage_3=self.encoder_stage_3_batch_normalization1(encoder_stage_3)
        encoder_stage_3=self.relu(encoder_stage_3)
        # conv2
        encoder_stage_3=self.encoder_stage_3_conv2(encoder_stage_3)
        encoder_stage_3=self.encoder_stage_3_batch_normalization2(encoder_stage_3)
        encoder_stage_3=self.relu(encoder_stage_3)
        # max pool
        encoder_stage_3,encoder_stage_3_indices=self.encoder_max_pool(encoder_stage_3)

        # Stage 4
        #########

        # conv1
        encoder_stage_4=self.encoder_stage_4_conv1(encoder_stage_3)
        encoder_stage_4=self.encoder_stage_4_batch_normalization1(encoder_stage_4)
        encoder_stage_4=self.relu(encoder_stage_4)
        # conv2
        encoder_stage_4=self.encoder_stage_4_conv2(encoder_stage_4)
        encoder_stage_4=self.encoder_stage_4_batch_normalization2(encoder_stage_4)
        encoder_stage_4=self.relu(encoder_stage_4)
        # max pool
        encoder_stage_4,encoder_stage_4_indices=self.encoder_max_pool(encoder_stage_4)

        # Stage 5
        #########

        # conv1
        encoder_stage_5=self.encoder_stage_5_conv1(encoder_stage_4)
        encoder_stage_5=self.encoder_stage_5_batch_normalization1(encoder_stage_5)
        encoder_stage_5=self.relu(encoder_stage_5)
        # conv2
        encoder_stage_5=self.encoder_stage_5_conv2(encoder_stage_5)
        encoder_stage_5=self.encoder_stage_5_batch_normalization2(encoder_stage_5)
        encoder_stage_5=self.relu(encoder_stage_5)
        # max pool
        encoder_stage_5,encoder_stage_5_indices=self.encoder_max_pool(encoder_stage_5)

        #########################
        #    End of Encoder     #
        #########################

        ###########################
        #    Start of Decoder     #
        ###########################
        ## Decoding
        # Stage 1
        decoder_stage_1=self.decoder_unpool(encoder_stage_5,encoder_stage_5_indices,encoder_stage_4.size())
        #conv1
        decoder_stage_1=self.decoder_stage_1_conv1(decoder_stage_1)
        decoder_stage_1=self.decoder_stage_1_batch_normalization1(decoder_stage_1)
        decoder_stage_1=self.relu(decoder_stage_1)
        #conv2
        decoder_stage_1=self.decoder_stage_1_conv2(decoder_stage_1)
        decoder_stage_1=self.decoder_stage_1_batch_normalization2(decoder_stage_1)
        decoder_stage_1=self.relu(decoder_stage_1)
        #conv3
        decoder_stage_1=self.decoder_stage_1_conv3(decoder_stage_1)
        decoder_stage_1=self.decoder_stage_1_batch_normalization3(decoder_stage_1)
        decoder_stage_1=self.relu(decoder_stage_1)

        # Stage 2
        decoder_stage_2=self.decoder_unpool(decoder_stage_1,encoder_stage_4_indices,encoder_stage_3.size())
        #conv1
        decoder_stage_2=self.decoder_stage_2_conv1(decoder_stage_2)
        decoder_stage_2=self.decoder_stage_2_batch_normalization1(decoder_stage_2)
        decoder_stage_2=self.relu(decoder_stage_2)
        #conv2
        decoder_stage_2=self.decoder_stage_2_conv2(decoder_stage_2)
        decoder_stage_2=self.decoder_stage_2_batch_normalization2(decoder_stage_2)
        decoder_stage_2=self.relu(decoder_stage_2)
        #conv3
        decoder_stage_2=self.decoder_stage_2_conv3(decoder_stage_2)
        decoder_stage_2=self.decoder_stage_2_batch_normalization3(decoder_stage_2)
        decoder_stage_2=self.relu(decoder_stage_2)

        # Stage 3
        decoder_stage_3=self.decoder_unpool(decoder_stage_2,encoder_stage_3_indices,encoder_stage_2.size())
        #conv1
        decoder_stage_3=self.decoder_stage_3_conv1(decoder_stage_3)
        decoder_stage_3=self.decoder_stage_3_batch_normalization1(decoder_stage_3)
        decoder_stage_3=self.relu(decoder_stage_3)
        #conv2
        decoder_stage_3=self.decoder_stage_3_conv2(decoder_stage_3)
        decoder_stage_3=self.decoder_stage_3_batch_normalization2(decoder_stage_3)
        decoder_stage_3=self.relu(decoder_stage_3)
        #conv3
        decoder_stage_3=self.decoder_stage_3_conv3(decoder_stage_3)
        decoder_stage_3=self.decoder_stage_3_batch_normalization3(decoder_stage_3)
        decoder_stage_3=self.relu(decoder_stage_3)

        # Stage 4
        decoder_stage_4=self.decoder_unpool(decoder_stage_3,encoder_stage_2_indices,encoder_stage_1.size())
        #conv1
        decoder_stage_4=self.decoder_stage_4_conv1(decoder_stage_4)
        decoder_stage_4=self.decoder_stage_4_batch_normalization1(decoder_stage_4)
        decoder_stage_4=self.relu(decoder_stage_4)
        #conv2
        decoder_stage_4=self.decoder_stage_4_conv2(decoder_stage_4)
        decoder_stage_4=self.decoder_stage_4_batch_normalization2(decoder_stage_4)
        decoder_stage_4=self.relu(decoder_stage_4)

        # Stage 5
        decoder_stage_5=self.decoder_unpool(decoder_stage_4,encoder_stage_1_indices)
        #conv1
        decoder_stage_5=self.decoder_stage_5_conv1(decoder_stage_5)
        decoder_stage_5=self.decoder_stage_5_batch_normalization1(decoder_stage_5)
        decoder_stage_5=self.relu(decoder_stage_5)
        #conv2
        decoder_stage_5=self.decoder_stage_5_conv2(decoder_stage_5)
        decoder_stage_5=self.decoder_stage_5_batch_normalization2(decoder_stage_5)
        # No Relu

        return decoder_stage_5
        #########################
        #    End of Decoder     #
        #########################
    