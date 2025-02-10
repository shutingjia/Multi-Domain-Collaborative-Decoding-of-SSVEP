# Time:2024/6/20 19:41
import torch
from torch import nn
from Utils import Constraint
import numpy as np
import torch.nn.functional as F
from Utils import Constraint
class LSTM(nn.Module):
    '''
        Employ the Bi-LSTM to learn the reliable dependency between spatio-temporal features
    '''
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=True, num_layers=1)
    # (30,32,1,60)->(30,32,60)
    def forward(self, x):
        b, c, T = x.size()
        x = x.view(x.size(-1), -1, c)  # (b, c, T) -> (T, b, c)(60,30,32)
        r_out, _ = self.rnn(x)  # r_out shape [time_step * 2, batch_size, output_size]
        out = r_out.view(b, 2 * T * c, -1)
        return out #(30,3840,1)

class multi_ch_Corr(nn.Module):
    def __init__(self, num_classes, **kwargs):
        self.num_classes = num_classes
        self.corr = None
        super(multi_ch_Corr, self).__init__(**kwargs)
    def forward(self, input, **kwargs):
        x = input[0]  # [bs, tw * kernel_size_2]  signal
        x_ = torch.reshape(x, (-1, 320, self.num_classes))  # [bs, tw, kernel_size_2, 1]


        t = input[1]  # [bs, 1, tw * kernel_size_2, cl] reference
        t_ = torch.reshape(t, (-1, 320, self.num_classes))  # [bs, tw, kernel_size_2, cl]

        corr_xt = torch.sum(x_*t_, dim=1)  # [bs, kernel_size_2, cl]
        #print('////////',corr_xt.shape)
        corr_xx = torch.sum(x_*x_, dim=1)  # [bs, kernel_size_2, 1]
        corr_tt = torch.sum(t_*t_, dim=1)  # [bs, kernel_size_2, cl]
        self.corr = corr_xt/torch.sqrt(corr_tt)/torch.sqrt(corr_xx)  # [bs, kernel_size_2, cl]
        self.out = self.corr  # [bs, kernel_size_2, cl]
        #self.out = torch.mean(self.out, dim=1)  # [bs, cl]
        return self.out

class ESNet(nn.Module):
    def calculateOutSize(self, model, nChan, nTime):
        '''
            Calculate the output based on input size
            model is from nn.Module and inputSize is a array
        '''
        data = torch.randn(1, 1, nChan, nTime)
        out = model(data).shape
        print('this is out.shape=',out)
        return out[1:]

    def spatial_block(self, nChan, dropout_level):
        '''
           Spatial filter block,assign different weight to different channels and fuse them
        (30,1,8,128)-->(30,16,1,128)
        '''
        block = []
        block.append(Constraint.Conv2dWithConstraint(in_channels=1, out_channels=nChan * 2, kernel_size=(nChan, 1),
                                                     max_norm=1.0))
        block.append(nn.BatchNorm2d(num_features=nChan * 2))
        block.append(nn.PReLU())
        block.append(nn.Dropout(dropout_level))
        layer = nn.Sequential(*block)
        return layer

    def enhanced_block(self, in_channels, out_channels, dropout_level, kernel_size, stride):
        '''
           Enhanced structure block,build a CNN block to absorb data and output its stable feature
        (30,16,1,128)-->(30,32,1,60)
        '''
        block = []
        block.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, kernel_size),
                               stride=(1, stride)))
        block.append(nn.BatchNorm2d(num_features=out_channels))
        block.append(nn.PReLU())
        block.append(nn.Dropout(dropout_level))
        layer = nn.Sequential(*block)
        return layer

    def __init__(self, num_channels, T, num_classes):
        super(ESNet, self).__init__()
        self.dropout_level = 0.5
        self.F = [num_channels * 2] + [num_channels * 4]
        self.K = 10
        self.S = 2

        net = []
        net.append(self.spatial_block(num_channels, self.dropout_level))
        net.append(self.enhanced_block(self.F[0], self.F[1], self.dropout_level,
                                           self.K, self.S))

        self.conv_layers = nn.Sequential(*net)

        self.fcSize = self.calculateOutSize(self.conv_layers, num_channels, T)
        self.fcUnit = self.fcSize[0] * self.fcSize[1] * self.fcSize[2] * 2
        self.D1 = self.fcUnit // 10
        self.D2 = self.D1 // 5
        #(30,32,1,60)
        self.rnn = LSTM(input_size=self.F[1], hidden_size=self.F[1])#(30,3840,1)
        self.flatten = nn.Flatten()
        self.bn = nn.BatchNorm1d(3840)
        self.corr = multi_ch_Corr(num_classes)
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5, 1), padding=(2, 0))

        self.dense_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.fcUnit, self.D1),
            nn.PReLU(),
            nn.Linear(self.D1, self.D2),
            nn.PReLU(),
            nn.Dropout(self.dropout_level),
            nn.Linear(self.D2, num_classes)

            # nn.PReLU(),
            # nn.Dropout(self.dropout_level),
            # nn.Linear(self.fcUnit, num_classes)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(self.fcUnit, self.D1),
            nn.PReLU())

        self.fc2 = nn.Sequential(nn.Linear(self.D1, self.D2),
                                 nn.PReLU(),
                                 nn.Dropout())

        self.fc3 = nn.Linear(self.D2, num_classes)

    def forward(self, x,template):#(30,1,8,128)

        out_sig = self.conv_layers(x)
        out_sig = out_sig.squeeze(2)          #(30,32,60)
        r_out = self.rnn(out_sig)         # (30,3840,1)
        all_datas = []
        rawdata = out_sig  # 1号被试
        t,y,u = out_sig.size()
        rawdata = rawdata.cpu()
        rawdata = rawdata.data.numpy()
        for k in range(t):  # 40个trail  permutation = [47, 53, 54, 55, 56, 57, 60, 61, 62]
            for ch in range(y):
                O1_data = rawdata[k, ch, :]  # 取中间5s,去除140ms latency
                O1_FFT_list = np.fft.fft(O1_data)   # DFT (sampling_rate/resolution) = 1250
                # O1_FFT_list = np.concatenate(((np.real(O1_FFT_list))[round(5 / 0.2):round(55 / 0.2) + 1],
                #                               (np.imag(O1_FFT_list))[round(5 / 0.2):round(55 / 0.2) + 1]), axis=0)
                all_datas.append(np.array([O1_FFT_list]))
        FFT_matrix = torch.Tensor(all_datas)
        FFT_matrix = FFT_matrix.view(t,y ,u )
        FFT_matrix = FFT_matrix.to('cuda')
        r_out2 = self.rnn(FFT_matrix)
        r_out = r_out + r_out2   #(30,3840,1)
        #print('/////////////',r_out.shape)
        sig = self.flatten(r_out)
        sig = self.bn(sig)

        out_template = self.conv_layers(template)
        #print('1=', out_template.shape)
        out_template = out_template.squeeze(2)  # (30,32,60)
        #print('2=', out_template.shape)
        r_out_template = self.rnn(out_template)  # (30,3840,1)
        #print('3=', r_out_template.shape)
        all_datas_template = []
        rawdata_template = out_template  # 1号被试
        t_template, y_template, u_template = out_template.size()
        rawdata_template = rawdata_template.cpu()
        rawdata_template = rawdata_template.data.numpy()
        for k_template in range(t_template):  # 40个trail  permutation = [47, 53, 54, 55, 56, 57, 60, 61, 62]
            for ch_template in range(y_template):
                O1_data_template = rawdata_template[k_template, ch_template, :]  # 取中间5s,去除140ms latency
                O1_FFT_list_template = np.fft.fft(O1_data_template)  # DFT (sampling_rate/resolution) = 1250
                # O1_FFT_list = np.concatenate(((np.real(O1_FFT_list))[round(5 / 0.2):round(55 / 0.2) + 1],
                #                               (np.imag(O1_FFT_list))[round(5 / 0.2):round(55 / 0.2) + 1]), axis=0)
                all_datas_template.append(np.array([O1_FFT_list_template]))
        FFT_matrix_template = torch.Tensor(all_datas_template)
        FFT_matrix_template = FFT_matrix_template.view(t_template, y_template, u_template)
        FFT_matrix_template = FFT_matrix_template.to('cuda')
        r_out2_template = self.rnn(FFT_matrix_template)

        r_out_template = r_out_template + r_out2_template
        #print('4=',r_out_template.shape)
        tpt = self.flatten(r_out_template)
        #print('5=', tpt.shape)
        tpt = self.bn(tpt)

        corr = self.corr([sig, tpt])
        #print('6=', corr.shape)
        out = torch.reshape(corr, [-1, 12, 1, 1])  # [bs, cl, 1, 1]
        #print('7=', out.shape)
        out = torch.transpose(out, 1, 2)  # [bs, 1, cl, 1]
        #print('8=', out.shape)
        out = self.conv(out)  # [bs, 1, cl, 1]
        #print('9=', out.shape)
        out = torch.reshape(out, [-1, 12])  # [bs, cl]
        #print('10=', out.shape)

        return   out, corr
