import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np
from torch.utils.cpp_extension import load
import time

cuda_module = load(name="multi_head_attention",
                   sources=["multi_head_attention.cpp", "multi_head_attention.cu"],
                   verbose=True)


batch_size = 32
tgt_len = 128
head_num = 4
hidden_size = 1024
output_size = hidden_size // head_num
torch.cuda.manual_seed(0)
torch.set_printoptions(sci_mode=False)
inp = torch.rand((batch_size,tgt_len,hidden_size), device = "cuda:0")
qkv = torch.rand((hidden_size,3*hidden_size),device = "cuda:0")
o = torch.rand((hidden_size,hidden_size), device = "cuda:0")


output = torch.rand((batch_size,tgt_len,head_num,output_size), device = "cuda:0")
ntest = 10

class MultiHeadAttentionLayer(nn.Module):
    
    #用来实现mask-attention layer
    def __init__(self,head_num,hidden_size,qkv,o):
        super(MultiHeadAttentionLayer,self).__init__()
        
        self.hidden_size = hidden_size

        self.head_num = head_num
        
        #下面使用nn的Linear层来定义Q，K，V矩阵
        self.QKV_linear = nn.Linear(hidden_size, 3 * hidden_size, bias = False)
        self.QKV_linear.weight=Parameter(qkv.permute(1, 0))
        self.O_linear = nn.Linear(hidden_size, hidden_size, bias = False)
        self.O_linear.weight=Parameter(o.permute(1, 0))
            
        
    def forward(self, inputs):
        
        batch_size , tgt_len , hidden_size =inputs.size()
        #计算生成QKV矩阵
        QKV = self.QKV_linear(inputs) 
        # print(QKV)
        QKV = QKV.view(batch_size, tgt_len , 3 ,head_num , hidden_size//head_num ).permute(2,0,3,1,4).contiguous().view(3,-1, tgt_len, output_size)
        Q = QKV[0]
        K = QKV[1]
        V = QKV[2]

        QK = torch.bmm(Q , K.contiguous().permute(0,2,1)).view(-1,tgt_len,tgt_len)
        
        QK = QK / np.power(hidden_size//head_num,0.5)

        QK = F.softmax( QK , dim = 2)
        
        output = torch.bmm(QK , V).view( -1 , head_num , tgt_len , output_size).permute(0,2,1,3).contiguous().view( -1 , tgt_len , hidden_size)
        # print(torch.bmm(QK , V))
        output= self.O_linear(output)

        return output.view(-1,tgt_len,head_num,output_size)

att_L = MultiHeadAttentionLayer(head_num,hidden_size,qkv,o)

def show_time(func):
    times = list()
    res = list()
    for _ in range(ntest):
        torch.cuda.synchronize(device="cuda:0")
        start_time = time.time()
        r = func()
        torch.cuda.synchronize(device="cuda:0")
        end_time = time.time()
        times.append((end_time-start_time)*1e6)
        res.append(r)
    return times , res

def run_torch():

    att_out = att_L(inp)
    return att_out

def run_cuda():
    cuda_module.torch_launch_multi_head_attention(inp,qkv,o,batch_size,tgt_len,head_num,hidden_size,output)
    return output

if __name__ == '__main__':

    print("Running cuda...")
    cuda_time , res1 = show_time(run_cuda)
    print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))

    print("Running torch...")
    torch_time , res2 = show_time(run_torch)
    print("Torch time:  {:.3f}us".format(np.mean(torch_time)))

    # print(res1[0])
    np.testing.assert_allclose(res1[0].cpu().numpy(), res2[0].cpu().detach().numpy(), rtol=1e-4)
    