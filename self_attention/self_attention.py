import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np
from torch.utils.cpp_extension import load
import time

cuda_module = load(name="self_attention",
                   sources=["self_attention.cpp", "self_attention.cu"],
                   verbose=True)


class Attention_Layer(nn.Module):
    
    #用来实现mask-attention layer
    def __init__(self,hidden_input_dim,hidden_output_dim,wq,wk,wv):
        super(Attention_Layer,self).__init__()
        
        self.hidden_input_dim = hidden_input_dim

        self.hidden_output_dim = hidden_output_dim
        
        #下面使用nn的Linear层来定义Q，K，V矩阵
        self.Q_linear = nn.Linear(hidden_input_dim, hidden_output_dim, bias = False)
        #mid=torch.rand(hidden_output_dim,hidden_input_dim)
        self.Q_linear.weight=Parameter(wq.permute(1, 0))
        self.K_linear = nn.Linear(hidden_input_dim, hidden_output_dim, bias = False)
        #mid=torch.rand(hidden_output_dim,hidden_input_dim)
        self.K_linear.weight=Parameter(wk.permute(1, 0))
        self.V_linear = nn.Linear(hidden_input_dim, hidden_output_dim, bias = False)
        #mid=torch.rand(hidden_output_dim,hidden_input_dim)
        self.V_linear.weight=Parameter(wv.permute(1, 0))
            
        
    def forward(self, inputs):
        
        #计算生成QKV矩阵
        Q = self.Q_linear(inputs) 
        K = self.K_linear(inputs).permute(0 ,2 ,1)#先进行一次转置
        V = self.V_linear(inputs)
        # print(Q)
        #下面开始计算啦
        alpha = torch.matmul(Q, K)
        # print(alpha)
        # 下面开始softmax
        alpha = F.softmax(alpha, dim = 2)
        #print('\nalpha is :', alpha)
        
        out = torch.matmul(alpha, V)
        
        return out

batch_size = 16 
leng = 128
input_size = 512
output_size = 256
torch.cuda.manual_seed(0)
torch.set_printoptions(sci_mode=False)
inp=torch.rand((batch_size,leng,input_size), device="cuda:0")
wq=torch.rand((input_size,output_size), device="cuda:0")
wk=torch.rand((input_size,output_size), device="cuda:0")
wv=torch.rand((input_size,output_size), device="cuda:0")
output=torch.rand((batch_size,leng,output_size), device="cuda:0")
ntest = 10

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
    att_L = Attention_Layer(input_size,output_size,wq,wk,wv)
    att_out = att_L(inp)
    return att_out

def run_cuda():
    cuda_module.torch_launch_self_attention(inp,wq,wk,wv,batch_size,leng,input_size,output_size,output)
    return output

if __name__ == '__main__':

    print("Running cuda...")
    cuda_time , res1 = show_time(run_cuda)
    print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))

    print("Running torch...")
    torch_time , res2 = show_time(run_torch)
    print("Torch time:  {:.3f}us".format(np.mean(torch_time)))

    np.testing.assert_allclose(res1[0].cpu().numpy(), res2[0].cpu().detach().numpy(), rtol=1e-4)
    
