  # -*- coding: UTF-8 -*- 
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np
from torch.utils.cpp_extension import load
import time
import os
import random


cuda_module = load(name="multi_head_attention",
                   sources=["multi_head_attention.cpp", "multi_head_attention.cu","multi_head_layer.cpp"],
                   verbose=True)

def attn_mask(batch_size, tgt_len, dtype=None):
    mask = torch.zeros((batch_size, tgt_len))
    for b in range(batch_size):
        valid_tgt_len = random.randint(1, tgt_len)
        mask[b, valid_tgt_len:] = 1
    return mask.to("cuda:0", dtype=dtype)

batch_size = 32 # 32
tgt_len = 256 # 128
head_num = 16 # 4
hidden_size = 4096 # 256
output_size = hidden_size // head_num
torch.cuda.manual_seed(0)
random.seed(0)
torch.set_printoptions(sci_mode=False)
# inp = torch.normal(mean=torch.full((batch_size,tgt_len,hidden_size),0.0),std=torch.full((batch_size,tgt_len,hidden_size),1))
# inp = torch.tensor(inp,device = "cuda:0",requires_grad=True)
inp = torch.randn((batch_size,tgt_len,hidden_size), device = "cuda:0",requires_grad=True)
mask = attn_mask(batch_size,tgt_len,bool)
weight = torch.ones((hidden_size),device = "cuda:0",requires_grad=True)
bias = torch.zeros((hidden_size),device = "cuda:0",requires_grad=True)
qkv = torch.rand((hidden_size,3*hidden_size),device = "cuda:0",requires_grad=True) * pow(2,0.5) * pow(3 / hidden_size, 0.5)
o = torch.rand((hidden_size,hidden_size), device = "cuda:0",requires_grad=True) * pow(2,0.5) * pow(3 / hidden_size, 0.5)
# print(inp,qkv)
torch_inp_grad, torch_normw_grad, torch_normb_grad, torch_qkv_grad, torch_o_grad = None, None, None, None, None
cuda_inp_grad, cuda_normw_grad, cuda_normb_grad, cuda_qkv_grad, cuda_o_grad = None, None, None, None, None

atten_dropout_radio = 0.0
output_dropout_radio = 0.0



output = torch.rand((batch_size,tgt_len,head_num,output_size), device = "cuda:0")
ntest = 10

class MultiHeadFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, mask, weight, bias, qkv, o, isPre, training):
        # ctx.save_for_backward(inp,qkv,o)
        output = cuda_module.torch_launch_multi_head_attention(inp,mask,weight,bias,qkv,o,isPre,training)
        # print(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # inp, qkv, o = ctx.saved_tensors
        # grad = grad_output.clone()
        input_grad,normw_grad,normb_grad,qkv_grad,o_grad = cuda_module.torch_launch_multi_head_attention_bw(grad_output.contiguous())
        return (input_grad,None,normw_grad,normb_grad,qkv_grad,o_grad,None,None)

class MultiHeadLayer(nn.Module):
    def __init__(self, batch_size, tgt_len, hidden_size, head_num , qkv , o, atten_dropout_radio, output_dropout_radio):
        super(MultiHeadLayer, self).__init__()
        cuda_module.init(batch_size,tgt_len,head_num,hidden_size,atten_dropout_radio,output_dropout_radio)
        self.batch_size = batch_size;
        self.tgt_len = tgt_len;
        self.hidden_size = hidden_size;
        self.head_num = head_num;
        self.output_size = hidden_size // head_num;
        self.atten_dropout_radio = atten_dropout_radio
        self.output_drouput_radio = output_dropout_radio
        self.weight = nn.Parameter(weight.float())
        self.bias = nn.Parameter(bias.float())
        self.qkv = nn.Parameter(qkv.float())
        self.o = nn.Parameter(o.float())
        

    def forward(self,inp, mask, isPre):
        batch_size , tgt_len , _ = inp.size() 
        output = MultiHeadFunction.apply(inp,mask,self.weight,self.bias,self.qkv,self.o,isPre,self.training)
        return output
    
    
    

def show_time(func):
    times = list()
    res = list()
    func() # warm up
    for _ in range(ntest):
        # inp = torch.randn((batch_size,tgt_len,hidden_size), device = "cuda:0",requires_grad=True)
        if inp.grad!=None:
            inp.grad.data.zero_()
        model.zero_grad()
        torch.cuda.synchronize(device="cuda:0")
        start_time = time.time()
        func()
        torch.cuda.synchronize(device="cuda:0")
        end_time = time.time()
        times.append((end_time-start_time)*1e3)
        # res.append(r)
    return times


    
model = MultiHeadLayer(batch_size, tgt_len, hidden_size, head_num, qkv, o, atten_dropout_radio, output_dropout_radio)
def run_cuda():
    output = model(inp,mask,0)
    loss = output.sum()
    print(loss)
    loss.backward()

if __name__ == '__main__':

    print("Running cuda...")
    cuda_time  = show_time(run_cuda)
    print("Cuda time:  {:.3f}ms".format(np.mean(cuda_time)))

   
    
    