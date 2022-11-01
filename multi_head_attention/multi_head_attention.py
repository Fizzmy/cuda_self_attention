import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np
import time
import random

from torch.utils.cpp_extension import load
cuda_module = load(name="multi_head_attention",
                   sources=["multi_head_attention.cpp", "multi_head_attention.cu","multi_head_layer.cpp"],
                   # extra_ldflags=["cublas.lib"],
                   verbose=True)

def attn_mask(batch_size, tgt_len, dtype=None):
    mask = torch.zeros((batch_size, tgt_len))
    for b in range(batch_size):
        valid_tgt_len = random.randint(1, tgt_len)
        mask[b, valid_tgt_len:] = 1
    return mask.to("cuda:0", dtype=dtype)

batch_size = 32 # 32
tgt_len = 256 # 128
head_num = 8 # 4
hidden_size = 2048 # 256
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
atten_dropout_radio = 0.0
output_dropout_radio = 0.0

# print(inp,qkv)
torch_inp_grad, torch_normw_grad, torch_normb_grad, torch_qkv_grad, torch_o_grad = None, None, None, None, None
cuda_inp_grad, cuda_normw_grad, cuda_normb_grad, cuda_qkv_grad, cuda_o_grad = None, None, None, None, None

output = torch.rand((batch_size,tgt_len,head_num,output_size), device = "cuda:0")
ntest = 1

class MultiHeadAttentionLayer(nn.Module):
    
    def __init__(self,batch_size,tgt_len,head_num,hidden_size,qkv,o):
        super(MultiHeadAttentionLayer,self).__init__()
        
        self.hidden_size = hidden_size
        self.head_num = head_num
        
        #下面使用nn的Linear层来定义Q，K，V矩阵
        self.layernorm = nn.LayerNorm(hidden_size,eps=1e-8).to(device = "cuda:0")
        self.layernorm.weight=Parameter(weight)
        self.layernorm.bias=Parameter(bias)
        self.QKV_linear = nn.Linear(hidden_size, 3 * hidden_size, bias = False)
        self.QKV_linear.weight=Parameter(qkv.permute(1, 0))
        self.O_linear = nn.Linear(hidden_size, hidden_size, bias = False)
        self.O_linear.weight=Parameter(o.permute(1, 0))
        self.input_norm = None
        self.QKV = None
        self.QKVT = None
        self.Q = None
        self.K = None
        self.V = None
        self.softmax_input = None
        self.softmax_output = None
        self.softmax_V = None

    def forward(self, inputs, mask, isPre):
        
        batch_size , tgt_len , hidden_size =inputs.size()
        self.input_norm = self.layernorm(inputs)
        #计算生成QKV矩阵
        self.QKV = self.QKV_linear(self.input_norm) 
        self.QKV.retain_grad()
        # print(QKV)
        self.QKVT = self.QKV.view(batch_size, tgt_len , 3 ,head_num , hidden_size//head_num ).permute(2,0,3,1,4).contiguous().view(3,-1, tgt_len, output_size)
        self.QKVT.retain_grad()

        self.Q = self.QKVT[0]
        self.K = self.QKVT[1]
        self.V = self.QKVT[2]
        self.Q.retain_grad()
        self.K.retain_grad()
        self.V.retain_grad()

        self.softmax_input = torch.bmm(self.Q , self.K.contiguous().permute(0,2,1)).view(-1,tgt_len,tgt_len)
        self.softmax_input.retain_grad()
        
        mask = mask.unsqueeze(1).broadcast_to(batch_size, head_num * tgt_len,tgt_len).reshape(-1,tgt_len,tgt_len)
        if isPre:
            mask = torch.max(mask,torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).to(torch.device("cuda:0"),torch.bool))
        
        self.softmax_input = self.softmax_input.masked_fill(mask == 1, -1e9)

        self.softmax_output = F.softmax(self.softmax_input / np.power(hidden_size//head_num,0.5), dim =2)
        self.softmax_output.retain_grad()

        self.softmax_V = torch.bmm(self.softmax_output , self.V).view( -1 , head_num , tgt_len , output_size).permute(0,2,1,3).contiguous().view( -1 , tgt_len , hidden_size)
        self.softmax_V.retain_grad()

        # print(torch.bmm(QK , V))
        output= self.O_linear(self.softmax_V)

        output += inputs
        return output # .view(-1,tgt_len,head_num,output_size)

att_L = MultiHeadAttentionLayer(batch_size,tgt_len,head_num,hidden_size,qkv,o)

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
    if inp.grad!=None:
        inp.grad.data.zero_()
    att_L.zero_grad()
    att_out = att_L(inp,mask,0)
    loss = att_out.sum()
    print(loss)
    loss.backward()
    # print(att_L.QKV.grad)
    # print(qkv.permute(1,0))
    global torch_inp_grad
    torch_inp_grad = inp.grad.cpu().numpy()
    global torch_normw_grad
    torch_normw_grad = att_L.layernorm.weight.grad.cpu().numpy()
    global torch_normb_grad
    torch_normb_grad = att_L.layernorm.bias.grad.cpu().numpy()
    global torch_qkv_grad
    torch_qkv_grad = att_L.QKV_linear.weight.grad.permute(1,0).cpu().numpy()
    global torch_o_grad
    torch_o_grad = att_L.O_linear.weight.grad.permute(1,0).cpu().numpy()

    return att_out

model = MultiHeadLayer(batch_size, tgt_len, hidden_size, head_num, qkv, o, atten_dropout_radio, output_dropout_radio)

def run_cuda():
    if inp.grad!=None:
        inp.grad.data.zero_()
    model.zero_grad()
    torch.cuda.empty_cache()
    output = model(inp,mask,0)
    loss = output.sum()
    print(loss)
    loss.backward()
    global cuda_inp_grad
    cuda_inp_grad = inp.grad.cpu().numpy()
    global cuda_normw_grad
    cuda_normw_grad = model.weight.grad.cpu().numpy()
    global cuda_normb_grad
    cuda_normb_grad = model.bias.grad.cpu().numpy()
    global cuda_qkv_grad
    cuda_qkv_grad = model.qkv.grad.cpu().numpy()
    global cuda_o_grad
    cuda_o_grad = model.o.grad.cpu().numpy()

    return output

if __name__ == '__main__':

    print("Running cuda...")
    cuda_time , res1 = show_time(run_cuda)
    print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))
    
    print("Running torch...")
    torch_time , res2 = show_time(run_torch)
    print("Torch time:  {:.3f}us".format(np.mean(torch_time)))
    
    

    # print(res1[0])
    np.testing.assert_allclose(res1[0].cpu().detach().numpy(), res2[0].cpu().detach().numpy(), atol=1e-5)
    
    np.testing.assert_allclose(cuda_o_grad, torch_o_grad, atol=1e-2)
    # print(torch_inp_grad)
    # print(cuda_inp_grad)
    
    # np.testing.assert_allclose(cuda_qkv_grad, torch_qkv_grad, rtol=40)
    np.testing.assert_allclose(cuda_normw_grad, torch_normw_grad, atol=1)
    np.testing.assert_allclose(cuda_normb_grad, torch_normb_grad, atol=5)
    np.testing.assert_allclose(cuda_inp_grad, torch_inp_grad, atol=1e-2)
    
   
    
    