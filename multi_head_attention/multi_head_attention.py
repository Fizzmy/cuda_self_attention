import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np
from torch.utils.cpp_extension import load
import time

cuda_module = load(name="multi_head_attention",
                   sources=["multi_head_attention.cpp", "multi_head_attention.cu","multi_head_layer.cpp"],
                   verbose=True)


batch_size = 32 # 32
tgt_len = 128 # 128
head_num = 4 # 4
hidden_size = 256 # 256
output_size = hidden_size // head_num
torch.cuda.manual_seed(0)
torch.set_printoptions(sci_mode=False)
# inp = torch.normal(mean=torch.full((batch_size,tgt_len,hidden_size),0.0),std=torch.full((batch_size,tgt_len,hidden_size),1))
# inp = torch.tensor(inp,device = "cuda:0",requires_grad=True)
inp = torch.randn((batch_size,tgt_len,hidden_size), device = "cuda:0",requires_grad=True)
weight = torch.randn((hidden_size),device = "cuda:0",requires_grad=True)
bias = torch.randn((hidden_size),device = "cuda:0",requires_grad=True)
qkv = torch.randn((hidden_size,3*hidden_size),device = "cuda:0",requires_grad=True)
o = torch.randn((hidden_size,hidden_size), device = "cuda:0",requires_grad=True)
# print(inp,qkv)
torch_inp_grad, torch_normw_grad, torch_normb_grad, torch_qkv_grad, torch_o_grad = None, None, None, None, None
cuda_inp_grad, cuda_normw_grad, cuda_normb_grad, cuda_qkv_grad, cuda_o_grad = None, None, None, None, None

output = torch.rand((batch_size,tgt_len,head_num,output_size), device = "cuda:0")
ntest = 5

class MultiHeadAttentionLayer(nn.Module):
    
    #用来实现mask-attention layer
    def __init__(self,head_num,hidden_size,qkv,o):
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

    def forward(self, inputs):
        
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

        self.softmax_output = F.softmax(self.softmax_input / np.power(hidden_size//head_num,0.5), dim =2)
        self.softmax_output.retain_grad()

        self.softmax_V = torch.bmm(self.softmax_output , self.V).view( -1 , head_num , tgt_len , output_size).permute(0,2,1,3).contiguous().view( -1 , tgt_len , hidden_size)
        self.softmax_V.retain_grad()
        # print(torch.bmm(QK , V))
        output= self.O_linear(self.softmax_V)

        return output # .view(-1,tgt_len,head_num,output_size)

att_L = MultiHeadAttentionLayer(head_num,hidden_size,qkv,o)

class MultiHeadFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, weight, bias, qkv, o, batch_size, tgt_len, head_num, hidden_size):
        # ctx.save_for_backward(inp,qkv,o)
        output = torch.empty((batch_size,tgt_len,hidden_size)).to(device="cuda:0")
        cuda_module.torch_launch_multi_head_attention(inp,weight,bias,qkv,o,batch_size,tgt_len,head_num,hidden_size,output)
        # print(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # inp, qkv, o = ctx.saved_tensors
        # grad = grad_output.clone()
        input_grad,normw_grad,normb_grad,qkv_grad,o_grad = cuda_module.torch_launch_multi_head_attention_bw(grad_output.contiguous())
        return (input_grad,normw_grad,normb_grad,qkv_grad,o_grad,None,None,None,None)

class MultiHeadLayer(nn.Module):
    def __init__(self, hidden_size, head_num , qkv ,  o):
        super(MultiHeadLayer, self).__init__()
        self.hidden_size = hidden_size;
        self.head_num = head_num;
        self.output_size = hidden_size // head_num;

        self.weight = nn.Parameter(weight.float())
        self.bias = nn.Parameter(bias.float())
        self.qkv = nn.Parameter(qkv.float())
        self.o = nn.Parameter(o.float())

    def forward(self,inp):
        batch_size , tgt_len , _ = inp.size() 
        output = MultiHeadFunction.apply(inp,self.weight,self.bias,self.qkv,self.o,batch_size,tgt_len,head_num,hidden_size)
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
    att_out = att_L(inp)
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

def run_cuda():
    model = MultiHeadLayer(hidden_size, head_num, qkv, o)
    if inp.grad!=None:
        inp.grad.data.zero_()
    model.zero_grad()
    torch.cuda.empty_cache()
    output = model(inp)
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
    np.testing.assert_allclose(res1[0].cpu().detach().numpy(), res2[0].cpu().detach().numpy(), rtol=1)
    
    np.testing.assert_allclose(cuda_o_grad, torch_o_grad, rtol=1e-2)
    # print(torch_inp_grad)
    # print(cuda_inp_grad)
    
    np.testing.assert_allclose(cuda_qkv_grad, torch_qkv_grad, rtol=5)
    np.testing.assert_allclose(cuda_normw_grad, torch_normw_grad, rtol=1e-2)
    np.testing.assert_allclose(cuda_normb_grad, torch_normb_grad, rtol=1e-2)
    np.testing.assert_allclose(cuda_inp_grad, torch_inp_grad, rtol=1e-2)
    
   
    
    