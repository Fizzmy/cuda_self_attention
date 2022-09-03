# lightseq分析

### forward

输入：[batch_size * tgt_len * hidden_size]

qkv_linear  ： [(batch_size * tgt_len )* hidden] * [ hidden * (3*hidden)]

一起计算qkv，其中每个qkv的输出维度是output_size = hidden/head

（矩乘使用cublasGemmEx）

得到结果是 [(batch_size * tgt_len)  * (3 * head * output_size) ]

转换成新的维度： [3  * batch_size * head * tgt_len * output_size]  （转换维度可以使用向量化reinterpret_cast 转成float4，有一个简单的提升）

这样可以拆分出q和k相乘，相乘后得到的维度为：[batch_size * head * tgt_len * tgt_len]

接着调用softmax （reduce可能调用了blockReduce API） 变成 softmax_output

最后乘上v，维度变成[batch_size * head * tgt_len * output_size]

转换成新维度：softmax_T = [batch_size * tgt_len * head  * output_size]

最后乘上一个线性变换

### backword

grad维度：[batch_size * tgt_len * hidden_size]

计算出o的梯度 o_grad  =  softmax_T .transform10 * grad

计算出softmax_T的梯度 softmax_T_grad = grad * o_weight.transform10

梯度维度可以看成[batch_size * tgt_len * head * output_size]

转换维度：[batch_size * head * tgt_len * output_size]

计算关于v的梯度 v_grad = softmax_output.transform021 * softmax_T_grad   [batch_size * head * tgt_len * output_size]

计算关于softmax_output的梯度 softmax_output_grad = softmax_T_grad * v_weight.transform 021 [batch_size * head * tgt_len * tgt_len]

关于softmax求梯度

大体可以参考 https://zhuanlan.zhihu.com/p/37740860  相当于softmax_output_grad * $\frac{dY}{dX}$

Y是softmax后，x是softmax前，导数的维度为[tgt_len,tgt_len]

求得softmax_input_grad

然后关于Q和K求导  (Q K^T)

Q_grad = softmax_input_grad * K

K_grad = softmax_input_grad.transform021 * Q

QKV_grad拼一起： [3  * batch_size * head * tgt_len * output_size]  

转换维度：  [(batch_size * tgt_len)  * (3 * head * output_size) ] =   [(batch_size * tgt_len)  * (3 * hidden_size) ] 

求QKV_weights导数：

QKV_weights_grad = input.transform10 * QKV_grad

input_grad = QKV_grad * QKV_weights.transform10