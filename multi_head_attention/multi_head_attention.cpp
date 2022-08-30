#include <torch/extension.h>
#include "multi_head_attention.h"


void torch_launch_multi_head_attention(torch::Tensor & input,torch::Tensor &qkv,torch::Tensor &o,int batch_size,int tgt_len,int head_num,int hidden_size,torch::Tensor &output)
{
    launch_multi_head_attention((float* )input.data_ptr(),(float*) qkv.data_ptr(),(float*) o.data_ptr(),batch_size,tgt_len,head_num,hidden_size,(float*) output.data_ptr());

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launch_multi_head_attention",
          &torch_launch_multi_head_attention,
          "self attention kernel");
}