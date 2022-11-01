#include <cublas_v2.h>
class MultiHeadLayer{
    public:
        MultiHeadLayer();
        ~MultiHeadLayer();
        void SetConfig(int bs,int tl,int hn,int hs,float adr,float odr);
        void SetWeight(float *input_w,float *norm_w,float *norm_b,float *qkv_w,float *o_w,bool* mask_w,bool is_pre,bool is_training);
        void SetMalloc();
        void SetGrad();
        void FreeAll();
        void Forward(float* output);
        void Backward(float* grad, float* input_grad,float* normw_grad,float *normb_grad,float* qkv_grad,float* o_grad);
        int getBatchSize();
        int getTgtLen();
        int getHeadNum();
        int getHiddenSize();
        int getOutputSize();
        torch::Tensor output, input_grad, normw_grad, normb_grad, qkv_grad, o_grad;
        bool isPre,isTraining;
        
        float atten_dropout_radio,output_dropout_radio;
        cublasHandle_t handle;
    private:
        int batch_size;
        int tgt_len;
        int head_num;
        int hidden_size;
        int output_size;
        bool *mask;
        bool *atten_mask,*output_mask;
        float *QKVT,*softmax_output,*softmax_T,*input_norm,*input_mean,*input_std,*input_hat;
        float *qkv,*o,*input,*normw,*normb;
        float *softmax_dropout,*output_dropout;
        float *QKV,*softmax_input,*Q,*K,*V,*softmax_T_grad,*softmax_T_grad_T,*QKV_grad,*softmax_output_grad,*Q_grad,*K_grad,*V_grad,*softmax_input_grad,*QKV_grad_T,*norm_grad;
};