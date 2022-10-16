class MultiHeadLayer{
    public:
        MultiHeadLayer();
        ~MultiHeadLayer();
        void SetConfig(int bs,int tl,int hn,int hs);
        void SetWeight(float *input_w,float *norm_w,float *norm_b,float *qkv_w,float *o_w);
        void FreeAll();
        void Forward(float* output);
        void Backward(float* grad, float* input_grad,float* normw_grad,float *normb_grad,float* qkv_grad,float* o_grad);
        int getBatchSize();
        int getTgtLen();
        int getHeadNum();
        int getHiddenSize();
        int getOutputSize();
    private:
        int batch_size;
        int tgt_len;
        int head_num;
        int hidden_size;
        int output_size;
        float *QKVT,*softmax_output,*softmax_T,*input_norm,*input_mean,*input_std,*input_hat;
        float *qkv,*o,*input,*normw,*normb;
};