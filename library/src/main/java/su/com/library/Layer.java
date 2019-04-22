package su.com.library;

import android.util.Log;

public class Layer {

    public interface LayerCallback {
        void layerResult(String result);
    }

    private int inNum;//输入维度
    private int outNum;//输出维度
    private float[][] mat;//知识矩阵
    private float[] inputs;//输入
    private float[] outputs;//输出
    private float[] v;//更新量
    private float[] bias;//偏置
    private int layerNo;//层级编号

    class ReturnModel{

        private float[] output;//当前层输出（给下一层使用）
        private float[] v;//当前层更新量（给上一层使用）
        private float[][] mat_next;//下一层权重矩阵

        ReturnModel( float[] output, float[] v,float[][] mat_next) {
            this.output = output;
            this.v=v;
            this.mat_next=mat_next;
        }

        float[] getOutput() {
            return output;
        }

        float[] getV() {
            return v;
        }

        public float[][] getMat_next() {
            return mat_next;
        }
    }

    void init(int layerNo, int inNum, int outNum){
        this.layerNo=layerNo;
        this.inNum=inNum;
        this.outNum=outNum;
        this.mat=new float[outNum][inNum];
        this.outputs=new float[outNum];
        this.bias=new float[outNum];
        this.v=new float[outNum];
        for(int i=0;i<outNum;i++){
            for(int j=0;j<inNum;j++){
                mat[i][j]= (float) (Math.random()*0.01);
            }
            bias[i]= (float) (Math.random()*0.01);
            outputs[i]=0;
            v[i]=0;
        }
    }

    ReturnModel forward(float[] inputs, LayerCallback layerCallback){
        this.inputs=inputs;
        for(int i=0;i<outNum;i++){
            float temp=0;
            for(int j=0;j<inNum;j++){
                temp+=mat[i][j]*inputs[j];
            }
            outputs[i]=temp+bias[i];
        }
        //输出修正
        if(layerNo==BPNN.Config.layer_count-1){//最后一层
            if(outNum==1){//二分类情况
                if(BPNN.Config.layers_name[layerNo+1].equals(BPNN.Sigmoid)){
                    outputs[0]=Util.sigmoid(outputs[0]);
                }else if(BPNN.Config.layers_name[layerNo+1].equals(BPNN.Relu)){
                    outputs[0]=Util.relu(outputs[0]);
                }else{
                    Log.e("激活函数选择错误","二分类最后一层请选择sigmoid或relu激活函数");
                    throw new IllegalArgumentException("参数配置错误");
                }
            }else{//多分类情况
                //TODO 多分类正向传播
            }
        }else{//不是最后一层
            if(BPNN.Config.layers_name[layerNo+1].equals(BPNN.Sigmoid)){
                for(int i=0;i<outputs.length;i++){
                    outputs[i]=Util.sigmoid(outputs[i]);
                }
            }else if(BPNN.Config.layers_name[layerNo+1].equals(BPNN.Relu)){
                for(int i=0;i<outputs.length;i++){
                    outputs[i]=Util.relu(outputs[i]);
                }
            }else{
                Log.e("激活函数选择错误","非最后一层的激活函数只能是relu或者sigmoid");
                throw new IllegalArgumentException("参数配置错误");
            }
        }
        //日志回调
        if(layerCallback!=null){
            StringBuilder log=new StringBuilder();
            for(int i=0;i<outNum;i++){
                log.append(outputs[i]+" ");
            }
            layerCallback.layerResult("第"+layerNo+"层的输出："+log.toString());
        }
        //结果返回
        return new ReturnModel(outputs,null,null);
    }

    ReturnModel backward(float[][] mat_next, float[] v_next, float[] o, float[] label){
        ReturnModel model=null;
        if(layerNo==BPNN.Config.layer_count-1){//如果是最后一层，计算更新量
            float err=0;
            if(outNum==1){//二分类情况
                for(int i=0;i<label.length;i++){
                    err=err+0.5f*(label[i]-o[i])*(label[i]-o[i]);//平方误差
                }
                if(BPNN.Config.layers_name[layerNo].equals(BPNN.Sigmoid)){
                    v[0]=(label[0]-o[0])*Util.deSigmoid(o[0]);
                }else if(BPNN.Config.layers_name[layerNo].equals(BPNN.Relu)){
                    v[0]=(label[0]-o[0])*Util.deRelu(o[0]);
                }else{
                    Log.e("激活函数选择错误","二分类最后一层激活函数只能选择sigmoid或relu");
                    throw new IllegalArgumentException("参数配置错误");
                }
                model=new ReturnModel(null,v,mat);
            }else{//多分类情况
                //TODO 多分类更新量反向传播
            }
        }else{//如果不是最后一层，用后一层的更新量计算当前层的更新量
            for(int i=0;i<outNum;i++){
                float temp=0;
                for(int j=0;j<v_next.length;j++){
                    temp+=v_next[j]*mat_next[j][i];
                }
                v[i]=temp;
            }
            model=new ReturnModel(null,v,mat);
        }
        return model;
    }

    public void apply(){
        //用更新量更新每一层的参数
        for(int i=0;i<outNum;i++){
            for(int j=0;j<inNum;j++){
                mat[i][j]+=BPNN.Config.learn_rate*v[i]*inputs[j];
            }
            bias[i]+=BPNN.Config.learn_rate*v[i];
        }
    }

    public float getErr(float[] label){
        if(layerNo==BPNN.Config.layer_count-1){//最后一层才有误差
            if(BPNN.Config.layers_count[BPNN.Config.layer_count]==1){//二分类
                return label[0]-outputs[0];
            }else{//多分类
                //TODO 多分类误差
            }
        }
        return 0;
    }
}
