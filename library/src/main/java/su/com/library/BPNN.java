package su.com.library;

public class BPNN {

    public interface BPNNCallback {
        void aSampleResult(String result);

        void allSampleResult(String result);
    }

    public static final String Input = "input";
    public static final String Sigmoid = "sigmoid";
    public static final String Relu = "relu";
    public static final String Softmax = "softmax";

    public static class Config {
        static String[] layers_name;//每层名称（激活函数名称）
        static int[] layers_count;//每层维度（节点数）
        static int layer_count;//层数（包括输入层和输出层）
        static int inNum;//网络输入维度
        static int outNum;//网络输出维度
        static float learn_rate;//学习率
        static float min_err;//最小误差
        static boolean useLog;//是否允许日志输出
    }

    private Layer[] layerModels;

    public String config(String[] layers_name, int[] layers_count, float learn_rate, float min_err, boolean useLog) {
        Config.layer_count = layers_name.length - 1;
        Config.layers_name = layers_name;
        Config.layers_count = layers_count;
        Config.learn_rate = learn_rate;
        Config.min_err = min_err;
        Config.inNum = layers_count[0];
        Config.outNum = layers_count[layers_count.length - 1];
        Config.useLog = useLog;
        layerModels = new Layer[Config.layer_count];
        for (int i = 0; i < Config.layer_count; i++) {
            Layer l = new Layer();
            l.init(i, Config.layers_count[i], Config.layers_count[i + 1]);
            layerModels[i] = l;
        }
        StringBuilder result = new StringBuilder();
        result.append("\n=======网络结构=======\n");
        result.append("层数：" + (Config.layer_count + 1) + "\n");
        result.append("每层名称：");
        for (int i = 0; i < Config.layer_count + 1; i++) {
            result.append(Config.layers_name[i] + " ");
        }
        result.append("\n");
        result.append("每层维度：");
        for (int i = 0; i < Config.layer_count + 1; i++) {
            result.append(Config.layers_count[i] + " ");
        }
        result.append("\n");
        result.append("学习率：" + Config.learn_rate + "\n");
        result.append("网络输入维度：" + Config.inNum + "\n");
        result.append("网络输出维度：" + Config.outNum + "\n");
        result.append("允许日志输出：" + Config.useLog + "\n");
        result.append("====================\n");
        return result.toString();
    }

    //训练
    public void train(float[][] samples, float[][] labels, BPNNCallback bpnnCallback, Layer.LayerCallback layerCallback) {
        int sample_count = samples.length;
        float err = Float.MAX_VALUE;
        while (err > Config.min_err) {//循环训练所有样本直到误差达标
            float inner_err = 0;
            for (int i = 0; i < sample_count; i++) {//训练所有样本
                float[] input = samples[i];//当前样本的输入
                float[] label = labels[i];//当前样本的标记
                if (!Config.useLog) {
                    layerCallback = null;
                }
                Layer.ReturnModel model = null;
                for (int j = 0; j < layerModels.length; j++) {//前向运算
                    model = layerModels[j].forward(input, layerCallback);
                    input = model.getOutput();
                }
                float[] output = model.getOutput();//一个样本输出
                inner_err += layerModels[layerModels.length - 1].getErr(label);//累加每一个样本的误差
                float[] v_next = null;
                float[] o = output;
                Layer.ReturnModel model2 = null;
                float[][] mat_next = null;
                for (int j = layerModels.length - 1; j >= 0; j--) {//反向运算
                    model2 = layerModels[j].backward(mat_next, v_next, o, label);
                    mat_next = model2.getMat_next();
                    v_next = model2.getV();
                }
                for (int j = 0; j < layerModels.length; j++) {//修改参数
                    layerModels[j].apply();
                }
            }
            err = inner_err;
            System.out.println(err);
        }
    }

    //测试
    public float test(float[][] samples, float[][] labels, BPNNCallback BPNNCallback) {
        int sample_count = samples.length;
        int sum=0;
        float rate;//测试正确率
        for (int i = 0; i < sample_count; i++) {
            float[] input = samples[i];
            float[] label = labels[i];
            Layer.ReturnModel model = null;
            for (int j = 0; j < layerModels.length; j++) {
                model = layerModels[j].forward(input, null);
                input = model.getOutput();
            }
            float[] output = model.getOutput();
            int maxIndex=Util.maxIndex(output);
            int targetIndex=Util.maxIndex(label);
            if(maxIndex==targetIndex){
                sum +=1;
            }
        }
        rate=(float)sum/sample_count;
        System.out.println("测试正确率："+rate);
        return rate;
    }

    //预测
    public void predict(float[][] samples, float[][] labels, BPNNCallback BPNNCallback) {

    }
}
