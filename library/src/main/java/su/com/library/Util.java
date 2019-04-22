package su.com.library;

public class Util {

    public static float sigmoid(float x){
        return (float) ((float)1/(1+Math.exp(-x)));
    }

    public static float deSigmoid(float y){
        return y*(1-y);
    }

    public static float relu(float x){
        return x>0?x:0;
    }

    public static float deRelu(float y){
        return y>0?1:0;
    }

    public static float softmax(float[] xs,int index){
        float base=0;
        int count=xs.length;
        for(int i=0;i<count;i++){
            base+=Math.exp(-(xs[i]-xs[index]));
        }
        return (float)1/base;
    }

    public static int maxIndex(float[] values){
        float max=Float.MIN_VALUE;
        int index=-1;
        for(int i=0;i<values.length;i++){
            if(max<values[i]){
                max=values[i];
                index=i;
            }
        }
        return index;
    }
}
