package su.com.bpnn;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;

import su.com.library.BPNN;
import su.com.library.Layer;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        BPNN bpnn=new BPNN();
        String configResult=bpnn.config(new String[]{BPNN.Input,BPNN.Sigmoid,BPNN.Sigmoid},new int[]{3,4,1},
                0.9f,0.01f,true);
        System.out.println(configResult);
        bpnn.train(new float[][]{
                {1,2,3},
                {1,2,3},
                {1,2,3},
                {1,2,3}
        }, new float[][]{
                {1},
                {1},
                {1},
                {1}
        }, new BPNN.BPNNCallback() {
            @Override
            public void aSampleResult(String result) {

            }

            @Override
            public void allSampleResult(String result) {

            }
        }, new Layer.LayerCallback() {
            @Override
            public void layerResult(String result) {

            }
        });
        bpnn.test(new float[][]{
                {1,2,3},
                {1,2,3},
                {1,2,3},
                {1,2,3}
        },new float[][]{
                {1},
                {1},
                {1},
                {1}
        },null);
    }
}
