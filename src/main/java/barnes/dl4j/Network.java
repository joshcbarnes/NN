package barnes.dl4j;

import java.util.ArrayList;
import java.util.List;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

/**
(P^q) v (q^r) 
TTTT T
TTFT T
TFTF F
TFFF F
FTTT T
FTFF F
FFTF F
FFFF F
 * @author Josh
 *
 */
public class Network {

    private static final int NUM_EPOCHS = 2000;

    public static void main(String [] args) {
        
        Layer hiddenLayer = new DenseLayer.Builder()
            .nIn(4)
            .nOut(7)
            .activation("sigmoid")
            .weightInit(WeightInit.XAVIER)
            .build();
        Layer outputLayer = new OutputLayer.Builder()
            .nIn(7)
            .nOut(1)
            .activation("sigmoid")
            .weightInit(WeightInit.XAVIER)
            .build();
        
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
            .iterations(1)
            .learningRate(0.01)
            .useDropConnect(false)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .biasInit(0)
            .miniBatch(false)
            .list()
                .layer(0, hiddenLayer)
                .layer(1, outputLayer)
                .backprop(true)
                .pretrain(false)
                .build();
        
        MultiLayerNetwork network = new MultiLayerNetwork(config);
        network.init();
        network.setListeners(new ScoreIterationListener(1));
        
        //Training data
        List<Data> inputs = new ArrayList<>();
        List<Data> labelData = new ArrayList<>();
        
        inputs = new ArrayList<>();
//        inputs.add(Data.of(0, 0, 0, 0));
//        inputs.add(Data.of(0, 0, 0, 1));
//        inputs.add(Data.of(0, 0, 1, 0));
        inputs.add(Data.of(0, 0, 1, 1));
        inputs.add(Data.of(0, 1, 0, 0));
        inputs.add(Data.of(0, 1, 0, 1));
        inputs.add(Data.of(0, 1, 1, 0));
        inputs.add(Data.of(0, 1, 1, 1));
        
        inputs.add(Data.of(1, 0, 0, 0));
        inputs.add(Data.of(1, 0, 0, 1));
        inputs.add(Data.of(1, 0, 1, 0));
        inputs.add(Data.of(1, 0, 1, 1));
        inputs.add(Data.of(1, 1, 0, 0));
        inputs.add(Data.of(1, 1, 0, 1));
        inputs.add(Data.of(1, 1, 1, 0));
//        inputs.add(Data.of(1, 1, 1, 1));
        INDArray input = convertToINDArray(inputs);
        
        labelData = new ArrayList<>();
//        labelData.add(Data.of(0));
//        labelData.add(Data.of(0));
//        labelData.add(Data.of(0));
        labelData.add(Data.of(1));
        labelData.add(Data.of(0));
        labelData.add(Data.of(0));
        labelData.add(Data.of(0));
        labelData.add(Data.of(1));
        labelData.add(Data.of(0));
        labelData.add(Data.of(0));
        labelData.add(Data.of(0));
        labelData.add(Data.of(1));
        labelData.add(Data.of(1));
        labelData.add(Data.of(1));
        labelData.add(Data.of(1));
//        labelData.add(Data.of(1));
        INDArray labels = convertToINDArray(labelData);
            
        DataSet data = new DataSet(input, labels);
        
        for (int i = 0; i < NUM_EPOCHS; i++) {
            network.fit(data);
        }
        
        
        inputs = new ArrayList<>();
        inputs.add(Data.of(0, 0, 0, 0));
        inputs.add(Data.of(0, 0, 0, 1));
        inputs.add(Data.of(0, 0, 1, 0));
        inputs.add(Data.of(0, 0, 1, 1));
        inputs.add(Data.of(0, 1, 0, 0));
        inputs.add(Data.of(0, 1, 0, 1));
        inputs.add(Data.of(0, 1, 1, 0));
        inputs.add(Data.of(0, 1, 1, 1));
        
        inputs.add(Data.of(1, 0, 0, 0));
        inputs.add(Data.of(1, 0, 0, 1));
        inputs.add(Data.of(1, 0, 1, 0));
        inputs.add(Data.of(1, 0, 1, 1));
        inputs.add(Data.of(1, 1, 0, 0));
        inputs.add(Data.of(1, 1, 0, 1));
        inputs.add(Data.of(1, 1, 1, 0));
        inputs.add(Data.of(1, 1, 1, 1));
        input = convertToINDArray(inputs);
        
        labelData = new ArrayList<>();
        labelData.add(Data.of(0));
        labelData.add(Data.of(0));
        labelData.add(Data.of(0));
        labelData.add(Data.of(1));
        labelData.add(Data.of(0));
        labelData.add(Data.of(0));
        labelData.add(Data.of(0));
        labelData.add(Data.of(1));
        labelData.add(Data.of(0));
        labelData.add(Data.of(0));
        labelData.add(Data.of(0));
        labelData.add(Data.of(1));
        labelData.add(Data.of(1));
        labelData.add(Data.of(1));
        labelData.add(Data.of(1));
        labelData.add(Data.of(1));
        labels = convertToINDArray(labelData);
            
        data = new DataSet(input, labels);
        
        INDArray output = network.output(data.getFeatureMatrix());
        System.out.println("Output: " + output);
    }

    private static INDArray convertToINDArray(List<Data> inputs) {
        int size = inputs.get(0).getNumbers().size();
        INDArray inputArray = Nd4j.zeros(inputs.size(), size);
        for (int i = 0; i < inputs.size(); i++) {
            Data input = inputs.get(i);
            if (input.getNumbers().size() != size) {
                throw new RuntimeException("Your data is fucked");
            }
            
            for (int j = 0; j < input.getNumbers().size(); j++) {
                int number = input.getNumbers().get(j);
                inputArray.putScalar(new int[] {i, j}, number);
            }
        }
        return inputArray;
    }
}
