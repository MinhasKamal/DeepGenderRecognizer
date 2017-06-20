/***********************************************************
* Developer: Minhas Kamal (minhaskamal024@gmail.com)       *
* Website: https://github.com/MinhasKamal/Intellectron     *
* License:  GNU General Public License version-3           *
***********************************************************/

package com.minhaskamal.deepGenderRecognizer;

import com.minhaskamal.egami.matrix.Matrix;
import com.minhaskamal.egami.matrix.MatrixUtilities;
import com.minhaskamal.intellectron.DeepNeuralNetworkImplementation;
import com.minhaskamal.intellectron.dataPrepare.DataPreparer;

public class DeepGenderRecognizer {
	public static void main(String[] args) throws Exception {
		System.out.println("OPERATION STARTED!!!");
		String workspace = System.getenv("SystemDrive") + System.getenv("HOMEPATH") + "\\Desktop\\";
		
		//prepare data//
		System.out.println("PREPARING DATA...");
		String rootImagePath = "src/res/img";
		DataPreparer dataPreparer = new DataPreparer(rootImagePath, 1000, 1) {
			@Override
			public double[] readFileVectorizeAndScale(String imageFilePath) {
				Matrix matrix = new Matrix(imageFilePath, Matrix.BLACK_WHITE);
				int[][] vector2D = MatrixUtilities.vectorize(matrix);
				int[] vector = new int[vector2D.length];
				for(int i=0; i<vector2D.length; i++){
					vector[i] = vector2D[i][0];
				}
				return this.scale(vector, Matrix.MIN_PIXEL, Matrix.MAX_PIXEL);
			}
		};
		double[][] trainingInputs = dataPreparer.getTrainingInputs();
		double[][] trainingOutputs = dataPreparer.getTrainingOutputs();
		double[][] testingInputs = dataPreparer.getTestingInputs();
		double[][] testingOutputs = dataPreparer.getTestingOutputs();
		
		
		/**/
		//create//
		System.out.println("CREATING NETWORK...");
		int matrixHeight = 25;
		int matrixWidth = 25;
		int[] numbersOfNeuronsInLayers = new int[]{75, 25, 5, 2};
		DeepNeuralNetworkImplementation neuralNetworkImplementation = new DeepNeuralNetworkImplementation(
				numbersOfNeuronsInLayers, 0.01, matrixHeight*matrixWidth);
		/**/
		
		/*/
		//load//
		System.out.println("LOADING...");
		DeepNeuralNetworkImplementation neuralNetworkImplementation = new DeepNeuralNetworkImplementation(
				workspace+"knowledge2\\knowledge.xml");
		/**/
		
		/**/
		//train//
		System.out.println("TRAINING NETWORK...");
		neuralNetworkImplementation = train(neuralNetworkImplementation, trainingInputs, trainingOutputs,
				workspace+"knowledge", 150, 0, 5);
		/**/
		
		/*/
		//store//
		System.out.println("STORING KNOWLEDGE...");
		neuralNetworkImplementation.dump(workspace+"knowledge.xml");
		/**/
		
		/*/
		//predict//
		System.out.println("PREDICTING...");
		predict(neuralNetworkImplementation, testingInputs);
		/**/
		
		/**/
		//test//
		System.out.println("TESTING...");
		test(neuralNetworkImplementation, testingInputs, testingOutputs);
		/**/
		
		/*/
		//generate//
		System.out.println("GENERATING...");
		Matrix matrix = createMatrix(neuralNetworkImplementation.generate(new double[]{0.591, 0.51}));
		matrix.write(workspace+"pic.png");
		/**/
	}
	
	//////////////////////////////////////////////////////////////////////////////
	
	public static DeepNeuralNetworkImplementation train(
			DeepNeuralNetworkImplementation neuralNetworkImplementation,
			double[][] inputs, double[][] expectedOutputs, String filePath,
			int numberOfCycles, int startFrom, int knowledgeStoringDelay){
		
		
		for(int c=1; c<=numberOfCycles; c++){
			//show improvement//
			System.out.println("*Epoch- " + c);
			
			//train//
			//neuralNetworkImplementation.train(inputs, expectedOutputs);
			neuralNetworkImplementation.train2(inputs, expectedOutputs);
			
			//validate//
			test(neuralNetworkImplementation, inputs, expectedOutputs);
			
			//store//
			if(c%knowledgeStoringDelay==0){
				neuralNetworkImplementation.dump(filePath+(c+startFrom)+".xml");
			}
		}
		
		return neuralNetworkImplementation;
	}
	
	public static void predict(
			DeepNeuralNetworkImplementation neuralNetworkImplementation,
			double[][] inputs){
		
		for(int i=0; i<2; i++){
			if(i==0){
				System.out.println("\n\n##Female##");
			}else{
				System.out.println("\n\n##Male##");
			}
			for(int c=0; c<100; c++){
				double[] out = neuralNetworkImplementation.predict(inputs[i]);
				if(out[0]>out[1]){
					System.out.println("female");
				}else{
					System.out.println("male");
				}
			}
		}
	}
	
	public static void test(DeepNeuralNetworkImplementation neuralNetworkImplementation,
			double[][] inputs, double[][] expectedOutputs){
		
		double accuracy = neuralNetworkImplementation.test(inputs, expectedOutputs, 0.3);
		
		int testDataNumber = 0;
		for(int i=0; i<expectedOutputs.length; i++){
			testDataNumber += expectedOutputs[i].length;
		}
		
		System.out.println("Test Data: "+testDataNumber+"; Accuracy: "+accuracy);
	}


}
