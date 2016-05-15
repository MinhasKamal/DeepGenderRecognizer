/***********************************************************
* Developer: Minhas Kamal (minhaskamal024@gmail.com)       *
* Website: https://github.com/MinhasKamal/Intellectron     *
* License:  GNU General Public License version-3           *
***********************************************************/

package com.minhaskamal.deepGenderRecognizer;

import java.io.File;
import java.io.FileFilter;

import com.minhaskamal.egami.matrix.Matrix;
import com.minhaskamal.egami.matrixUtil.MatrixUtilities;
import com.minhaskamal.intellectron.NeuralNetworkImplementation;

public class DeepGenderRecognizer {
	public static void main(String[] args) throws Exception {
		int matrixHeight = 25;
		int matrixWidth = 25;
		
		System.out.println("OPERATION STARTED!!!");
		/**/
		//prepare data//
		System.out.println("PREPARING DATA...");
		String[][] allFilePaths = dataPaths("src/res/img");
		double[][][] inputs = new double[allFilePaths.length][][];
		for(int i=0; i<allFilePaths.length; i++){
			inputs[i] = new double[allFilePaths[i].length][];
			for(int j=0; j<allFilePaths[i].length; j++){
				Matrix matrix = new Matrix(allFilePaths[i][j], Matrix.BLACK_WHITE);
				matrix = new MatrixUtilities().convertToBinary(matrix, 170);
				int[] rawInputVector = vectorize(matrix);
				inputs[i][j] = scale(rawInputVector, 0, 255);
			}
		}
		
		double[][] outputs = new double[inputs.length][];
		for(int i=0; i<outputs.length; i++){
			outputs[i] = new double[inputs.length];
			outputs[i][i] = 1;
		}
		/**/
		/**/
		//train//
		System.out.println("TRAINING NETWORK...");
		int[] numbersOfNeuronsInLayers = new int[]{20, 5, 2};
		NeuralNetworkImplementation neuralNetworkImplementation = new NeuralNetworkImplementation(numbersOfNeuronsInLayers,
				0.1, matrixHeight*matrixWidth);
		
		int cycle=10;
		for(int c=0; c<cycle; c++){
			for(int j=0; j<1000; j++){
				for(int i=0; i<inputs.length; i++){
					neuralNetworkImplementation.train(inputs[i][j], outputs[i]);
				}
			}
			System.out.println("Cycle- " + c);
		}
		/**/
		/**/
		String workspace = System.getenv("SystemDrive") + System.getenv("HOMEPATH") + "\\Desktop\\";
		neuralNetworkImplementation.dump(workspace+"know.xml");
//		NeuralNetworkImplementation neuralNetworkImplementation = new NeuralNetworkImplementation(workspace+"know.xml");
		
		/**/
		for(int i=0; i<2; i++){
			for(int c=0; c<100; c++){
				double[] out = neuralNetworkImplementation.predict(inputs[i][1000+c]);
				if(out[0]>out[1]){
					System.out.println("female");
				}else{
					System.out.println("male");
				}
			}
			System.out.println("#######");
		}
		/**/
		/*/
		Matrix matrix = createMatrix(neuralNetworkImplementation.generate(new double[]{0.591, 0.51}));
		matrix.write(workspace+"pic.png");
		/**/
	}
	
	public static int[] vectorize(Matrix matrix){
		int height = matrix.getRows();
		int width = matrix.getCols();
		
		int[] vector = new int[height*width];
		
		for(int i=0, k=0; i<height; i++){
			for(int j=0; j<width; j++){
				vector[k] = matrix.pixels[i][j][0];
				k++;
			}
		}
		
		return vector;
	}
	
	public static double[] scale(int[] vector, int minValue, int maxValue){
		double[] scaledVector = new double[vector.length];
		
		for(int i=0; i<vector.length; i++){
			scaledVector[i] = (vector[i]-minValue)/(maxValue-minValue);
		}
		
		return scaledVector;
	}
	
	public static Matrix createMatrix(double[] vector){
		int row = (int) Math.sqrt(vector.length);
		int col = row;
		
		Matrix matrix = new Matrix(row, col, Matrix.BLACK_WHITE);
		
		int k=0;
		for(int i=0; i<row; i++){
			for(int j=0; j<col; j++){
				matrix.pixels[i][j] = new int[]{ (int) (vector[k]*254) };
				k++;
			}
		}
		
		return matrix;
	}
	
	public static String[][] dataPaths(String rootFolderPath){
		File[] directories = new File(rootFolderPath).listFiles(new FileFilter() {
			@Override
			public boolean accept(File arg0) {
				return arg0.isDirectory();
			}
		});
		
		String[][] allFilePaths = new String[directories.length][];
		for(int i=0; i<directories.length; i++){
			File[] files = directories[i].listFiles();
			allFilePaths[i] = new String[files.length];
			
			for(int j=0; j<files.length; j++){
				allFilePaths[i][j] = files[j].getAbsolutePath();
			}
		}
		
		return allFilePaths;
	}
}
