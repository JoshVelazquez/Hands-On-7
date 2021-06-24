package examples.backpropagation;
import java.util.Random;

public class Backpropagation {
    private double[][] entradas = { { 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0 },
            { 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0 },
            { 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0 },
            { 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1 },
            { 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1 },
            { 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1 } };
    private double[][] salidas = { { 1, 1 }, { 1, 1 }, { 1, 1 }, { 0, 0 }, { 0, 0 }, { 0, 0 } };
    private double[][] pesos1;
    private double[][] pesos2;
    private double errorDeseado;
    private double tasaDeAprendizaje;
    private double errorCuadratico;
    private boolean convergio;
    private int epocas;
    private int clases;
    private int neuronas;

    public Backpropagation() {
        errorCuadratico = Double.MAX_VALUE;
        convergio = false;
        errorDeseado = 0.5;
        tasaDeAprendizaje = 0.2;
        clases = 2;
        neuronas = 5;
        pesos1 = new double[neuronas][entradas[0].length + 1];
        pesos2 = new double[clases][neuronas + 1];
        epocas = 1000;
    }

    public void entrenar() {
        int epoca = 0;
        inicializarPesos();
        while (epoca < epocas && errorCuadratico > errorDeseado) {
            errorCuadratico = 0;
            double[] error = new double[clases];
            double[] net1 = new double[neuronas];
            double[] net2 = new double[clases];
            double[] salida1 = new double[neuronas];
            double[] salida2 = new double[clases];
            double[] sensibilidad1 = new double[neuronas];
            double[] sensibilidad2 = new double[clases];
            for (int i = 0; i < entradas.length; i++) {
                double[] entrada = entradas[i];
                double[] esperada = salidas[i];

                for (int j = 0; j < neuronas; j++) {
                    net1[j] = pesos1[j][0] + productoPunto(pesos1[j], entrada);
                    salida1[j] = funcionSigmoide(net1[j]);
                }

                for (int j = 0; j < clases; j++) {
                    net2[j] = pesos2[j][0] + productoPunto(pesos2[j], salida1);
                    salida2[j] = funcionSigmoide(net2[j]);
                }

                for (int j = 0; j < clases; j++) {
                    double resultado = esperada[j] - salida2[j];
                    sensibilidad2[j] = derivadaFuncionSigmoide(net2[j] * resultado);
                    error[j] = resultado;
                }

                double[][] transPesos = transpuesta(pesos1);
                for (int j = 0; j < neuronas; j++) {
                    sensibilidad1[j] = (derivadaFuncionSigmoide(net1[j]) * transPesos[0][j] * sensibilidad2[0]
                            + derivadaFuncionSigmoide(net1[j]) * transPesos[1][j] * sensibilidad2[1]);
                }

                // double[][] transEntrada = transpuesta(entrada);
                for (int j = 0; j < neuronas; j++) {
                    for (int k = 0; k < entrada.length; k++) {
                        pesos1[j][k + 1] += (tasaDeAprendizaje * sensibilidad1[j] * entrada[k]);
                    }
                    pesos1[j][0] += (tasaDeAprendizaje * sensibilidad1[j]);
                }

                for (int j = 0; j < clases; j++) {
                    for (int k = 0; k < salida1.length; k++) {
                        pesos2[j][k + 1] += (tasaDeAprendizaje * sensibilidad2[j] * salida1[k]);
                    }
                    pesos2[j][0] += (tasaDeAprendizaje * sensibilidad2[j]);
                }
                for (int j = 0; j < error.length; j++) {
                    errorCuadratico += error[j];
                }
                errorCuadratico = Math.pow(errorCuadratico, 2);

            }
            epoca++;
            errorCuadratico /= entradas.length;
            if (errorCuadratico < errorDeseado) {
                System.out.println("Convergio en la epoca: " + epoca + " error cuadratico: " + errorCuadratico);
                convergio = true;
            }
        }
        if (!convergio) {
            System.out.println("No convergio error cuadratico:" + errorCuadratico);
        }
    }

    private void inicializarPesos() {
        Random rnd = new Random();
        for (int i = 0; i < neuronas; i++) {
            for (int j = 0; j < entradas[0].length + 1; j++) {
                pesos1[i][j] = rnd.nextDouble();
            }
        }
        for (int i = 0; i < clases; i++) {
            for (int j = 0; j < neuronas + 1; j++) {
                pesos2[i][j] = rnd.nextDouble();
            }
        }
    }

    private double productoPunto(double[] pesos, double[] entradas) {
        double producto = 0;
        for (int i = 0; i < entradas.length; i++) {
            producto += entradas[i] * pesos[i];
        }
        return producto;
    }

    private double funcionSigmoide(double valor) {
        return 1 / (1 + Math.exp(-valor));
    }

    private double derivadaFuncionSigmoide(double valor) {
        return funcionSigmoide(valor) * (1 - funcionSigmoide(valor));
    }

    private double[][] transpuesta(double[][] matriz) {
        double[][] trans = new double[matriz[0].length][matriz.length];
        for (int i = 0; i < matriz.length; i++) {
            for (int j = 0; j < matriz[0].length; j++) {
                trans[j][i] = matriz[i][j];
            }
        }
        return trans;
    }
}