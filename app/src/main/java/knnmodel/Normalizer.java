package knnmodel;

import java.util.Arrays;

public class Normalizer {
    private double[] min, max;

    // Нахождение диапазона и трансформация
    double[][] fitTransform(double[][] X) {
        int m = X[0].length; // количество признаков
        min = new double[m];
        max = new double[m];
        Arrays.fill(min, Double.POSITIVE_INFINITY); // Заполнение минимальными
        Arrays.fill(max, Double.NEGATIVE_INFINITY); // Заполнение макчимальными

        for (double[] row : X) // пребираем значение, и если находится меньше или больше, то заменяется на нвовое
            for (int j = 0; j < m; j++) {
                if (row[j] < min[j]) min[j] = row[j];
                if (row[j] > max[j]) max[j] = row[j];
            }

        return transform(X);
    }

    // Трансфоромация в [0, 1]
    double[][] transform(double[][] X) {
        int n = X.length;
        int m = X[0].length;
        double[][] out = new double[n][m];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++) {
                double range = max[j] - min[j];
                out[i][j] = (range == 0) ? 0.0 : (X[i][j] - min[j]) / range; // Избежание деления на ноль
            }
        return out;
    }
}