package knnmodel;

import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;

public class KNearestNeighbors {
    private int k = 3; // Количество ближайших соседей
    private final double[][] X; // Датасет опроса
    private final String[] y; // Айфон или андроид

    KNearestNeighbors(double[][] X, String[] y) {
        this.X = X;
        this.y = y; 
    }

    void changeK(int k) {
        this.k = k;
    }

    // Предсказание
    public String predict(double[] x) {
        // Подсчёт расстояний
        int n = X.length;
        double[] dist = new double[n];
        for (int i = 0; i < n; i++) 
            dist[i] = euclidean(x, X[i]);

        // Сортировка индексов по расстоянию
        Integer[] idx = new Integer[n];
        for (int i = 0; i < n; i++) 
            idx[i] = i;
        Arrays.sort(idx, Comparator.comparingDouble(i -> dist[i]));

        // Взвешенное голосование: вес = 1/(d + eps)
        Map<String, Double> votes = new HashMap<>(); // Ключём является название устройства, а значением вес
        for (int i = 0; i < Math.min(k, n); i++) {
            String label = y[idx[i]];
            double w = 1.0 / (dist[idx[i]] + 1e-9); // Добавляем одну миллиардную, чтобы не поделить на ноль
            votes.put(label, votes.getOrDefault(label, 0.0) + w); // Если запись в словаре уже есть, добавляется вес к уже 
        }
        return Collections.max(votes.entrySet(), Map.Entry.comparingByValue()).getKey(); // Возвращает название устройства
    }

    // Рачсёт евклидова расстояния
    private double euclidean(double[] a, double[] b) {
        double s = 0.0;
        for (int i = 0; i < a.length; i++) {
                double d = a[i] - b[i];
                s += d * d;
        }
        return Math.sqrt(s);
    }
}