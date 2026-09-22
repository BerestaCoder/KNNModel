package knnmodel;

import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import com.opencsv.CSVReader;
import com.opencsv.exceptions.CsvValidationException;

public class App {
    private static final int TARGET_COL = 22;
    private static final int TOTAL_COLS = 23;
    private static final int RANDOM_SEED = 3;
    private static final Set<Integer> NUMERIC_COLS = new HashSet<>(Arrays.asList(7, 8, 9, 13, 18, 21));
    //Словарь для строковых значений
    public static void main(String[] args) throws Exception {
        List<String[]> raw = readCSV("input.csv");
        System.out.println("Загружено строк: " + raw.size());
        
        // Разделение на учебный и тестовый списки
        Collections.shuffle(raw, new Random(RANDOM_SEED));
        int split = (int) (raw.size() * 0.8);
        List<String[]> trainRaw = raw.subList(0, split);
        List<String[]> testRaw  = raw.subList(split, raw.size());

        // Создание словаря
        Map<Integer, Map<String, Integer>> vocab = buildVocabulary(trainRaw);
        
        // Преобразование в числовые параметры
        double[][] XTrain = new double[trainRaw.size()][];
        String[]   YTrain = new String[trainRaw.size()];
        for (int i = 0; i < trainRaw.size(); i++) {
            XTrain[i] = featurize(trainRaw.get(i), vocab);
            YTrain[i] = trainRaw.get(i)[TARGET_COL].trim();
        }

        double[][] XTest = new double[testRaw.size()][];
        String[]   YTest = new String[testRaw.size()];
        for (int i = 0; i < testRaw.size(); i++) {
            XTest[i] = featurize(testRaw.get(i), vocab);
            YTest[i] = testRaw.get(i)[TARGET_COL].trim();
        }

        // Нормализация признаков
        Normalizer norm = new Normalizer();
        XTrain = norm.fitTransform(XTrain);
        XTest = norm.transform(XTest);

        // Обучение модлели
        KNearestNeighbors knn = new KNearestNeighbors(XTrain, YTrain);

        // Тестирование
        System.out.println("Тестирование");
        int[] ks = {1, 3, 5, 7, 9, 11, 13, 15};
        for (int k : ks) {
            test(k, knn, XTest, YTest);
        }
        
    }

    // Создаём список из содержимого файла/датасета
    private static List<String[]> readCSV(String path) throws IOException, CsvValidationException{
        List<String[]> rows = new ArrayList<>();
        try (CSVReader reader = new CSVReader(new FileReader(path))) {
            String[] line;
            reader.readNext(); // Пропускаем заголовки в датасете
            while ((line = reader.readNext()) != null) {
                rows.add(line);
            }
        }
        return rows;
    }

    private static Map<Integer, Map<String, Integer>> buildVocabulary(List<String[]> rows) {
        Map<Integer, Map<String, Integer>> vocab = new HashMap<>();

        // начинаем с col = 1, потому что col = 0 (дата) игнорируется
        for (String[] row : rows) {
            for (int col = 1; col < TOTAL_COLS; col++) {
                if (col == TARGET_COL) continue;
                if (NUMERIC_COLS.contains(col)) continue;

                String[] values = row[col].trim().split(";"); // мульти-значения типа "a;b"
                vocab.computeIfAbsent(col, k -> new LinkedHashMap<>());
                for (String v : values) {
                    String key = v.trim().toLowerCase();
                    if (key.isEmpty()) continue;
                    vocab.get(col).putIfAbsent(key, vocab.get(col).size());
                }
            }
        }
        return vocab;
    }

    private static double[] featurize(String[] row, Map<Integer, Map<String, Integer>> vocab) {
        List<Double> feats = new ArrayList<>();

        // col = 0 (дата) пропускается
        for (int col = 1; col < TOTAL_COLS; col++) {
            if (col == TARGET_COL) continue;
            String val = row[col].trim();

            if (NUMERIC_COLS.contains(col)) {
                feats.add(parseDouble(val));
            } else {
                Map<String, Integer> map = vocab.getOrDefault(col, Collections.emptyMap());
                int size = map.size();
                double[] oh = new double[size];
                for (String v : val.split(";")) {
                    Integer idx = map.get(v.trim().toLowerCase());
                    if (idx != null) oh[idx] = 1.0;
                }
                for (double v : oh) feats.add(v);
            }
        }

        double[] arr = new double[feats.size()];
        for (int i = 0; i < arr.length; i++) arr[i] = feats.get(i);
        return arr;
    }

    private static double parseDouble(String s) {
        try { return Double.parseDouble(s.trim()); }
        catch (Exception e) { return 0.0; }
    }

    // Тестирование
    private static void test(int k, KNearestNeighbors knn, double[][] XTest, String[] YTest) {
        knn.changeK(k);
        int correct = 0;

        System.out.println();
        System.out.printf("k = %d%n", k);
        System.out.printf("%-6s | %-8s | %-15s | %-15s%n", "№", "Верно", "Предсказано", "Ожидалось");
        System.out.println("-".repeat(54));

        for (int i = 0; i < XTest.length; i++) {
            String pred = knn.predict(XTest[i]);
            boolean flag = pred.equals(YTest[i]);
            System.out.printf("%-6d | %-8s | %-15s | %-15s%n",
                i,
                flag ? "+" : "",
                pred,
                YTest[i]);
            if (flag) correct++;
        }

        System.out.println("-".repeat(54));
        System.out.printf("Точность: %.2f%%  (%d / %d)%n", 100.0 * correct / XTest.length, correct, XTest.length);
    }

    // Тестирование модели не на основе датасета
    // private static void testKNN() {
    //     double[][] X = {
    //             {0.0, 0.0}, {0.1, 0.1}, {0.2, 0.0},   // класс A (лево-низ)
    //             {5.0, 5.0}, {5.1, 5.1}, {5.0, 5.2}    // класс B (право-верх)
    //     };
    //     String[] Y = {"A", "A", "A", "B", "B", "B"};

    //     KNearestNeighbors knn = new KNearestNeighbors(3, X, Y);
    //     knn.fit(X, Y);

    //     System.out.println("A : " + knn.predict(new double[]{0.15, 0.05}));
    //     System.out.println("B : " + knn.predict(new double[]{5.05, 5.05}));
    //     System.out.println("A : " + knn.predict(new double[]{0.0, 0.0}));
    // }
}

