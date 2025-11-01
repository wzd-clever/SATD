package main.methods;

import main.Settings;
import others.FileHandle;
import others.tm.process.DataReader;

import java.io.File;
import java.util.ArrayList;
import java.util.List;


public class Mat extends Method {

    {
        methodPath = rootPath + "mat/";
    }

    // 关键词表
    public static String[] keyWords = {"todo", "hack", "fixme", "xxx", "workaround", "tbd", "dms", "revisit", "notused"};

    public static void main(String[] args) throws Exception {
        Mat mat = new Mat();
        mat.prepareData();
        mat.runMultiRoundExperiments();
    }

    /**
     * 数据准备阶段：
     * 从 exp_data/data/IRx/ 读取原始文件，
     * 提取注释并生成 exp_data/mat/data--项目名.arff 文件
     */
    @Override
    public void prepareData() {
        String baseInputPath = "exp_data/data/IR1/"; // 默认第一轮结构一致
        String outputPath = "exp_data/mat/";

        File outDir = new File(outputPath);
        if (!outDir.exists()) outDir.mkdirs();

        System.out.println("===== 开始 MAT 数据准备 =====");
        for (String project : Settings.projectNames) {
            String inputFile = baseInputPath + "data--" + project + ".txt";
            String outputFile = outputPath + "data--" + project + ".arff";

            if (!new File(inputFile).exists()) {
                System.out.println("[警告] 找不到输入文件：" + inputFile);
                continue;
            }

            List<String> comments = FileHandle.readFileToLines(inputFile);
            List<String> arff = new ArrayList<>();
            arff.add("@relation " + project);
            arff.add("");
            arff.add("@attribute comment string");
            arff.add("@attribute label {positive, negative}");
            arff.add("");
            arff.add("@data");

            for (String line : comments) {
                line = line.replace("\"", "'").replace(",", " ");
                arff.add("'" + line.trim() + "', ?");
            }

            FileHandle.writeLinesToFile(outputFile, arff);
            System.out.println("[生成] " + outputFile);
        }
        System.out.println("===== MAT 数据准备完成 =====");
    }

    /**
     * 主实验流程（20轮循环 + MTO + OTO）
     */
    public void runMultiRoundExperiments() throws Exception {
        String originPath = "exp_data/data/IR";

        for (int I = 1; I <= 20; I++) {
            System.out.println("\n------------------------开始第" + I + "轮 MAT 实验------------------------");
            DataReader.readComments(originPath + I + '/');

            // ========================== MTO ==========================
            System.out.println("--------- 第 " + I + " 轮 MTO 训练 ---------");
            for (String targetProject : Settings.projectNames) {
                List<Double> finalPred = null;
                int dataCount = -1;

                // 多源投票
                for (String sourceProject : Settings.projectNames) {
                    if (sourceProject.equals(targetProject)) continue;

                    // 读取目标项目的注释
                    List<String> instances = FileHandle.readFileToLines("exp_data/mat/data--" + targetProject + ".arff");
                    List<Double> pred = new ArrayList<>();
                    for (int i = 7; i < instances.size(); i++) {
                        String text = instances.get(i).split(",")[0];
                        int result = classify(text, keyWords, true);
                        pred.add(result == 1 ? 1.0 : -1.0);
                    }

                    if (finalPred == null) {
                        dataCount = pred.size();
                        finalPred = new ArrayList<>();
                        for (int k = 0; k < dataCount; k++) finalPred.add(0.0);
                    }

                    // 投票
                    for (int k = 0; k < pred.size(); k++) {
                        finalPred.set(k, finalPred.get(k) + pred.get(k));
                    }
                }

                // 投票结果输出
                List<String> predictionLabels = new ArrayList<>();
                for (double v : finalPred) predictionLabels.add(v > 0 ? "1" : "0");

                String resultPath = "Res/IR" + I + "/MTO_MAT/result--" + targetProject + ".txt";
                FileHandle.writeLinesToFile(resultPath, predictionLabels);
            }

            // ========================== MTO 评估 ==========================
            System.out.println("--------- 第 " + I + " 轮 MTO 评估 ---------");
            StringBuilder text = new StringBuilder("TP, FN, FP, TN, P, R, F1, ER, RI\n");
            for (String projectName : Settings.projectNames) {
                double tp = 0, fn = 0, fp = 0, tn = 0;
                String resultPath = "Res/IR" + I + "/MTO_MAT/result--" + projectName + ".txt";
                String oraclePath = "dataset/label--" + projectName + ".txt";

                List<String> result = FileHandle.readFileToLines(resultPath);
                List<String> oracle = FileHandle.readFileToLines(oraclePath);

                for (int i = 0; i < result.size() && i < oracle.size(); i++) {
                    String label = oracle.get(i).trim();
                    String pred = result.get(i).trim();
                    if (label.equals("positive") && pred.equals("1")) tp++;
                    if (label.equals("positive") && pred.equals("0")) fn++;
                    if (label.equals("negative") && pred.equals("1")) fp++;
                    if (label.equals("negative") && pred.equals("0")) tn++;
                }

                double precision = tp / (tp + fp + 1e-6);
                double recall = tp / (tp + fn + 1e-6);
                double f1 = 2 * precision * recall / (precision + recall + 1e-6);
                double N = tp + tn + fp + fn;
                double ER = (tp * N - (tp + fp) * (tp + fn)) / (tp * N + 1e-6);
                double RI = (tp * N - (tp + fp) * (tp + fn)) / ((tp + fp) * (tp + fn) + 1e-6);

                text.append((int) tp).append(", ")
                        .append((int) fn).append(", ")
                        .append((int) fp).append(", ")
                        .append((int) tn).append(", ")
                        .append(String.format("%.3f", precision)).append(", ")
                        .append(String.format("%.3f", recall)).append(", ")
                        .append(String.format("%.3f", f1)).append(", ")
                        .append(String.format("%.3f", ER)).append(", ")
                        .append(String.format("%.3f", RI)).append("\n");
            }
            FileHandle.writeStringToFile("Res/IR" + I + "/MTO_MAT/Evaluation.csv", text.toString());
            System.out.println(text.toString());

            // ========================== OTO ==========================
            System.out.println("--------- 第 " + I + " 轮 OTO 训练评估 ---------");
            List<Double> P = new ArrayList<>();
            List<Double> R = new ArrayList<>();
            List<Double> F1 = new ArrayList<>();

            for (String testProject : Settings.projectNames) {
                StringBuilder text1 = new StringBuilder("Training project, TP, FN, FP, TN, P, R, F1, ER, RI\n");
                double sumP = 0, sumR = 0, sumF1 = 0;

                for (String trainProject : Settings.projectNames) {
                    if (trainProject.equals(testProject)) continue;

                    String testPath = "exp_data/mat/data--" + testProject + ".arff";
                    List<String> instances = FileHandle.readFileToLines(testPath);

                    // 预测
                    List<String> predictions = new ArrayList<>();
                    for (int i = 7; i < instances.size(); i++) {
                        String textLine = instances.get(i).split(",")[0];
                        int label = classify(textLine, keyWords, true);
                        predictions.add(label == 1 ? "1" : "0");
                    }

                    String outPath = "Rest/IR" + I + "/OTO_MAT/result--" + trainProject + "-" + testProject + ".txt";
                    FileHandle.writeLinesToFile(outPath, predictions);

                    // ===== 计算指标 =====
                    String oraclePath = "dataset/label--" + testProject + ".txt";
                    List<String> oracle = FileHandle.readFileToLines(oraclePath);

                    double tp = 0, fn = 0, fp = 0, tn = 0;
                    for (int i = 0; i < predictions.size() && i < oracle.size(); i++) {
                        String label = oracle.get(i).trim();
                        String pred = predictions.get(i).trim();
                        if (label.equals("positive") && pred.equals("1")) tp++;
                        if (label.equals("positive") && pred.equals("0")) fn++;
                        if (label.equals("negative") && pred.equals("1")) fp++;
                        if (label.equals("negative") && pred.equals("0")) tn++;
                    }

                    double precision = tp / (tp + fp + 1e-6);
                    double recall = tp / (tp + fn + 1e-6);
                    double f1 = 2 * precision * recall / (precision + recall + 1e-6);
                    double N = tp + tn + fp + fn;
                    double ER = (tp * N - (tp + fp) * (tp + fn)) / (tp * N + 1e-6);
                    double RI = (tp * N - (tp + fp) * (tp + fn)) / ((tp + fp) * (tp + fn) + 1e-6);

                    text1.append(trainProject).append(", ")
                            .append((int) tp).append(", ")
                            .append((int) fn).append(", ")
                            .append((int) fp).append(", ")
                            .append((int) tn).append(", ")
                            .append(String.format("%.3f", precision)).append(", ")
                            .append(String.format("%.3f", recall)).append(", ")
                            .append(String.format("%.3f", f1)).append(", ")
                            .append(String.format("%.3f", ER)).append(", ")
                            .append(String.format("%.3f", RI)).append("\n");

                    sumP += precision;
                    sumR += recall;
                    sumF1 += f1;
                }

                int len = Settings.projectNames.length - 1;
                P.add(sumP / len);
                R.add(sumR / len);
                F1.add(sumF1 / len);

                FileHandle.writeStringToFile("Rest/IR" + I + "/OTO_MAT/" + testProject + ".csv", text1.toString());
            }

            // 输出平均结果
            List<String> r = new ArrayList<>();
            for (int i = 0; i < P.size(); i++) {
                System.out.printf("Avg., %.3f, %.3f, %.3f\n", P.get(i), R.get(i), F1.get(i));
                r.add("Avg., " + P.get(i) + ", " + R.get(i) + ", " + F1.get(i));
            }
            FileHandle.writeLinesToFile("Rest/IR" + I + "/OTO_MAT/Evaluation_all.csv", r);
        }
    }

    /**
     * 基于关键字的分类器
     */
    public static int classify(String instance, String[] keyWords, boolean isFuzzy) {
        String[] words = instance.replace("'", "").split(" ");
        if (isFuzzy) {
            for (String word : words) {
                for (String key : keyWords) {
                    if (word.toLowerCase().contains(key)) {
                        if (word.contains("xxx") && !word.equalsIgnoreCase("xxx")) return 0;
                        return 1;
                    }
                }
            }
        } else {
            for (String word : words) {
                for (String key : keyWords)
                    if (word.equalsIgnoreCase(key)) return 1;
            }
        }
        return 0;
    }
}
