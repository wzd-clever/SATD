package main.methods;

import edu.stanford.nlp.classify.ColumnDataClassifier;
import edu.stanford.nlp.ling.Datum;
import edu.stanford.nlp.objectbank.ObjectBank;
import main.Settings;
import others.FileHandle;
import others.tm.process.DataReader;

import java.io.File;
import java.util.ArrayList;
import java.util.List;


public class NLP extends Method {

    {
        methodPath = rootPath + "nlp/";
    }

    public static void main(String[] args) throws Exception {
        String originPath = "exp_data/data/IR";

        // 多轮实验
        for (int I = 1; I <= 20; I++) {
            System.out.println("------------------------开始第" + I + "轮 NLP 实验------------------------");

            // ========================== MTO ==========================
            System.out.println("------------------------第" + I + "轮 MTO 训练------------------------");
            DataReader.readComments(originPath + I + '/');

            for (int target = 0; target < Settings.projectNames.length; target++) {
                String targetProject = Settings.projectNames[target];
                System.out.println("Target: " + targetProject);

                // 初始化预测结果容器
                List<Double> finalPred = new ArrayList<>();
                int dataCount = -1;

                for (int source = 0; source < Settings.projectNames.length; source++) {
                    if (source == target) continue;
                    String sourceProject = Settings.projectNames[source];

                    String trainFile = "exp_data/nlp/data--" + sourceProject + ".txt";
                    String testFile = "exp_data/nlp/data--" + targetProject + ".txt";

                    // 训练分类器
                    ColumnDataClassifier cdc = new ColumnDataClassifier(Settings.rootPath + "dic/cheese2007.prop");
                    cdc.trainClassifier(trainFile);

                    // 对目标项目进行预测
                    List<Double> pred = new ArrayList<>();
                    for (String line : ObjectBank.getLineIterator(testFile, "utf-8")) {
                        Datum<String, String> d = cdc.makeDatumFromLine(line);
                        if (cdc.classOf(d).equals("WITHOUT_CLASSIFICATION"))
                            pred.add(0.0);
                        else
                            pred.add(1.0);
                    }

                    // 初始化 finalPred 长度
                    if (dataCount == -1) {
                        dataCount = pred.size();
                        for (int k = 0; k < dataCount; k++) finalPred.add(0.0);
                    }

                    // 投票加权（简单求和）
                    for (int k = 0; k < pred.size(); k++) {
                        finalPred.set(k, finalPred.get(k) + (pred.get(k) == 1.0 ? 1.0 : -1.0));
                    }
                }

                // 多源结果平均
                List<String> predictionLabels = new ArrayList<>();
                for (double v : finalPred) predictionLabels.add(v > 0 ? "1" : "0");

                String resultPath = "Res/IR" + I + "/MTO_NLP/result--" + targetProject + ".txt";
                FileHandle.writeLinesToFile(resultPath, predictionLabels);
            }

            // ========================== MTO 评估 ==========================
            System.out.println("------------------------第" + I + "轮 MTO 评估------------------------");
            StringBuilder text = new StringBuilder("TP, FN, FP, TN, P, R, F1, ER, RI\n");
            for (String projectName : Settings.projectNames) {
                double tp = 0, fn = 0, fp = 0, tn = 0;
                String resultPath = "Res/IR" + I + "/MTO_NLP/result--" + projectName + ".txt";
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

            FileHandle.writeStringToFile("Res/IR" + I + "/MTO_NLP/Evaluation.csv", text.toString());
            System.out.println(text.toString());

            // ========================== OTO ==========================
            System.out.println("------------------------第" + I + "轮 OTO 训练评估------------------------");
            List<Double> P = new ArrayList<>();
            List<Double> R = new ArrayList<>();
            List<Double> F1 = new ArrayList<>();

            for (int test = 0; test < Settings.projectNames.length; test++) {
                String testProject = Settings.projectNames[test];
                String testFile = "exp_data/nlp/data--" + testProject + ".txt";

                StringBuilder text1 = new StringBuilder("Training project, TP, FN, FP, TN, P, R, F1, ER, RI\n");
                double sumP = 0, sumR = 0, sumF1 = 0;

                for (int train = 0; train < Settings.projectNames.length; train++) {
                    if (train == test) continue;
                    String trainProject = Settings.projectNames[train];
                    String trainFile = "exp_data/nlp/data--" + trainProject + ".txt";
                    String outPath = "Rest/IR" + I + "/OTO_NLP/result--" + trainProject + "-" + testProject + ".txt";

                    ColumnDataClassifier cdc = new ColumnDataClassifier(Settings.rootPath + "dic/cheese2007.prop");
                    cdc.trainClassifier(trainFile);

                    List<String> predictions = new ArrayList<>();
                    for (String line : ObjectBank.getLineIterator(testFile, "utf-8")) {
                        Datum<String, String> d = cdc.makeDatumFromLine(line);
                        if (cdc.classOf(d).equals("WITHOUT_CLASSIFICATION"))
                            predictions.add("0");
                        else
                            predictions.add("1");
                    }
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

                FileHandle.writeStringToFile("Rest/IR" + I + "/OTO_NLP/" + testProject + ".csv", text1.toString());
            }

            // 输出平均结果
            List<String> r = new ArrayList<>();
            for (int i = 0; i < P.size(); i++) {
                System.out.printf("Avg., %.3f, %.3f, %.3f\n", P.get(i), R.get(i), F1.get(i));
                r.add("Avg., " + P.get(i) + ", " + R.get(i) + ", " + F1.get(i));
            }
            FileHandle.writeLinesToFile("Rest/IR" + I + "/OTO_NLP/Evaluation_all.csv", r);
        }
    }
}
