package main.methods;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import main.Settings;
import main.Statistics;
import others.FileHandle;
import others.tm.process.DataReader;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.stemmers.SnowballStemmer;
import weka.core.stopwords.WordsFromFile;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.StringToWordVector;
import others.tm.model.EnsembleLearner;

public class TM extends Method {

    {
        methodPath = rootPath + "tm/";
    }

    public static void main(String args[]) throws Exception {
        String originPath = "exp_data/data/IR";
        for (int I = 1; I <=20; I++) {
            System.out.println("------------------------开始第"+I+"轮训练------------------------");
            System.out.println("------------------------准备第"+I+"轮训练数据------------------------");
            DataReader.readComments(originPath+I+'/');
            WordsFromFile stopWords = new WordsFromFile();
            stopWords.setStopwords(new File("exp_data" + "/dic/stopwords.txt")); // 停用词列表

            StringToWordVector stw = new StringToWordVector(100000);
            stw.setOutputWordCounts(true); //设置记录单词在文档中出现的次数（词频变量）若使用TFIDF公式，该选项必须设置为true
            stw.setTFTransform(true);      //执行TF转换 TF(t,d)=log(f(t,d)+1)
            stw.setIDFTransform(true);     //执行IDF(t,D)=log(|D|/|{d in D, t in d}|)
            stw.setStemmer(new SnowballStemmer());
            stw.setStopwordsHandler(stopWords);
            for (String project : projects) {
                String filePath = "exp_data/tm/" + "data--" + project + ".arff";
                System.out.println(filePath);
                DataReader.outputArffData(DataReader.selectProject(project), filePath);
            }
//-------------------------------------------------------------------------------------------
//     new TM().predict();


            System.out.println("------------------------开始第"+I+"轮MTO训练------------------------");
            DataReader.readComments(originPath+I+'/');

            WordsFromFile StopWords = new WordsFromFile();
            StopWords.setStopwords(new File(Settings.rootPath + "/dic/stopwords.txt"));

            StringToWordVector stw1 = new StringToWordVector(100000);
            stw1.setOutputWordCounts(true);
            stw1.setTFTransform(true);      // TF(t,d)=log(f(t,d)+1)
            stw1.setIDFTransform(true);     // IDF(t,D)=log(|D|/|{d in D, t in d}|)
            stw1.setStemmer(new SnowballStemmer());
            stw1.setStopwordsHandler(StopWords);

            String trainDataPath, testDataPath;
            double ratio = 0.1;
            // Processing each test project
            for (int target = 0; target < Settings.projectNames.length; target++) {
                System.out.print("Target: " + Settings.projectNames[target] + ", ");
                testDataPath = "data/tm/"  + "data--" + Settings.projectNames[target] + ".arff";

                EnsembleLearner eLearner = new EnsembleLearner();
                // Processing each training project
                for (int source = 0; source < Settings.projectNames.length; source++) {
                    trainDataPath = "exp_data/tm/"  + "data--" + Settings.projectNames[source] + ".arff";
                    if (source == target) continue;

                    if (eLearner.getTestData() == null) {
                        System.out.println(testDataPath);
                        Instances tmp = DataSource.read(testDataPath);
                        tmp.setClassIndex(1);
                        eLearner = new EnsembleLearner(tmp);
                    }
                    Instances trainSet = DataSource.read(trainDataPath);
                    Instances testSet = DataSource.read(testDataPath);
                    stw1.setInputFormat(trainSet);
                    trainSet = Filter.useFilter(trainSet, stw1);
                    testSet = Filter.useFilter(testSet, stw1);
                    trainSet.setClassIndex(0);
                    testSet.setClassIndex(0);

                    // feature selection IG
                    AttributeSelection attSelection = new AttributeSelection();
                    Ranker ranker = new Ranker();
                    ranker.setNumToSelect((int) (trainSet.numAttributes() * ratio)); // the selection ratio
                    InfoGainAttributeEval ifg = new InfoGainAttributeEval();
                    attSelection.setEvaluator(ifg);
                    attSelection.setSearch(ranker);
                    attSelection.setInputFormat(trainSet);
                    trainSet = Filter.useFilter(trainSet, attSelection);
                    testSet = Filter.useFilter(testSet, attSelection);

                    // NBM classifier
                    Classifier classifier = new NaiveBayesMultinomial();
                    classifier.buildClassifier(trainSet);

                    for (int i = 0; i < testSet.numInstances(); i++) {
                        Instance instance = testSet.instance(i);
                        double score;
                        if (classifier.classifyInstance(instance) == 1.0) score = 1;
                        else score = -1;
                        eLearner.vote(i, score);
                    }
                }

                double[] predictionLabels = eLearner.evaluate();
                String resultPath = "Res/IR" + I + "/MTO_TM/result--" + Settings.projectNames[target] + ".txt";
                FileHandle.writeDoubleArrayToFile(resultPath, predictionLabels);

            } // end for test project
            //Statistics.evaluate("TM");

            System.out.println("------------------------开始第"+I+"轮MTO评估------------------------");
            StringBuilder text = new StringBuilder("TP, FN, FP, TN, P    , R    , F1   , ER   , RI\n");
            // 处理每个项目的结果
            for (String projectName : Settings.projectNames) {
                double tp = .0, fp = .0, tn = .0, fn = .0;
                String resultPath ="Res/IR" + I + "/MTO_TM/result--" + projectName+ ".txt";
                String oraclePath ="dataset"+ "/label--" + projectName + ".txt";
                List<String> result = FileHandle.readFileToLines(resultPath);
                List<String> oracle = FileHandle.readFileToLines(oraclePath);
                for (int i = 1; i < result.size(); i++) {
                    String label = oracle.get(i).trim(), prediction = result.get(i).trim();
                    if (label.equals("positive") && prediction.equals("1")) tp++;
                    if (label.equals("positive") && prediction.equals("0")) fn++;
                    if (label.equals("negative") && prediction.equals("1")) fp++;
                    if (label.equals("negative") && prediction.equals("0")) tn++;
                }
                // 准确度指标
                double precision = tp / (tp + fp);
                double recall = tp / (tp + fn);
                double f1 = 2 * precision * recall / (precision + recall);
                // 工作量感知指标
                double x = tp + fp, y = tp, n = tp + fn, N = tp + fp + fn + tn;
                double ER = (y * N - x * n) / (y * N);
                double RI = (y * N - x * n) / (x * n);

                text.append((int) tp).append(", ").append((int) fn).append(", ").append((int) fp).append(", ").append((int) tn).append(", ");

                text.append(String.format("%.3f", precision)).append(", ")
                        .append(String.format("%.3f", recall)).append(", ")
                        .append(String.format("%.3f", f1)).append(", ")
                        .append(String.format("%.3f", ER)).append(", ")
                        .append(String.format("%.3f", RI)).append("\n");
            }
            System.out.println(text.toString());
            FileHandle.writeStringToFile("Res/IR" + I + "/MTO_TM"+ "/Evaluation.csv", text.toString());
//        new TM().predictWithLimitedTrainingSet();

            System.out.println("------------------------开始第"+I+"轮OTO训练评估------------------------");
            List<Double> P = new ArrayList<>();
            List<Double> R = new ArrayList<>();
            List<Double> F1 = new ArrayList<>();

            DataReader.readComments(originPath+I+'/');  //读取注释数据

// 将（训练集和测试集）中的字符串转换为词向量
            WordsFromFile stopWords1 = new WordsFromFile();
            stopWords1.setStopwords(new File(Settings.rootPath + "/dic/stopwords.txt")); // 停用词列表

            StringToWordVector stw2 = new StringToWordVector(100000);
            stw2.setOutputWordCounts(true);
            stw2.setTFTransform(true);
            stw2.setIDFTransform(true);
            stw2.setStemmer(new SnowballStemmer());
            stw2.setStopwordsHandler(stopWords1);

// 每个测试项目
            for (int test = 0; test < Settings.projectNames.length; test++) {
                System.out.println("Target: " + Settings.projectNames[test] + ", ");
                String testDataPath1 = "data/tm/data--" + Settings.projectNames[test] + ".arff";

                StringBuilder text1 = new StringBuilder("Training project, TP, FN, FP, TN, P, R, F1, ER, RI\n");
                double sumP = 0.0, sumR = 0.0, sumF1 = 0.0;

                // 每个训练项目
                for (int train = 0; train < Settings.projectNames.length; train++) {
                    String trainDataPath1 = "exp_data/tm/data--" + Settings.projectNames[train] + ".arff";

                    // 集成学习器
                    Instances tmp = DataSource.read(testDataPath1);
                    tmp.setClassIndex(1);
                    EnsembleLearner eLearner = new EnsembleLearner(tmp);

                    Instances trainSet = DataSource.read(trainDataPath1);
                    Instances testSet = DataSource.read(testDataPath1);
                    stw2.setInputFormat(trainSet);
                    trainSet = Filter.useFilter(trainSet, stw2);
                    testSet = Filter.useFilter(testSet, stw2);
                    trainSet.setClassIndex(0);
                    testSet.setClassIndex(0);

                    // 特征选择 IG
                    double ratio1 = 0.1;
                    AttributeSelection attSelection = new AttributeSelection();
                    Ranker ranker = new Ranker();
                    ranker.setNumToSelect((int) (trainSet.numAttributes() * ratio1));
                    InfoGainAttributeEval ifg = new InfoGainAttributeEval();
                    attSelection.setEvaluator(ifg);
                    attSelection.setSearch(ranker);
                    attSelection.setInputFormat(trainSet);
                    trainSet = Filter.useFilter(trainSet, attSelection);
                    testSet = Filter.useFilter(testSet, attSelection);

                    // NBM分类器
                    Classifier classifier = new NaiveBayesMultinomial();
                    classifier.buildClassifier(trainSet);

                    for (int i = 0; i < testSet.numInstances(); i++) {
                        Instance instance = testSet.instance(i);
                        double score;
                        if (classifier.classifyInstance(instance) == 1.0) score = 1;
                        else score = -1;
                        eLearner.vote(i, score);
                    }

                    double[] predictionLabels = eLearner.evaluate();
                    String outPath ="Rest/IR" + I + "/OTO_TM/result--" + Settings.projectNames[train] + "-" + Settings.projectNames[test] + ".txt";
                    FileHandle.writeDoubleArrayToFile(outPath, predictionLabels);

                    // ===== 新增：计算混淆矩阵 =====
                    String oraclePath = "dataset/label--" + Settings.projectNames[test] + ".txt";
                    List<String> oracle = FileHandle.readFileToLines(oraclePath);
                    List<String> result = FileHandle.readFileToLines(outPath);

                    double tp = 0, fn = 0, fp = 0, tn = 0;
                    for (int i = 1; i < result.size(); i++) {
                        String label = oracle.get(i).trim();
                        String prediction = result.get(i).trim();
                        if (label.equals("positive") && prediction.equals("1")) tp++;
                        if (label.equals("positive") && prediction.equals("0")) fn++;
                        if (label.equals("negative") && prediction.equals("1")) fp++;
                        if (label.equals("negative") && prediction.equals("0")) tn++;
                    }

                    // 指标
                    double precision = tp / (tp + fp + 1e-6);
                    double recall = tp / (tp + fn + 1e-6);
                    double f1 = 2 * precision * recall / (precision + recall + 1e-6);
                    double N = tp + tn + fp + fn;
                    double ER = (tp * N - (tp + fp) * (tp + fn)) / (tp * N + 1e-6);
                    double RI = (tp * N - (tp + fp) * (tp + fn)) / ((tp + fp) * (tp + fn) + 1e-6);

                    // 写入结果行
                    text1.append(Settings.projectNames[train]).append(", ")
                            .append((int) tp).append(", ")
                            .append((int) fn).append(", ")
                            .append((int) fp).append(", ")
                            .append((int) tn).append(", ")
                            .append(String.format("%.3f", precision)).append(", ")
                            .append(String.format("%.3f", recall)).append(", ")
                            .append(String.format("%.3f", f1)).append(", ")
                            .append(String.format("%.3f", ER)).append(", ")
                            .append(String.format("%.3f", RI)).append("\n");

                    // 累加平均
                    sumP += precision;
                    sumR += recall;
                    sumF1 += f1;
                } //end for training project

                int len = projects.length - 1;
                P.add(sumP / len);
                R.add(sumR / len);
                F1.add(sumF1 / len);

                // 写每个测试项目结果
                FileHandle.writeStringToFile("Rest/IR" + I  + "/OTO_TM/" + Settings.projectNames[test] + ".csv", text1.toString());
            } // end for test project

// print 平均值结果
            List<String> r = new ArrayList<>();
            for (int i = 0; i < projects.length; i++) {
                System.out.printf("Avg., %.3f, %.3f, %.3f\n", P.get(i), R.get(i), F1.get(i));
                r.add("Avg., " + P.get(i) + ", " + R.get(i) + ", " + F1.get(i));
            }
            FileHandle.writeLinesToFile("Rest/IR" + I + "/OTO_TM/Evaluation_all.csv", r);

        }



}}
