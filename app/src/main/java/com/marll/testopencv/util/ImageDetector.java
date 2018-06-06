/**
 * Image Detector 2018
 * This project is a work for appBelchior.
 *
 * @author marllons
 * @version 0.1 04/2018
 */

package com.marll.testopencv.util;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.features2d.DMatch;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.features2d.KeyPoint;

import android.util.Log;

public class ImageDetector {
    // Parameters for matching
    public static final double RATIO_TEST_RATIO = 0.89; //
    public static final int RATIO_TEST_MIN_NUM_MATCHES = 49;
    private static final String TAG = "Debug Detection";
    //Basicos
    private Mat frame;
    private double minDist;
    private List<Mat> trainImages;
    private ArrayList<String> objectNames;
    private List<Point> usedKP;
    private List<KeyPoint> kp, kp2;
    private boolean[] flags;
    private int numMatches;
    private int matchIndex;
    private int[] numMatchesInImage;
    //Personagens
    private int good;
    //Keypoints
    private MatOfKeyPoint frameKeypoints;
    private MatOfKeyPoint testeKeypoints;
    private ArrayList<MatOfKeyPoint> trainKeypoints;
    //Armazedores de Descricao
    private Mat frameDescriptors;
    private List<Mat> trainDescriptors;
    private Mat previousDescriptors;
    private List<MatOfDMatch> atualMatch = new ArrayList<MatOfDMatch>();
    private MatchingStrategy matchingStrategy = MatchingStrategy.RATIO_TEST;
    //Feature Detector
    private FeatureDetector detector;
    //Colors
    //Extrator de Descricao
    private DescriptorExtractor descriptor;
    //Descriptors Matcher
    private DescriptorMatcher matcher;
    //

    /**
     * Constroi o objeto para previamente gerar os descritores e os keypoints das imagens base.
     * Evitando o delay na deteccao em tempo real do frame da camera.
     * @param trainImages Recebe a lista de imagens.
     */
    public ImageDetector(List<Mat> trainImages, ArrayList<String> objectNames) {
        this.trainImages = trainImages;
        this.objectNames = objectNames;

        frameKeypoints = new MatOfKeyPoint();
        frameDescriptors = new Mat();

        usedKP = new ArrayList<Point>();

        good = 0;

        detector = FeatureDetector.create(FeatureDetector.FAST);
        descriptor = DescriptorExtractor.create(FeatureDetector.SURF);
        matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE);

        trainKeypoints = new ArrayList<MatOfKeyPoint>();
        trainDescriptors = new ArrayList<Mat>();

        //computeImages();
    }

    /**
     * Extrai keypoints e Descritores das imagens.
     * Para recuperar os Keypoints chame getKeypoints.
     * Para recuperar os Descritores chame getDescriptors.
     */
    public void computeImages() {
        for (int c = 0; c < trainImages.size(); c++) {
            trainKeypoints.add(new MatOfKeyPoint());
            detector.detect(trainImages.get(c), trainKeypoints.get(c));
            trainDescriptors.add(new Mat());
            descriptor.compute(trainImages.get(c), trainKeypoints.get(c), trainDescriptors.get(c));
        }
        matcher.add(trainDescriptors);
        matcher.train();
    }

    public String recognize(Mat mGray) {
        MatOfKeyPoint keypoints = new MatOfKeyPoint();
        Mat descriptors = new Mat();
        List<MatOfDMatch> matches = new LinkedList<MatOfDMatch>();
        detector.detect(mGray, keypoints);
        descriptor.compute(mGray, keypoints, descriptors);
        kp = keypoints.toList();

        return match(keypoints, descriptors, matches, matchingStrategy);
    }

    public String match(MatOfKeyPoint keypoints, Mat descriptors,
                        List<MatOfDMatch> matches, MatchingStrategy matchingStrategy) {
        return match_ratioTest(descriptors, matches, RATIO_TEST_RATIO,
                RATIO_TEST_MIN_NUM_MATCHES);

    }

    private String match_ratioTest(Mat descriptors, List<MatOfDMatch> matches,
                                   double ratio, int minNumMatches) {
        getMatches_ratioTest(descriptors, matches, ratio);
        return getDetectedObjIndex(matches, minNumMatches);
    }

    // adds to the matches list matches that satisfy the ratio test with ratio
    // ratio
    private void getMatches_ratioTest(Mat descriptors,
                                      List<MatOfDMatch> matches, double ratio) {
        LinkedList<MatOfDMatch> knnMatches = new LinkedList<MatOfDMatch>();
        DMatch bestMatch, secondBestMatch;
        matcher.knnMatch(descriptors, knnMatches, 2);
        for (MatOfDMatch matOfDMatch : knnMatches) {
            bestMatch = matOfDMatch.toArray()[0];
            secondBestMatch = matOfDMatch.toArray()[1];
            if (bestMatch.distance / secondBestMatch.distance < ratio) { //
                MatOfDMatch goodMatch = new MatOfDMatch();
                goodMatch.fromArray(new DMatch[]{bestMatch});
                matches.add(goodMatch);
            }
        }
    }

    // uses the list of matches to count the number of matches to each database
    // object. The object with the maximum such number nmax is considered to
    // have been recognized if nmax > minNumMatches.
    // if for a query descriptor there exists multiple matches to train
    // descriptors of the same train image, all such matches are counted as only
    // one match.
    // returns the name of the object detected, or "-" if no object is detected.
    private String getDetectedObjIndex(List<MatOfDMatch> matches,
                                       int minNumMatches) {
        numMatchesInImage = new int[trainImages.size()];
        matchIndex = -1;
        numMatches = 0;

        for (MatOfDMatch matOfDMatch : matches) {
            DMatch[] dMatch = matOfDMatch.toArray();
            boolean[] imagesMatched = new boolean[trainImages.size()];
            for (int i = 0; i < dMatch.length; i++) {
                if (!imagesMatched[dMatch[i].imgIdx]) {
                    numMatchesInImage[dMatch[i].imgIdx]++;
                    imagesMatched[dMatch[i].imgIdx] = true;
                }
                Core.circle(frame, kp.get(dMatch[0].queryIdx).pt, 6, new Scalar(255, 0, 255));
            }
        }

        for (int i = 0; i < numMatchesInImage.length; i++) {
            if (numMatchesInImage[i] > numMatches) {
                matchIndex = i;
                numMatches = numMatchesInImage[i];
            }
        }

        if (numMatches < minNumMatches) { //
            return " -  -> " + numMatches + "%";
        } else {
            return objectNames.get(matchIndex) + " -> " + numMatches + "%";

        }


    }

    /**
     * Usa o descritor recebido e compara com o descritor do frame usando K-Nearest Neighbors (K-Vizinhos).
     * @param PreviousDescriptors Recebe descritor de uma imagem para compara��o.
     * @param name Recebe nome a ser mostrado quando o objeto for detectado.
     */
    public boolean Process(Mat PreviousDescriptors, String name) {
        //MatOfDMatch matches = new MatOfDMatch();
        //Process Camera Frame
        previousDescriptors = PreviousDescriptors;
        matcher.knnMatch(frameDescriptors, previousDescriptors, atualMatch, 2);
        //Matcher.match(FrameDescriptors, previousDescriptors, matches);

        double minY = 9999, maxY = 0, minX = 9999, maxX = 0;

        for (int i = 0; i < atualMatch.size(); i++) {
            DMatch[] atual = atualMatch.get(i).toArray();

            for (int j = 0; j < atualMatch.get(i).rows(); j++) {
                if (atual[0].distance * 2.0 < atual[1].distance) {


                    Point ptAtual = kp.get(atual[0].queryIdx).pt;

                    if (true || !usedKP.contains(ptAtual)) {
                        good++;
                        usedKP.add(ptAtual);
                        if (flags[1]) {
                            double x = ptAtual.x;
                            double y = ptAtual.y;

                            if (x < minX) {
                                minX = x;
                            } else if (x > maxX) {
                                maxX = x;
                            }
                            if (y < minY) {
                                minY = y;
                            } else if (y > maxY) {
                                maxY = y;
                            }
                        }
                    }
                    if (flags[0]) {//DrawMatches
                        Core.circle(frame, kp.get(atual[0].queryIdx).pt, 6, new Scalar(255, 0, 255));
                    }


                }
            }
            //i++;
        }
        int pts = 9;

        if (frameKeypoints.size().height > 5000)
            pts = 17;
        else if (frameKeypoints.size().height > 9000)
            pts = 30;


        if (good >= pts) {
            return true;
        } else {
            good = 0;
        }

        return false;
    }

    /**
     * Habilita o desenho de do Keypoints que "batem".
     */
    public void DrawMatches() {
        flags[0] = true;
    }

    /**
     * Habilita o desenho de um quadrado em volta do objeto detectado.
     */
    public void DrawSquare() {
        flags[1] = true;
    }

    public boolean clean() {
        good = 0;
        return true;
    }

    public int getGood() {
        int x = good;
        good = 0;
        return x;
    }

    public void DrawKeypoints() {
        //Features2d.drawKeypoints(frame, frameKeypoints, frame);
        Features2d.drawKeypoints(frame, frameKeypoints, frame, new Scalar(255, 255, 255), Features2d.DRAW_RICH_KEYPOINTS);
    }

    /**
     * Desenha um texto com proposito de debugar.
     * N�o sera usado no programa final.
     */
    public void Debug() {
        Log.d(TAG, "Total Keypoints" + frameKeypoints.size().height);
        Core.putText(frame, "Total Keypoints: " + frameKeypoints.size(), new Point(10, 100), 5, 1.8, new Scalar(255, 255, 255));

    }

    /**
     * Retorna o Frame processado.
     * @return Frame com desenhos e textos.
     */
    public Mat getFrame() {
        return frame;
    }

    /**
     * Setar o frame atual.
     * @param frame Recebe o frame da comera em RGB
     */
    public void setFrame(Mat frame) {
        this.frame = frame;
    }

    /**
     * Pega os keypoits referentes as imagens passadas.
     * Primeiro e necessario chamar ComputeImages.
     * @return Keypoints referentes a imagem.
     */
    public Mat getKeypoints(int i) {
        return trainKeypoints.get(i);
    }

    /**
     * Pegos os descritores referentes as imagens passadas.
     * @return Descritores referentes as imagens.
     */
    public List<Mat> getDescriptors() {
        return trainDescriptors;
    }


}