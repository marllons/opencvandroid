package com.marll.testopencv.activity;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.speech.tts.TextToSpeech;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.SurfaceView;
import android.os.Handler;
import android.widget.TextView;


import com.marll.testopencv.R;
import com.marll.testopencv.util.ImageDetector;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.features2d.Features2d;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {
    //Constants
    private static final String TAG = "======= LOGTAG ======= " + MainActivity.class.toString();
    private static final int SELECT_PHOTO = 100;
    private static final int REQUEST_PERMISSION = 1001;

    static {
        System.loadLibrary("opencv_java");
        System.loadLibrary("nonfree");
    }

    private Mat mRgba;
    private Mat mGray;
    private Handler handler;
    private ImageDetector imageDetector;
    private String detectedObj, a;
    private String lastDetectedObj;
    //Attributes
    private CameraBridgeViewBase mOpenCvCameraView;
    private String ModeValue;
    //Characters(Pre-Load)
    private Mat belchior1, test1;
    private List<Mat> loadedImages;
    private ArrayList<String> imageNames;
    private boolean[] DetectedChars;
    //Thread Control
    private boolean[] threadControl;
    private boolean[] objControl;


    //Constructors

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");
                    try {
                        initializeOpenCVDependencies(mOpenCvCameraView);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    private void initializeOpenCVDependencies(CameraBridgeViewBase mOpenCvCameraView) throws IOException {
        mOpenCvCameraView.enableView();
        mOpenCvCameraView.setFocusable(true);
    }

    //@SuppressLint("HandlerLeak")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, REQUEST_PERMISSION);
        }
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.HelloOpenCvView);

        detectedObj = "-";

        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
        mOpenCvCameraView.setUserRotation(90);

        handler = new Handler();

    }

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_11, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }


    @Override
    public void onCameraViewStarted(int width, int height) {
        //Starts Characters Variables
        belchior1 = new Mat();
        test1 = new Mat();
        Mat belc = new Mat();
        Mat belc2 = new Mat();



        loadedImages = new ArrayList<Mat>();
        imageNames = new ArrayList<String>();


        try {
            belchior1 = Utils.loadResource(this, R.drawable.belchior);
            Imgproc.Canny(belchior1, belc, 60, 60*3);
            loadedImages.add(belc);//0
            imageNames.add("belchior1");

            test1 = Utils.loadResource(this, R.drawable.belchior4);
            Imgproc.Canny(test1, belc2, 60, 60*3);
            loadedImages.add(belc2);//1
            imageNames.add("belchior4");

            imageDetector = new ImageDetector(loadedImages, imageNames); //constroi as entradas com as img ja salvas
            imageDetector.computeImages(); //grava as entradas e computa elas

        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    @Override
    public void onCameraViewStopped() {

    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();
        Mat edges = new Mat();
        Imgproc.Canny(mGray, edges, 60, 60*3);
        lastDetectedObj = detectedObj;
        imageDetector.setFrame(mRgba);
        detectedObj = imageDetector.recognize(edges);


        MatOfInt rgb = new MatOfInt(CvType.CV_32S);
        edges.convertTo(rgb,CvType.CV_32S);
        int[] rgba = new int[(int)(rgb.total()*rgb.channels())];
        //rgb.get(0,0,rgba);
        handler.post(new EditViewRunnable());

        return mRgba;
    }

    private class EditViewRunnable  implements Runnable {
        @Override
        public void run() {
            TextView detectedObjTextView = (TextView) findViewById(R.id.detectedObjTextView);
            detectedObjTextView.setText(detectedObj);

        }
    }


}
