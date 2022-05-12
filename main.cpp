//James Rogers Mar 2022 (c) Plymouth University
//Student ID: 10609827
#include<iostream>
#include<fstream>
#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;

//a drawing function that can draw a line based on rho and theta values.
//useful for drawing lines from the hough line detector.
void lineRT(Mat &Src, Vec2f L, Scalar color, int thickness){
    Point pt1, pt2;
    double a = cos(static_cast<double>(L[1]));
    double b = sin(static_cast<double>(L[1]));
    double x0 = a*static_cast<double>(L[0]);
    double y0 = b*static_cast<double>(L[0]);
    pt1.x = cvRound(x0 + 1000*(-b));
    pt1.y = cvRound(y0 + 1000*(a));
    pt2.x = cvRound(x0 - 150*(-b));
    pt2.y = cvRound(y0 - 150*(a));
    line(Src, pt1, pt2, color, thickness, LINE_AA);
}

void lineRT2(Mat &Src, Vec2f L, Scalar color, int thickness){
    Point pt3, pt4;
    double c = cos(static_cast<double>(L[1]));
    double d = sin(static_cast<double>(L[1]));
    double x0 = c*static_cast<double>(L[0]);
    double y0 = d*static_cast<double>(L[0]);
    pt3.x = cvRound(x0 + 10000*(-d));
    pt3.y = cvRound(y0 + 10000*(c));
    pt4.x = cvRound(x0 - 10000*(-d));
    pt4.y = cvRound(y0 - 10000*(c));
    line(Src, pt3, pt4, color, thickness, LINE_AA);
}

int main()
{
    //Open video file
    VideoCapture CarVideo("../Task5/DashCam.mp4");
    if(!CarVideo.isOpened()){
        cout<<"Error opening video"<<endl;
        return -1;
    }

    //main program loop
    while(true){

        //open the next frame from the video file, or exit the loop if its the end
        Mat Frame;
        CarVideo.read(Frame);
        if(Frame.empty()){
            break;
        }

        resize(Frame, Frame, Size(1280, 720), INTER_LINEAR);

        //==========================Your code goes here==========================

        //Create an edge map using grayscale, blur and canny functions
        Mat blur, gray, canny, edgeMap;
        cvtColor(Frame, gray, COLOR_BGR2GRAY);
        GaussianBlur(gray, blur, Size(9, 9), 0);
        Canny(blur, canny, 50, 150);

        Mat element = getStructuringElement(MORPH_RECT, Size(3,3));
        dilate(canny, edgeMap, element);

        //Set up triangular region of interest (ROI) using poly functions
        vector<Point> myPoly = {
            Point(160,715), Point(680, 340), Point(1100, 715)
        };

        vector<vector<Point>> contours = {myPoly}; //Stores poly as a vector of polys
        drawContours(Frame, contours, 0, Scalar(0,0,0), 0);

        //Define mask dimensions to match original frame size and set all pixels to black
        Mat maskZeros(Frame.size(), CV_8U, Scalar(0));

        //Fill triangular area in white within mask
        fillPoly(maskZeros, contours, Scalar(255));

        //Establish mat for bitwise function to create a masked lane frame
        Mat maskedLaneFrame;
        bitwise_and(edgeMap, maskZeros, maskedLaneFrame);

        ///// Hough Transformation /////
        double rhoRes = 1;              //Positional resolution
        double thetaRes = M_PI/180;     //Rotational resolution
        int middleThreshold = 330;      //Line-detection sensitivity (middle marking)
        int laneThreshold = 400;        //Line-detection sensitivity (lane marking)

        //Vectors to store rho and theta output values
        vector<Vec2f> laneMarking;
        vector<Vec2f> middleMarking;

        //HoughLines functions to detect lines
        HoughLines(maskedLaneFrame, laneMarking, rhoRes, thetaRes, laneThreshold);
        HoughLines(maskedLaneFrame, middleMarking, rhoRes, thetaRes, middleThreshold);

        //Print Hough lines on original frame
        for (int i = 0; i < (int)laneMarking.size(); i++){
            for (int j = 0; j < (int)middleMarking.size(); j++){
                lineRT(Frame, laneMarking[i], Scalar(0, 0, 255), 2);
                lineRT2(Frame, middleMarking[j], Scalar(0, 255, 0), 2);
                }
        }

        //Display frames
        imshow("Frame", Frame);
        imshow("Edge Map", edgeMap);
        imshow("Mask Zeros", maskZeros);
        imshow("Masked Lane Frame", maskedLaneFrame);
        waitKey(10);
    }
}
