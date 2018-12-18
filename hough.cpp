//
//  houghCircle.cpp
//  coin
//
//  Created by Yifan Jin on 2018/11/27.
//  Copyright © 2018 Yifan Jin. All rights reserved.
//

#include "hough.hpp"

Hough::Hough(string file)
:fileName(file),minR(0),maxR(0),distance(0),rows(0),cols(0){
    // do sobel, get the gradient gratitude and angle
    sobel();
}

void Hough::houghCircleInit(int min, int max, int dist){
    minR = min;
    maxR = max;
    distance = dist;
    rows = gradientMag.rows;
    cols = gradientMag.cols;
    
    // initialise the hough voting 3-D array (x[], y[], r[])
    for(int i=0;i<rows;i++){
        vector<RadiusVoting> row;
        for(int j=0;j<cols;j++){
            RadiusVoting rv;
            for(int k=0;k<max-min;k++){
                rv.push_back(0);
            }
            row.push_back(rv);
        }
        houghCircleVoting.push_back(row);
    }
    
    // initialise the hough voting accumulator which marginalises radius r
    houghCircleAcc.create(rows,cols,CV_32FC1);
    for(int i=0;i<houghCircleAcc.rows;i++){
        for(int j=0;j<houghCircleAcc.cols;j++)
            houghCircleAcc.at<float>(i,j) = 0;
    }
}

void Hough::printMat(Mat A){
    for(int i=0;i<A.rows;i++)
    {
        for(int j=0;j<A.cols;j++)
            cout<<A.at<float>(i,j)<<' ';
        cout<<endl;
    }
    cout<<endl;
}

void Hough::printMag(){
    printMat(gradientMag);
}

void Hough::printDir(){
    printMat(gradientDir);
}

void Hough::sobel(){
    Mat img = imread(fileName);
    
    Mat gray;
    cvtColor(img,gray,CV_BGR2GRAY);
    //    medianBlur(gray,gray,3);
    //equalizeHist(gray,gray);
    GaussianBlur(gray, gray, Size(5,5), 0,0);
    
    // get x and y gradient maps
    Mat sobelX;
    Mat sobelY;
    Sobel(gray,sobelX,CV_32FC1,1,0,3);
    Sobel(gray,sobelY,CV_32FC1,0,1,3);
    
    Mat mag;
    Mat dir;
    cartToPolar(sobelX,sobelY,mag,dir);
    
    gradientMag = mag;
    gradientDir = dir;
    
    // normalise in order to output
    Mat grad;
    normalize(mag,grad,0,255,NORM_MINMAX,-1,Mat());
    Mat phase;
    normalize(dir,phase,0,255,NORM_MINMAX,-1,Mat());

    imwrite("grad_Mag.jpg", grad);
    imwrite("grad_Dir.jpg", phase);
    
    grad.release();
    phase.release();
    
    img.release();
    gray.release();
    sobelX.release();
    sobelY.release();
    mag.release();
    dir.release();
}

void Hough::houghCircleVote(float th) {
    for(int r=0;r<maxR-minR;r++){
        for(int x=0;x<rows;x++){
            for(int y=0;y<cols;y++){
                if(gradientMag.at<float>(x,y)<=th)
                    continue;
                int x0=0, y0=0, x1=0,y1=0;
                x0 = cvRound(x + r*sin(gradientDir.at<float>(x,y)));
                y0 = cvRound(y + r*cos(gradientDir.at<float>(x,y)));
                x1 = cvRound(x - r*sin(gradientDir.at<float>(x,y)));
                y1 = cvRound(y - r*cos(gradientDir.at<float>(x,y)));
                if(x0 >= 0 && y0 >= 0 && x0 < rows && y0 < cols)
                    houghCircleVoting[x0][y0][r]+=1;
                if(x1 >= 0 && y1 >= 0 && x1 < rows && y1 < cols)
                    houghCircleVoting[x1][y1][r]+=1;
            }
        }
    }
}

void Hough::houghCircleDetect(float outThre) {
    // marginalise the radius dimension into accumulator
    for(int x=0;x<houghCircleVoting.size();x++) {
        for(int y=0; y<houghCircleVoting[0].size();y++){
            RadiusVoting rv = houghCircleVoting[x][y];
            float sum = 0;
            for(int i=0;i<rv.size();i++)
                sum+=rv[i];
            houghCircleAcc.at<float>(x,y) = sum;
        }
    }
    
    // normalise in order to output
    Mat houghCircleAccOut = houghCircleAcc;
    normalize(houghCircleAcc,houghCircleAccOut,0,255,NORM_MINMAX,-1,Mat());
    //imwrite("hough_Circle_Space.jpg",houghCircleAccOut);
    
    Mat centers;
    centers.create(houghCircleAccOut.rows,houghCircleAccOut.cols,CV_32FC1);
    for(int x=0;x<houghCircleAccOut.rows;x++) {
        for(int y=0; y<houghCircleAccOut.cols;y++){
            // try to find the local max by eliminating all that lower than itself
            float pixel = houghCircleAccOut.at<float>(x,y);
            // only care pixels higher than outThre
            if (pixel>outThre){
                bool isMax = true;
                // scan the square area around the pixel
                for(int i=0;i<distance;i++){
                    // deal with out-of-range problem
                    if(cvRound(x-distance/2 + i) <= 0 || cvRound(x-distance/2 + i)>=houghCircleAccOut.rows) continue;
                    for(int j=0;j<distance;j++){
                        if(cvRound(y-distance/2 + j) <= 0 ||  cvRound(y-distance/2 + j)>=houghCircleAccOut.cols) continue;
                        // erase all that smaller than the current pixel
                        if(houghCircleAccOut.at<float>(cvRound(x-distance/2 + i),cvRound(y-distance/2 + j)) <= pixel)
                            houghCircleAccOut.at<float>(cvRound(x-distance/2 + i),cvRound(y-distance/2 + j)) = 0;
                        // if the current pixel is not the largest
                        else
                            isMax = false;
                    }
                }
                if(isMax){
                    centers.at<float>(x,y) = 255;
                    // find the most possible radius of the current circle center
                    RadiusVoting temp = houghCircleVoting[x][y];
                    int r = 0, maxVote = 0;
                    for(int i=0;i<temp.size();i++){
                        if (temp[i]>maxVote){
                            r = i;
                            maxVote = temp[i];
                        }
                    }
                    circles.push_back(make_pair(make_pair(x, y), r));
                    circleVotes.push_back(maxVote);
                }
                else
                    centers.at<float>(x,y) = 0;
            }
            else
                centers.at<float>(x,y) = 0;
        }
    }
    //imwrite("hough_Circle_Centers.jpg",centers);
    centers.release();
    houghCircleAccOut.release();
    
}

void Hough::houghCircleDraw(string outFileName) {
    Mat img = imread(outFileName);
    
    for(int i=0;i<circles.size();i++){
        int x = circles[i].first.first;
        int y = circles[i].first.second;
        int r = circles[i].second;
        rectangle(img, Point(cvRound(y-r),cvRound(x-r)), Point(cvRound(y+r),cvRound(x+r)), Scalar(0,255,255),2);
    }
    imwrite("hough_Circle_Object.jpg", img);
    img.release();

}

void Hough::houghLineInit() {
    rows = gradientMag.rows;
    cols = gradientMag.cols;
    
    int maxRho = max(rows,cols);
    houghLineAcc.create(maxRho,360,CV_32FC1);
    
    for(int i=0;i<houghLineAcc.rows;i++){
        for(int j=0;j<houghLineAcc.cols;j++)
            houghLineAcc.at<int>(i,j)=0;
    }
}

void Hough::houghLineVote(float th) {
    for(int x=0;x<rows;x++){
        for(int y=0;y<cols;y++){
            if(gradientMag.at<float>(x,y)<=th)
                continue;
            for (int degree = 0; degree<360;degree++){
                float theta = (float)degree*CV_PI/180;
                float rho = y*cos(theta) + x*sin(theta);
                if(rho<=0 || rho>houghLineAcc.rows)
                    continue;
                houghLineAcc.at<float>((int)rho,degree)+=1;
            }
        }
    }
}

void Hough::houghLineDetect(float outThre) {
    Mat houghLineAccOut;
    normalize(houghLineAcc,houghLineAccOut,0,255,NORM_MINMAX,-1,Mat());
    //imwrite("hough_Line_Space.jpg",houghLineAccOut);
    
    Mat features;
    features.create(houghLineAccOut.rows,houghLineAccOut.cols,CV_32FC1);
    
    vector<Line> candidateLines;
    for(int rho=0;rho<houghLineAccOut.rows;rho++) {
        for(int degree=0; degree<houghLineAccOut.cols;degree++){
            float pixel = houghLineAccOut.at<float>(rho,degree);
            // only care pixels higher than outThre
            if(abs(degree%90)<10) continue;
            if (pixel>outThre){
                features.at<float>(rho,degree) = pixel;
                float theta = (float)degree/180*CV_PI;
                candidateLines.push_back(make_pair(rho,theta));
            }
            else
                features.at<float>(rho,degree) = 0;
            
        }
    }
    
    if(candidateLines.size()>0){
    vector<Line> uniqueLines;
    uniqueLines.push_back(candidateLines[0]);
    for(int i=0;i<candidateLines.size();i++){
        Line line1 = candidateLines[i];
        bool sim = false;
        for(int j=0;j<uniqueLines.size();j++){
            Line line2 = uniqueLines[j];
            if(lineSimilar(line1, line2)){
                sim = true;
                break;
            }
        }
        if(sim == false)
            uniqueLines.push_back(candidateLines[i]);
    }
    for(auto iter = uniqueLines.begin();iter!=uniqueLines.end();iter++){
        lines.push_back(*iter);
    }
    }
    //imwrite("hough_Line_Features.jpg",features);
    features.release();
    houghLineAccOut.release();
}

void Hough::houghLineDraw(std::string outFileName) {
    Mat linePoints = imread(outFileName);

    for(int i=0;i<lines.size();i++){
        float rho = lines[i].first;
        float theta = lines[i].second;
        
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        
        pt1.x = cvRound(x0 + 1000*(-b));
        pt1.y = cvRound(y0 + 1000*(a));
        pt2.x = cvRound(x0 - 1000*(-b));
        pt2.y = cvRound(y0 - 1000*(a));
        
        line(linePoints, pt1, pt2, Scalar(255,0,255), 1, CV_AA);
    }
    
    Mat crossVote(rows, cols,CV_32FC1,Scalar(0));
    // count crossPoint voting
    if(lines.size()>1){
    for(int i=0;i<lines.size()-1;i++){
        Line line1 = lines[i];
        for(int j=i+1;j<lines.size();j++){
            Line line2 = lines[j];
            Point crossPoint = getCrossPoint(line1, line2);
            if(crossPoint.x<=0 || crossPoint.x>=cols || crossPoint.y<=0 || crossPoint.y>=rows)
                continue;
            crossVote.at<float>(cvRound(crossPoint.y),cvRound(crossPoint.x))+=10;
//            circle(crossVote, crossPoint, 3, Scalar(1));
        }
    }
    GaussianBlur(crossVote, crossVote, Size(3,3), 0);
    for(int i=0;i<crossVote.rows;i++){
        for(int j=0;j<crossVote.cols;j++){
            if(crossVote.at<float>(i,j)>1){
                crossPoints.push_back(Point(j,i));
                circle(linePoints, Point(j,i), 3, Scalar(255,255,0));
            }
        }
    }
    }
    Mat linePointsOut = linePoints;
    imwrite("hough_Line_Object.jpg", linePointsOut);
    linePointsOut.release();
}

cv::Point Hough::getCrossPoint(Line line1, Line line2) {
    float rho1 = line1.first;
    float theta1 = line1.second;
    float rho2 = line2.first;
    float theta2 = line2.second;
    
    float k1 = -cos(theta1)/sin(theta1);
    float b1 = rho1/sin(theta1);
    float k2 = -cos(theta2)/sin(theta2);
    float b2 = rho2/sin(theta2);
    
    float x = (b2-b1)/(k1-k2);
    float y = k1*x+b1;
    
    return Point(cvRound(x),cvRound(y));
}

bool Hough::lineSimilar(Line line1, Line line2) { 
    float rho1 = line1.first;
    float theta1 = line1.second;
    float rho2 = line2.first;
    float theta2 = line2.second;
    
    if(abs(theta1-theta2)/CV_PI*180 > 30)
        return false;
    if(abs(rho1-rho2) > 50 )
        return false;
    return true;
}

void Hough::houghCombine(string outFileName, vector<Circle> VJcircles) {
    Mat combinedImg = imread(outFileName);
    for(int i=0;i<VJcircles.size();i++){
        circle(combinedImg, Point(VJcircles[i].first.first,VJcircles[i].first.second), VJcircles[i].second, Scalar(0,255,0),2);
    }
    imwrite("hough_Combined_Object.jpg", combinedImg);
    combinedImg.release();
    
    Mat img = imread(fileName);
    
//    auto maxIter = max_element(circleVotes.begin(), circleVotes.end());
//    //int maxVote = *maxIter;
//    int maxIdx = maxIter-circleVotes.begin();
//    Circle maxCircle = circles[maxIdx];
//    int x = maxCircle.first.first;
//    int y = maxCircle.first.second;
//    int r = maxCircle.second;
//    combined.push_back(make_pair(make_pair(x, y), r));
//    rectangle(img, Point(cvRound(x-r),cvRound(y-r)), Point(cvRound(x+r),cvRound(y+r)), Scalar(0,0,255),3);

    for(int i=0;i<circles.size();i++){
        int x = circles[i].first.first;
        int y = circles[i].first.second;
        int r = circles[i].second;
        for(int j=0;j<VJcircles.size();j++){
            int vx = VJcircles[j].first.first;
            int vy = VJcircles[j].first.second;
            int vr = VJcircles[j].second;
            if((vx-y)*(vx-y)+(vy-x)*(vy-x)<=(vr+r)*(vr+r)){
                y = cvRound(0.6*y+0.4*vx);
                x = cvRound(0.6*x+0.4*vy);
                r = cvRound(r+vr);
                combined.push_back(make_pair(make_pair(y, x),r));
                //rectangle(img, Point(cvRound(y-r),cvRound(x-r)), Point(cvRound(y+r),cvRound(x+r)), Scalar(0,255,0),4);
                break;
            }
        }
    }
    if(combined.size()<1){
        int crossThre = cvFloor((double)crossPoints.size()/(VJcircles.size()+circles.size()));
        
        for(int i=0;i<circles.size();i++){
            int x = circles[i].first.first;
            int y = circles[i].first.second;
            int r = circles[i].second;
            int crossCount = 0;
            for(int j =0;j<crossPoints.size();j++){
                Point p = crossPoints[j];
                if(pow(p.y-x,2)+pow(p.x-y,2)<pow(r,2))
                    crossCount++;
                if(crossCount>crossThre)
                    break;
            }
            if(crossCount>crossThre){
                combined.push_back(make_pair(make_pair(y, x), r));
                // rectangle(img, Point(cvRound(x-r),cvRound(y-r)), Point(cvRound(x+r),cvRound(y+r)), Scalar(0,0,255),2);
            }
        }
        for(int i=0;i<VJcircles.size();i++){
            int x = VJcircles[i].first.first;
            int y = VJcircles[i].first.second;
            int r = VJcircles[i].second;
            int crossCount = 0;
            for(int j =0;j<crossPoints.size();j++){
                Point p = crossPoints[j];
                if(pow(p.y-y,2)+pow(p.x-x,2)<pow(r,2))
                    crossCount++;
                if(crossCount>crossThre)
                    break;
            }
            if(crossCount>crossThre){
                for(int c=0;c<combined.size();c++){
                    combined.push_back(make_pair(make_pair(y, x), r));
                    //rectangle(img, Point(cvRound(y-r),cvRound(x-r)), Point(cvRound(y+r),cvRound(x+r)), Scalar(0,0,255),2);
                }
            }
        }
    }
//
    vector<Circle> uniqueCircle;
    if(combined.size()>0){
    uniqueCircle.push_back(combined[0]);
    for(int i=0;i<combined.size();i++){
        int x = combined[i].first.first;
        int y = combined[i].first.second;
        int r = combined[i].second;
        for(int j=0;j<uniqueCircle.size();j++){
            int cx = uniqueCircle[j].first.first;
            int cy = uniqueCircle[j].first.second;
            int cr = uniqueCircle[j].second;
            if((cx-x)*(cx-x)+(cy-y)*(cy-y)<=(cr+r)*(cr+r)){
                int xav = cvRound(0.5*(x+cx));
                int yav = cvRound(0.5*(y+cy));
                int rav = cvRound(0.5*(cr+r));
                Circle ave = make_pair(make_pair(xav, yav), rav);
                uniqueCircle[j]=ave;
                break;
            }
            else{
                uniqueCircle.push_back(combined[i]);
            }
        }
    }
    }
    for (int i=0;i<uniqueCircle.size();i++){
        int x = uniqueCircle[i].first.first;
        int y = uniqueCircle[i].first.second;
        int r = uniqueCircle[i].second;
        rectangle(img, Point(cvRound(x-r),cvRound(y-r)), Point(cvRound(x+r),cvRound(y+r)), Scalar(0,255,0),4);
    }
    imwrite("hough_Detected.jpg", img);
    img.release();

}



