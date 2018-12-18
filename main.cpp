#include "hough.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include <fstream>
//#include "FaceDetect.cpp"

String cascade_name = "cascade.xml";
CascadeClassifier cascade;
vector<Circle> detectAndDisplay( Mat frame)
{
    std::vector<Rect> darts;
    vector<Circle> res;
    Mat frame_gray;
    
    // 1. 灰度图像并增强对比度
    cvtColor( frame, frame_gray, CV_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
    
    // 2. 套用V-J 框架
    cascade.detectMultiScale( frame_gray, darts, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );
    
    // 3. 控制台输出检测到的dart个数
    std::cout << darts.size() << std::endl;
    
    
    // 5. 用圆形标记dart的边界
    for( int i = 0; i < darts.size(); i++ )
    {
        int centerpoint_x = (darts[i].x + darts[i].x + darts[i].width)/2;
        int centerpoint_y = (darts[i].y + darts[i].y + darts[i].height)/2;
        int r = darts[i].width/2;
        Circle center = make_pair(make_pair(centerpoint_x, centerpoint_y), r);
        res.push_back(center);
        circle(frame, Point(centerpoint_x, centerpoint_y),r, Scalar(0,255,0),2);
        
    }
    return res;
}

int doDetect(string fileName){
    vector<Circle> VJcircles;
    // 1. Read Input Image
    Mat frame = imread(fileName);
    
    // 2. Load the Strong Classifier in a structure called `Cascade'
    if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
    
    // 3. Detect darts and Display Result
    VJcircles = detectAndDisplay( frame);
 
    // 用hough检测目标图片，记录圆心，半径与classifier输出的结果比较
    
    Hough hough(fileName);
    hough.houghLineInit();
    hough.houghLineVote(120);
    hough.houghLineDetect(120);
    hough.houghLineDraw(fileName);
    
    //int minR, int maxR, int distance
    hough.houghCircleInit(1,200,100);
    float voteThre = 140;
    hough.houghCircleVote(voteThre);
    float outThre = 150;
    hough.houghCircleDetect(outThre);
    hough.houghCircleDraw("hough_Line_Object.jpg");
    
    hough.houghCombine("hough_Circle_Object.jpg", VJcircles);
    
    return 0;
}


int main(int argc, const char** argv )
{
    doDetect(argv[1]);
    //doDetect("images/dart15.jpg");
    
    return 0;
}
