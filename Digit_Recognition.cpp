#include <opencv2/opencv.hpp>  
#include<iostream>  
#include<stdio.h>
#include <string>
#include <fstream>


using namespace std;
using namespace cv;
using namespace cv::ml;



#define     Car_Num         74      //测试集数量
#define     Train_Num       14504   //训练集数量
#define     Test_Num        1665    //测试集数量
#define     Train_Rows      20      //训练集行数
#define     Train_Cols      20      //训练集列数
#define     _CRT_SECURE_NO_WARNINGS


Mat one_hot(Mat label, int classes_num);
void RandomArray(Mat Train, Mat Label, int num);
void read_image(const string path, Mat output_img, Mat label);


// 训练集数组，将训练集对应于一个数组
string Test_Arr[] = { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", \
                      "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", \
                      "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", \
                      "W", "X", "Y", "Z", \
                      "川", "鄂", "赣", "甘", "贵", "桂", "黑", "沪", \
                      "冀", "津", "京", "吉", "辽", "鲁", "蒙", "闽", \
                      "宁", "青", "琼", "陕", "苏", "晋", "皖", "湘", \
                      "新", "豫", "渝", "粤", "云", "藏", "浙" \
                   };



int main()
{
    /*    ---------第一部分：读取训练集及其标签----------    */
    // 训练集
    Mat TrainMat = Mat::zeros(Train_Num, Train_Rows* Train_Cols, CV_32FC1);
    // 训练集标签
    Mat TrainLabel = Mat::zeros(Train_Num, 1, CV_32SC1);
    // 训练集路径
    string Train_Path = "C:\\Users\\Tiam\\Desktop\\Digit_Recognition\\image\\train_picture";

    // 测试集
    Mat TestMat = Mat::zeros(Test_Num, Train_Rows * Train_Cols, CV_32FC1);
    // 测试集标签
    Mat TestLabel = Mat::zeros(Test_Num, 1, CV_32SC1);
    // 测试集路径
    string Test_Path = "C:\\Users\\Tiam\\Desktop\\Digit_Recognition\\image\\test_picture";

    // 读取训练集
    read_image(Train_Path, TrainMat, TrainLabel);

    // 随机打乱训练集和标签
    RandomArray(TrainMat, TrainLabel, Train_Num);

    // ann神经网络的标签数据需要转为one-hot型
    TrainLabel = one_hot(TrainLabel, size(Test_Arr));
    

    /*    ---------第二部分：构建ann训练模型并进行训练-----------   */
    cv::Ptr<cv::ml::ANN_MLP> ann = cv::ml::ANN_MLP::create();
    // 定义模型的层次结构 输入层为400 隐藏层为64 输出层为65
    Mat layerSizes = (Mat_<int>(1, 3) << Train_Rows* Train_Cols, 64, size(Test_Arr));
    ann->setLayerSizes(layerSizes);
    // 设置参数更新为误差反向传播法
    ann->setTrainMethod(ANN_MLP::BACKPROP, 0.001, 0.1);
    // 设置激活函数为sigmoid
    ann->setActivationFunction(ANN_MLP::SIGMOID_SYM, 1.0, 1.0);
    // 设置跌打条件 最大训练次数为100
    ann->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 10, 0.0001));
    
    // 开始训练
    cv::Ptr<cv::ml::TrainData> train_data = cv::ml::TrainData::create(TrainMat, cv::ml::ROW_SAMPLE, TrainLabel);
    cout << "开始进行训练..." << endl;
    ann->train(train_data);
    cout << "训练完成" << endl;

    
    /*    ---------第三部分：测试神经网络-----------   */
    // 读取测试集
    read_image(Test_Path, TestMat, TestLabel);
    Mat pre_out;
    // 返回值为第一个图像的预测值 pre_out为整个batch的预测值集合
    cout << "开始进行预测..." << endl;
    float ret = ann->predict(TestMat, pre_out);
    cout << "预测完成" << endl;

    // 计算准确率
    int equal_nums = 0;
    Mat img_original;

    for (int i = 0; i < pre_out.rows; i++)
    {
        // 获取每一个结果的最大值所在下标
        Mat temp = pre_out.rowRange(i, i + 1);
        double maxVal = 0;
        cv::Point maxPoint;
        cv::minMaxLoc(temp, NULL, &maxVal, NULL, &maxPoint);
        int max_index = maxPoint.x;
        int test_index = TestLabel.at<int32_t>(i, 0);
        if (max_index == test_index)
        {
            equal_nums++;
        }
        //// 此处可以查看每张图片的测试结果
        //img_original = TestMat.row(i);
        //img_original = img_original.reshape(0, Train_Rows);
        //imshow("test", img_original);
        //waitKey(0);
        //cout << Test_Arr[max_index] << endl;
    }
    float acc = float(equal_nums) / float(pre_out.rows);
    cout << "测试数据集上的准确率为：" << acc * 100 << "%" << endl;
}




void read_image(const string path, Mat output_img, Mat label)
{
    // 路径
    char folder[100];
    int n = 0;

    // 读取训练集
    for (int i = 0; i < size(Test_Arr); i++)
    {
        // 写入路径
        sprintf_s(folder, "%s\\%d", path.c_str(), i);
        vector<cv::String> imagePathList;
        // 读取路径下所有图片
        glob(folder, imagePathList);

        for (int j = 0; j < imagePathList.size(); j++)
        {
            // 第n行的首地址
            int* labelPtr = label.ptr<int>(n);
            // 赋值
            labelPtr[0] = i;
            // 读取
            auto img = imread(imagePathList[j]);
            // 转换成灰度图
            cvtColor(img, img, COLOR_RGB2GRAY);
            // 二值化
            threshold(img, img, 50, 255, THRESH_BINARY);
            // 归一化
            img = img / 255.0;
            //imshow("img", img);
            //waitKey(1);
            // 转换成一行
            Mat sample = img.reshape(0, 1);
            // 将源图像的行复制到目标图像的特定行
            sample.row(0).copyTo(output_img.row(n));
            n++;
        }
        imagePathList.clear();
    }
}

//将标签数据改为one-hot型
Mat one_hot(Mat label, int classes_num)
{
    //[2]->[0 1 0 0 0 0 0 0 0 0]
    int rows = label.rows;
    Mat one_hot = Mat::zeros(rows, classes_num, CV_32FC1);
    for (int i = 0; i < label.rows; i++)
    {
        int index = label.at<int32_t>(i, 0);
        one_hot.at<float>(i, index) = 1.0;
    }
    return one_hot;
}


//随机打乱训练集和标签
void RandomArray(Mat Train, Mat Label,int num)
{
    int tmp;
    Mat img;

    srand((int)time(NULL));
    for (int i = 0; i < num; i++)
    {
        tmp = rand() % num;

        Train.row(i).copyTo(img);
        Train.row(tmp).copyTo(Train.row(i));
        img.copyTo(Train.row(tmp));

        int t2 = Label.at<int>(i, 0);
        Label.at<int>(i, 0) = Label.at<int>(tmp, 0);
        Label.at<int>(tmp, 0) = t2;
    }
}

