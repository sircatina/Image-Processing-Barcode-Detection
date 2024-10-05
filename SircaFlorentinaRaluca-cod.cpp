#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <unordered_map>
#include <vector>
#include "stdafx.h"
#include "common.h"
#include <math.h>
#include <queue>
#include <random>
#include <stdio.h>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <numeric> 
#include <cmath> 

wchar_t* projectPath;

using namespace cv;
using namespace std;
int NO_EDGE = 0;
int WEAK_EDGE = 128;
int STRONG_EDGE = 255;

// Function prototypes
Mat convertToGray(const Mat& image);
vector<vector<double>> createGaussianKernel(int ksize, double sigma);
Mat applyGaussianBlur(const Mat& gray, int ksize, double sigma);
Mat preprocessImage(const Mat& image);
Rect findBarcodeContour(const Mat& edges, double scaleFactor = 0.8);
Mat extractBarcodeRegion(const Mat& image, const Rect& barcodeRect);
Mat grayscaleToBinary(const Mat& image);
vector<int> extractPattern(const Mat& line, int moduleWidth, int startPos);


Mat convertToGray(const Mat& image) {
    int height = image.rows;
    int width = image.cols;
    Mat gray = Mat(height, width, CV_8UC1);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            Vec3b pixel = image.at<Vec3b>(i, j);
            uchar grayValue = static_cast<uchar>(pixel[0] * 0.114 + pixel[1] * 0.587 + pixel[2] * 0.299);
            gray.at<uchar>(i, j) = grayValue;
        }
    }
    return gray;
}

vector<vector<double>> createGaussianKernel(int ksize, double sigma) {
    int halfSize = ksize / 2;
    vector<vector<double>> kernel(ksize, vector<double>(ksize));
    double sum = 0.0;

    for (int i = -halfSize; i <= halfSize; ++i) {
        for (int j = -halfSize; j <= halfSize; ++j) {
            kernel[i + halfSize][j + halfSize] = (1 / (2 * CV_PI * sigma * sigma)) * exp(-(i * i + j * j) / (2 * sigma * sigma));
            sum += kernel[i + halfSize][j + halfSize];
        }
    }

    // Normalizarea kernelului
    for (int i = 0; i < ksize; ++i) {
        for (int j = 0; j < ksize; ++j) {
            kernel[i][j] /= sum;
        }
    }

    return kernel;
}

// Funcția pentru aplicarea blur-ului Gaussian
Mat applyGaussianBlur(const Mat& gray, int ksize, double sigma) {
    Mat blurred(gray.rows, gray.cols, CV_8UC1);
    vector<vector<double>> kernel = createGaussianKernel(ksize, sigma);
    int halfSize = ksize / 2;

    for (int i = 0; i < gray.rows; ++i) {
        for (int j = 0; j < gray.cols; ++j) {
            double sum = 0.0;
            for (int ki = -halfSize; ki <= halfSize; ++ki) {
                for (int kj = -halfSize; kj <= halfSize; ++kj) {
                    int x = min(max(i + ki, 0), gray.rows - 1);
                    int y = min(max(j + kj, 0), gray.cols - 1);
                    sum += gray.at<uchar>(x, y) * kernel[ki + halfSize][kj + halfSize];
                }
            }
            blurred.at<uchar>(i, j) = static_cast<uchar>(sum);
        }
    }

    return blurred;
}

Mat preprocessImage(const Mat& image) {
    Mat gray, blurred;
    gray = convertToGray(image);
    blurred = applyGaussianBlur(gray, 5, 0.5);
    return blurred;
}

Rect findBarcodeContour(const Mat& edges, double scaleFactor) {
    vector<vector<Point>> contours;
    findContours(edges, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    Rect barcodeRect;
    double maxArea = 0;

    for (const auto& contour : contours) {
        double area = contourArea(contour);
        RotatedRect rotatedRect = minAreaRect(contour); 
        // Calculăm chenarul rotit pentru a gestiona cazurile când chenarul nu este un dreptunghi perfect

        // Verificăm dacă chenarul conține suficiente puncte și dacă aria este maximă
        if (contour.size() > 4 && area > maxArea) {
            maxArea = area;
            barcodeRect = rotatedRect.boundingRect();
        }
    }

    // Redimensionăm chenarul pentru a include un spațiu înconjurător
    if (barcodeRect.width != 0 && barcodeRect.height != 0) {
        int offsetX = static_cast<int>(barcodeRect.width * (1 - scaleFactor) / 2);
        int offsetY = static_cast<int>(barcodeRect.height * (1 - scaleFactor) / 2);
        barcodeRect.x += offsetX;
        barcodeRect.y += offsetY;
        barcodeRect.width = static_cast<int>(barcodeRect.width * scaleFactor);
        barcodeRect.height = static_cast<int>(barcodeRect.height * scaleFactor);
    }

    return barcodeRect;
}



Mat extractBarcodeRegion(const Mat& image, const Rect& barcodeRect) {
    if (barcodeRect.width == 0 || barcodeRect.height == 0) {
        throw runtime_error("No barcode detected.");
    }
    return image(barcodeRect).clone();
}



Mat grayscaleToBinary(const Mat& image) {
    Mat grayscaleImage, binaryImage;
    cvtColor(image, grayscaleImage, COLOR_BGR2GRAY);
    threshold(grayscaleImage, binaryImage, 128, 255, THRESH_BINARY);
    return binaryImage;
}

vector<int> extractPattern(const Mat& line, int moduleWidth, int startPos) {
    vector<int> pattern;
    int count = 1;
    for (int i = startPos + 1; i < line.cols; ++i) {
        if (line.at<uchar>(0, i) == line.at<uchar>(0, i - 1)) {
            ++count;
        }
        else {
            pattern.push_back(round(static_cast<double>(count) / moduleWidth));
            count = 1;
        }
    }
    pattern.push_back(round(static_cast<double>(count) / moduleWidth));
    return pattern;
}
void saveConsecutiveSegments(const Mat& middleRow, vector<int>& segments) {

    if (middleRow.empty() || middleRow.cols == 0) {
        cerr << "Empty or invalid middle row!" << endl;
        return;
    }

    int currentPixel = middleRow.at<uchar>(0, 0);
    int count = 1;

    for (int i = 1; i < middleRow.cols; ++i) {
        int pixel = middleRow.at<uchar>(0, i);
        if (pixel == currentPixel) {
            ++count;
        }
        else {
            segments.push_back(count);
            currentPixel = pixel;
            count = 1;
        }
    }
    // Save the last segment
    segments.push_back(count);
}

int countMatchingChars(const string& str1, const string& str2) {
    int count = 0;
    for (size_t i = 0; i < str1.size(); ++i) {
        if (str1[i] == str2[i]) {
            ++count;
        }
    }
    return count;
}
string reverseString(const string& str) {
    string reversed = str;
    reverse(reversed.begin(), reversed.end());
    return reversed;
}
string processSegments(const vector<int>& segments) {
    string decodedBarcode;
    if (segments.size() < 57) {
        decodedBarcode = "Not enough segments to process according to the specified pattern!";
        //cerr << "Not enough segments to process according to the specified pattern!" << endl;
        return decodedBarcode;
    }
    vector<string> digitPatterns = { "3211", "2221", "2122", "1411", "1132",
                                 "1231", "1114", "1312", "1213", "3112" };
    // Process the first segment
    //std::cout << "First segment: " << segments[0] << endl;

    // Calculate the unit size using the 2nd, 3rd, and 4th segments
    double unit = accumulate(segments.begin() + 1, segments.begin() + 4, 0.0) / 3.0;
    int aproximat = round(unit);

    int index = 1;
    

    //std::cout << "The starting group of 3 segments: "; ---------------------------------------------------------------------
    for (int i = 0; i < 3; ++i) {
        int numSegments = round((double)segments[index] / aproximat); 
        ++index;
    }
    std::cout << endl;


    // Group the next 24 segments in groups of 4 (6 numbers x 4 units) -----------------------------------------------------------------------------
    for (int j = 0; j < 6; j++) {
        vector<int> generatedNumbers;
        for (int i = 0; i < 4; ++i) {
            int numSegments = round((double)segments[index] / aproximat);
            //std::cout << numSegments << " ";
            if (segments[index] >= aproximat * 4) numSegments = 4;
            else if (segments[index] >= aproximat * 3) numSegments = 3;
            generatedNumbers.push_back(numSegments);
            ++index;
        }

        string generatedStr = "";
        for (int num : generatedNumbers) {
            generatedStr += to_string(num);
        }

        string reversedStr = reverseString(generatedStr);

        bool foundMatch = false;
        for (size_t i = 0; i < digitPatterns.size(); ++i) {
            if (countMatchingChars(digitPatterns[i], generatedStr) >= 3 ||
                countMatchingChars(digitPatterns[i], reversedStr) > 3) {
                foundMatch = true;
                decodedBarcode += to_string(i) + " ";
                break;
            }
        }
        if (!foundMatch) {
            decodedBarcode += "? ";
        }
        //cout << endl;
    }
    // Group the next 5 segments which are the middle part - half of the code
    //std::cout<< "Group of 5 segments: "; -------------------------------------------------------------------------------
    for (int i = 0; i < 5; ++i) {
        int numSegments = round((double)segments[index] / aproximat); 
        ++index;
    }
    // Group the next 24 segments in groups of 4 (6 numbers x 4 units) ------------------------------------------------------------------------
     for (int j = 0; j < 6; j++) {
        vector<int> generatedNumbers;
        for (int i = 0; i < 4; ++i) {
            int numSegments = round((double)segments[index] / aproximat);
            if (segments[index] >= aproximat * 4) numSegments = 4;
            else if (segments[index] >= aproximat * 3) numSegments = 3;
            generatedNumbers.push_back(numSegments);
            ++index;
        }

        string generatedStr = "";
        for (int num : generatedNumbers) {
            generatedStr += to_string(num);
        }

        string reversedStr = reverseString(generatedStr);

        bool foundMatch = false;
        for (size_t i = 0; i < digitPatterns.size(); ++i) {
            if (countMatchingChars(digitPatterns[i], generatedStr) >= 3 ||
                countMatchingChars(digitPatterns[i], reversedStr) >= 3) {
                foundMatch = true;
                decodedBarcode += to_string(i) + " ";
                break;
            }
        }
        if (!foundMatch) {
            decodedBarcode += "? ";
        }
    }
     cout << endl;
     return decodedBarcode;
    // end Group- 3 segments
    //std::cout << "Grupul de final de 3 segmente: ";-------------------------------------------------------------------------------
    //for (int i = 0; i < 3; ++i) {
    //    //std::cout << segments[index] << " ";
    //    int numSegments = round((double)segments[index] / aproximat); // Calculăm de câte ori se potrivește segmentul în valoarea aproximată
    //   // std::cout << " " << numSegments;
    //    ++index;
    //}
    // Print the last segment
    //std::cout << "Last segment: " << segments[index] << endl;
}


 Mat applyFilter(const Mat& src, float filter[3][3]) {
     int height = src.rows;
     int width = src.cols;
     int halfFilterSize = 1;
     Mat dst = Mat(height, width, CV_8UC1, Scalar(0));

     for (int i = halfFilterSize; i < height - halfFilterSize; ++i) {
         for (int j = halfFilterSize; j < width - halfFilterSize; ++j) {
             float total = 0;
             for (int m = 0; m < 3; ++m) {
                 for (int n = 0; n < 3; ++n) {
                     int rowIndex = i + m - halfFilterSize;
                     int colIndex = j + n - halfFilterSize;
                     total += filter[m][n] * src.at<uchar>(rowIndex, colIndex);
                 }
             }
             dst.at<uchar>(i, j) = saturate_cast<uchar>(total);
         }
     }
     return dst;
 }

 void sobelFilter(const Mat& src, Mat& gradX, Mat& gradY) {
     int height = src.rows;
     int width = src.cols;
     int sobelX[3][3] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
     int sobelY[3][3] = { 1, 2, 1, 0, 0, 0, -1, -2, -1 };

     for (int i = 1; i < height - 1; ++i) {
         for (int j = 1; j < width - 1; ++j) {
             float gx = 0, gy = 0;
             for (int m = 0; m < 3; ++m) {
                 for (int n = 0; n < 3; ++n) {
                     int rowIndex = i + m - 1;
                     int colIndex = j + n - 1;
                     gx += sobelX[m][n] * src.at<uchar>(rowIndex, colIndex);
                     gy += sobelY[m][n] * src.at<uchar>(rowIndex, colIndex);
                 }
             }
             gradX.at<float>(i, j) = gx;
             gradY.at<float>(i, j) = gy;
         }
     }
 }

 Mat nonMaximumSuppression(const Mat& magnitude, const Mat& direction) {
     int height = magnitude.rows;
     int width = magnitude.cols;
     Mat suppressed = Mat(height, width, CV_8UC1, Scalar(0));

     for (int i = 1; i < height - 1; ++i) {
         for (int j = 1; j < width - 1; ++j) {
             float angle = direction.at<float>(i, j) * 180.0 / CV_PI;
             angle = angle < 0 ? angle + 180 : angle;

             float mag = magnitude.at<uchar>(i, j);
             float mag1, mag2;

             if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle < 180)) {
                 mag1 = magnitude.at<uchar>(i, j - 1);
                 mag2 = magnitude.at<uchar>(i, j + 1);
             }
             else if (angle >= 22.5 && angle < 67.5) {
                 mag1 = magnitude.at<uchar>(i - 1, j + 1);
                 mag2 = magnitude.at<uchar>(i + 1, j - 1);
             }
             else if (angle >= 67.5 && angle < 112.5) {
                 mag1 = magnitude.at<uchar>(i - 1, j);
                 mag2 = magnitude.at<uchar>(i + 1, j);
             }
             else {
                 mag1 = magnitude.at<uchar>(i - 1, j - 1);
                 mag2 = magnitude.at<uchar>(i + 1, j + 1);
             }

             if (mag >= mag1 && mag >= mag2) {
                 suppressed.at<uchar>(i, j) = mag;
             }
         }
     }
     return suppressed;
 }

 void doubleThreshold(Mat& image, int lowThreshold, int highThreshold) {
     for (int i = 0; i < image.rows; ++i) {
         for (int j = 0; j < image.cols; ++j) {
             uchar pixel = image.at<uchar>(i, j);
             if (pixel >= highThreshold) {
                 image.at<uchar>(i, j) = STRONG_EDGE;
             }
             else if (pixel >= lowThreshold) {
                 image.at<uchar>(i, j) = WEAK_EDGE;
             }
             else {
                 image.at<uchar>(i, j) = NO_EDGE;
             }
         }
     }
 }

 void edgeTracking(Mat& image) {
     int dx[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };
     int dy[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };

     for (int i = 1; i < image.rows - 1; ++i) {
         for (int j = 1; j < image.cols - 1; ++j) {
             if (image.at<uchar>(i, j) == STRONG_EDGE) {
                 std::queue<Point> q;
                 q.push(Point(j, i));
                 while (!q.empty()) {
                     Point p = q.front();
                     q.pop();
                     for (int k = 0; k < 8; ++k) {
                         int ni = p.y + dy[k];
                         int nj = p.x + dx[k];
                         if (image.at<uchar>(ni, nj) == WEAK_EDGE) {
                             image.at<uchar>(ni, nj) = STRONG_EDGE;
                             q.push(Point(nj, ni));
                         }
                     }
                 }
             }
         }
     }

     for (int i = 0; i < image.rows; ++i) {
         for (int j = 0; j < image.cols; ++j) {
             if (image.at<uchar>(i, j) == WEAK_EDGE) {
                 image.at<uchar>(i, j) = NO_EDGE;
             }
         }
     }
 }

 Mat edge_detection2(const Mat& img) {
     int height = img.rows;
     int width = img.cols;

     float gaussianFilter[3][3];

     
     float sigma = 0.5f;
     for (int k = 0; k < 3; k++) {
         for (int l = 0; l < 3; l++) {
             float putere = (pow((k - 1), 2) + pow((l - 1), 2)) / (2 * sigma * sigma);
             float ex = exp(-putere);
             gaussianFilter[k][l] = 1 / (2 * CV_PI * sigma * sigma) * ex;
         }
     }

     Mat blurred = applyFilter(img, gaussianFilter);

     Mat gradX = Mat(height, width, CV_32FC1);
     Mat gradY = Mat(height, width, CV_32FC1);

     sobelFilter(blurred, gradX, gradY);

     Mat magnitude = Mat(height, width, CV_8UC1, Scalar(0));
     Mat direction = Mat(height, width, CV_32FC1, Scalar(0));

     for (int i = 0; i < height; ++i) {
         for (int j = 0; j < width; ++j) {
             float gx = gradX.at<float>(i, j);
             float gy = gradY.at<float>(i, j);
             magnitude.at<uchar>(i, j) = saturate_cast<uchar>(sqrt(gx * gx + gy * gy));
             direction.at<float>(i, j) = atan2(gy, gx);
         }
     }
     Mat copie_magnitude = magnitude.clone();
     Mat nonMaxSupp = nonMaximumSuppression(magnitude, direction);

     int hist[256];

     for (int l = 0; l < 256; l++) {
         hist[l] = 0;
     }

     for (int k = 0; k < copie_magnitude.rows; k++) {
         for (int l = 0; l < copie_magnitude.cols; l++) {
             hist[copie_magnitude.at<uchar>(k, l)]++;
         }
     }
     float p = 0.1;
     int sum = 0;
     int ThresholdHigh = 0;
     int  ThresholdLow = 0;
     float kapa = 0.4;
     float nonEdgePixelss = (1 - p) * ((copie_magnitude.rows - 2) * (copie_magnitude.cols - 2) - hist[0]);
     for (int l = 1; l < 256; l++) {
         sum += hist[l];
         if (nonEdgePixelss < sum)
         {
             ThresholdHigh = l;
             break;
         }
     }
     ThresholdLow = ThresholdHigh * kapa;
     doubleThreshold(nonMaxSupp, ThresholdLow, ThresholdHigh);

     edgeTracking(nonMaxSupp);

     return nonMaxSupp;
 }
int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
    projectPath = _wgetcwd(0, 0);
    char fname[MAX_PATH];
    while (openFileDlg(fname)) {
        Mat src = imread(fname);
        imshow("input image", src);
        if (src.empty()) {
            cerr << "Image not found!" << endl;
            return -1;
        }

        try {
            
            Mat preprocessedImage = preprocessImage(src);
            imshow("preprocessed Image", preprocessedImage);
            Mat preprocessedImageClone = preprocessedImage.clone();
            Mat edge= edge_detection2(preprocessedImageClone);
            imshow("edge detection", edge);
            Rect barcodeRect = findBarcodeContour(edge, 0.8);
            Mat barcodeRegion = extractBarcodeRegion(src, barcodeRect);
            imshow("barcode Region", barcodeRegion);
            Mat binaryBarcodeRegion = grayscaleToBinary(barcodeRegion);

            Mat middleRow(1, binaryBarcodeRegion.cols, CV_8UC1);
            for (int i = 0; i < binaryBarcodeRegion.cols; i++) {
                middleRow.at<uchar>(0, i) = binaryBarcodeRegion.at<uchar>(binaryBarcodeRegion.rows / 2, i);
            }
            vector<int> segments;
            saveConsecutiveSegments(middleRow, segments);

            std::cout << "Decoded barcode: " <<std::endl; 
            String decodedBarcode =processSegments(segments);
            std::cout << decodedBarcode << std::endl;
        }
       
        catch (const exception& e) {
            cerr << e.what() << endl;
        }

        waitKey(0);
    }
    return 0;
}
