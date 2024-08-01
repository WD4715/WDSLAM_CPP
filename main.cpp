#include <opencv2/opencv.hpp>
#include <iostream>
#include <pangolin/pangolin.h>
#include <GL/gl.h> // Make sure to include the OpenGL header

void drawPlane(int num_divs = 200, float div_size = 10.0f) {
    float minx = -num_divs * div_size;
    float minz = -num_divs * div_size;
    float maxx = num_divs * div_size;
    float maxz = num_divs * div_size;

    glColor3f(0.7f, 0.7f, 0.7f);
    glBegin(GL_LINES);
    for (int n = 0; n <= 2 * num_divs; ++n) {
        glVertex3f(minx + div_size * n, 0, minz);
        glVertex3f(minx + div_size * n, 0, maxz);
        glVertex3f(minx, 0, minz + div_size * n);
        glVertex3f(maxx, 0, minz + div_size * n);
    }
    glEnd();
}

// Function to convert cv::Mat to pangolin::OpenGlMatrix
void ConvertToPangolinMatrix(const cv::Mat& cvMat, pangolin::OpenGlMatrix& pgMat) {
    if (cvMat.rows != 4 || cvMat.cols != 4 || cvMat.type() != CV_64F) {
        throw std::runtime_error("Invalid matrix size or type.");
    }

    // Directly copy data
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            pgMat.m[j*4 + i] = cvMat.at<double>(i, j);  // Transpose to column-major order
        }
    }
}

void DrawCurrentCamera(const pangolin::OpenGlMatrix &Twc, float r, float g, float b)
{
    const float w = 1;
    const float h = w*0.75;
    const float z = w*0.6;

    glPushMatrix();
    glMultMatrixd(Twc.m); // Apply the transformation matrix.

    glLineWidth(1);
    glColor3f(r, g, b); // Set the color as per the parameters.
    glBegin(GL_LINES);
    glVertex3f(0, 0, 0);
    glVertex3f(w, h, z);
    glVertex3f(0, 0, 0);
    glVertex3f(w, -h, z);
    glVertex3f(0, 0, 0);
    glVertex3f(-w, -h, z);
    glVertex3f(0, 0, 0);
    glVertex3f(-w, h, z);

    glVertex3f(w, h, z);
    glVertex3f(w, -h, z);

    glVertex3f(-w, h, z);
    glVertex3f(-w, -h, z);

    glVertex3f(-w, h, z);
    glVertex3f(w, h, z);

    glVertex3f(-w, -h, z);
    glVertex3f(w, -h, z);
    glEnd();

    glPopMatrix();
}

void DrawFrames(const std::vector<pangolin::OpenGlMatrix> &Poses) {
    for (size_t i = 0; i < Poses.size(); ++i) {
        if (i == Poses.size() - 1) {
            // Draw the last pose in red
            DrawCurrentCamera(Poses[i], 1.0f, 0.0f, 0.0f);
        } else {
            // Draw other poses in green
            DrawCurrentCamera(Poses[i], 0.0f, 1.0f, 0.0f);
        }
    }
}


int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <Video_Path>\n";
        return 1;
    }

    cv::VideoCapture capture(argv[1]);
    if (!capture.isOpened()) {
        std::cerr << "Error opening video file\n";
        return 1;
    }



    // Feature Extractor
    const static auto& orb = cv::ORB::create(3000);
    int img_idx = 0;

    
    std::vector<cv::Point2f> predicted_kpts, previous_kpts;
    std::vector<cv::Point2f> good_curr_kpts, good_prev_kpts; // KeyPoints
    cv::Mat previous_frame;
    
    float fx = 7.188560000000e+02;
    float cx = 6.071928000000e+02;
    float fy = 7.188560000000e+02;
    float cy = 1.852157000000e+02;

    cv::Mat K = (cv::Mat_<double>(3,3) << 7.188560000000e+02, 0, 6.071928000000e+02, 0, 7.188560000000e+02, 1.852157000000e+02, 0, 0, 1);


    // R, t를 저장할 Mat 객체
    cv::Mat R = cv::Mat::eye(3, 3, CV_64F);  // 3x3 단위 행렬 생성, 데이터 타입은 double
    cv::Mat t = cv::Mat::zeros(3, 1, CV_64F); // 3x1 영 벡터 생성, 데이터 타입은 double
    cv::Mat T = cv::Mat::eye(4, 4, CV_64F);  // 3x3 단위 행렬 생성, 데이터 타입은 double
    cv::Mat prev_R, prev_t, prev_T;
    


    cv::Mat New_R= cv::Mat::eye(3, 3, CV_64F); 
    cv::Mat new_t= cv::Mat::zeros(3, 1, CV_64F);


    // Pangolin Visualization
    // Window dimensions
    const int width = 1024, height = 1024;
    const int kUiWidth = 180; // Width of the UI panel

    // Create and bind a named OpenGL window
    pangolin::CreateWindowAndBind("Map Viewer", width, height);
    glEnable(GL_DEPTH_TEST);

    // Camera setup
    float viewpoint_x = 0, viewpoint_y = -40, viewpoint_z = -80;
    float viewpoint_f = 1000;
    pangolin::OpenGlMatrix proj = pangolin::ProjectionMatrix(width, height, viewpoint_f, viewpoint_f, width / 2, height / 2, 0.1, 5000);
    pangolin::OpenGlMatrix look_view = pangolin::ModelViewLookAt(viewpoint_x, viewpoint_y, viewpoint_z, 0, 0, 0, pangolin::AxisNegY);

    pangolin::OpenGlRenderState scam(proj, look_view);

    // Create Interactive View in window
    pangolin::View& dcam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(kUiWidth), 1.0, -(float)width/height)
        .SetHandler(new pangolin::Handler3D(scam));

    // Panel for UI elements
    pangolin::CreatePanel("ui").SetBounds(1.0, 0.0, 0.0, pangolin::Attach::Pix(kUiWidth));

    // UI variables
    pangolin::Var<bool> checkboxFollow("ui.Follow", true, true);
    pangolin::Var<bool> checkboxCams("ui.Draw Cameras", true, true);
    pangolin::Var<bool> checkboxCovisibility("ui.Draw Covisibility", true, true);
    pangolin::Var<bool> checkboxSpanningTree("ui.Draw Tree", true, true);
    pangolin::Var<bool> checkboxGrid("ui.Grid", true, true);
    pangolin::Var<bool> checkboxPause("ui.Pause", false, true);
    pangolin::Var<int> int_slider("ui.Point Size", 2, 1, 10);



    // Camear Pose list
    std::vector<pangolin::OpenGlMatrix> Poses;


    while (!pangolin::ShouldQuit()) {

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(1.0, 1.0, 1.0, 1.0);

        dcam.Activate(scam);

        if (checkboxGrid) {
            drawPlane();
        }

        cv::Mat frame, viz;
        capture >> frame;

        frame.copyTo(viz);

        if (frame.empty()) {
            break;  // When no more frames, exit the loop
        }

        if (good_curr_kpts.size() < 1000){

            // std::vector<cv::Point2f> good_curr_kpts;
            std::cout << "KeyFrame pops up : " << img_idx << std::endl;
            cv::Mat curr_descs;
            std::vector<cv::KeyPoint> curr_kpts;
            // Visualization for the Feature Extractuib
            orb->detectAndCompute(frame, cv::noArray(), curr_kpts, curr_descs);
            // 특징점의 위치만 추출하여 Point2f 벡터로 변환
            
            good_curr_kpts.clear(); // 좋은 키포인트 벡터를 클리어
            for (const auto& kp : curr_kpts) {
                good_curr_kpts.push_back(kp.pt);
            }


        } else {
            std::vector<cv::Point2f> predicted_kpts; // KeyPoints
            std::vector<uchar> status; // Status
            std::vector<float> err; // error for the OpticalFlow
            
            
            cv::calcOpticalFlowPyrLK(previous_frame, frame, previous_kpts, predicted_kpts, status, err);

            good_curr_kpts.clear(); // 좋은 키포인트 벡터를 클리어
            good_prev_kpts.clear();
            for (int i=0; i < status.size(); i++){
                if (status[i] == 1){
                    good_curr_kpts.push_back(predicted_kpts[i]);
                    good_prev_kpts.push_back(previous_kpts[i]);
                }
            }

            // Finding Essential Matrix
            std::vector<cv::Point2f> norm_prev_kpts, norm_curr_kpts;
            for (auto& kp : good_curr_kpts) {
                norm_curr_kpts.push_back(cv::Point2f((kp.x - cx) / fx, (kp.y - cy) / fy));
            }
            for (auto& kp : good_prev_kpts) {
                norm_prev_kpts.push_back(cv::Point2f((kp.x - cx) / fx, (kp.y - cy) / fy));
            }

            cv::Mat essential_matrix, mask;
            // // 매개변수: 점 집합, 초점 거리(focal), 주점(pp), 메서드(method), 확률(prob), 임계값(threshold), 마스크(mask)
            essential_matrix = cv::findEssentialMat(norm_curr_kpts, norm_prev_kpts, 1.0, cv::Point2d(0, 0), cv::RANSAC, 0.999, 0.003, mask);

            std::vector<cv::Point2f> good_curr_kpts_,good_prev_kpts_;  
            for (size_t i = 0; i < mask.rows; i++) {
                if (mask.at<uchar>(i, 0) == 1) {  // mask가 1인 경우만 good vectors에 추가
                    good_curr_kpts_.push_back(good_curr_kpts[i]);
                    good_prev_kpts_.push_back(good_prev_kpts[i]);
                }
            }

            // Calculate the pose
            int inliers = cv::recoverPose(essential_matrix, norm_curr_kpts, norm_prev_kpts, R, t);
            
            // ######################################
            // ######################################
            // ######################################
            // Triangulation Process
            cv::Mat previous_pose =cv::Mat::eye(3, 4, CV_64F);
            cv::Mat current_pose = cv::Mat::eye(3, 4, CV_64F);  
            cv::Mat pts4D;
            
            R.copyTo(current_pose(cv::Range(0, 3), cv::Range(0, 3)));
            // Copy new_t into the first three rows of the fourth column of T
            for (int i = 0; i < 3; i++) {
                current_pose.at<double>(i, 3) = t.at<double>(i, 0);
            }
 
            cv::triangulatePoints(previous_pose, current_pose, norm_curr_kpts, norm_prev_kpts, pts4D);

            // Homegenious -> inHomogenious
            for (int i = 0; i < pts4D.cols; ++i) {
                pts4D.col(i) /= pts4D.at<float>(3, i);
            }
            
            cv::Mat pts4DConverted;
            pts4D.convertTo(pts4DConverted, CV_64F);
            
            // to current camera pose
            // 3D 포인트 필터링 및 출력
            std::vector<cv::Point3f> keypoints_;
            for (int i = 0; i < pts4DConverted.cols; ++i) {
                cv::Mat p = pts4DConverted.col(i).rowRange(0, 4).clone();  // 동차 좌표를 4x1 행렬로 추출
                // p.at<double>(3, 0) = 1.0;  // 동차 좌표로 만듦

                cv::Mat pl1 = previous_pose * p;  // 이전 포즈와의 행렬 곱
                cv::Mat pl2 = current_pose * p;  // 현재 포즈와의 행렬 곱

                // 점이 두 카메라의 앞에 있는지 확인
                if (pl1.at<double>(2, 0) < 0 || pl2.at<double>(2, 0) < 0) {
                    continue;
                }

                cv::Point3f pt3d(p.at<double>(0, 0), p.at<double>(1, 0), p.at<double>(2, 0));
                keypoints_.push_back(pt3d);
            }
            



            // ######################################
            // ######################################
            // ######################################




            New_R = prev_R * R;
            new_t = prev_t + prev_R * t;
            
            New_R.copyTo(T(cv::Range(0, 3), cv::Range(0, 3)));
            // Copy new_t into the first three rows of the fourth column of T
            for (int i = 0; i < 3; i++) {
                T.at<double>(i, 3) = new_t.at<double>(i, 0);
            }

            // Declare a Pangolin matrix
            pangolin::OpenGlMatrix pangolinMatrix;
            ConvertToPangolinMatrix(T, pangolinMatrix);
            Poses.push_back(pangolinMatrix);

            DrawFrames(Poses);
            

            std::cout << "Check for the current Camera Pose : \n" << T << std::endl;  


            // ######################################
            // ######################################
            // ######################################
            glPointSize(2.0f);
            glBegin(GL_POINTS);
            for (size_t i = 0; i < keypoints_.size(); ++i) {
                cv::Point3f pt = keypoints_[i];
                cv::Mat p = (cv::Mat_<double>(4, 1) << pt.x, pt.y, pt.z, 1.0);  // 동차 좌표를 4x1 행렬로 추출

                cv::Mat pl1 = T * p;  // 이전 포즈와의 행렬 곱

                // 점이 두 카메라의 앞에 있는지 확인
                if (pl1.at<double>(2, 0) < 0) {
                    continue
                    ;
                } else{
                    cv::Point3f pl1_point(pl1.at<double>(0, 0), pl1.at<double>(1, 0), pl1.at<double>(2, 0));
                    glColor3f(1.0f, 0.0f, 0.0f); // Red color for points
                    glVertex3f(pl1_point.x, pl1_point.y, pl1_point.z);
                }
                

                
            }
            glEnd();

            // ######################################
            // ######################################
            // ######################################

            



            
            
            // Visualization for the Feature Tracking using RANSAC
            for (int i = 0; i < good_curr_kpts_.size(); i++){
                cv::circle(viz, good_curr_kpts_[i], 1, cv::Scalar(255, 0, 0), 1);
                cv::arrowedLine(viz, good_curr_kpts_[i], good_prev_kpts_[i], cv::Scalar(0, 255, 0));
            }
            cv::imshow("Visualization for the Tracking with RANSAC : ", viz);
            
            

        }
        pangolin::FinishFrame();
        
        if (cv::waitKey(25) >= 0) {
            break;  // Stop if any key is pressed
        }
    
    img_idx += 1;
    previous_kpts = good_curr_kpts;
    previous_frame = frame;

    prev_R = New_R;
    prev_t = new_t;
    


    }

    capture.release();
    cv::destroyAllWindows();
    return 0;
}

