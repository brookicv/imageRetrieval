#include <iostream>

#include <opencv2/opencv.hpp>
#include <string>
extern "C" {
    #include <vl/kmeans.h>
}

using namespace std;
using namespace cv;

void vlkmeans()  
{  
    int data_num = 10;  
    int data_dim = 2;  
    int k = 2;  
  
    float *data = new float[data_dim * data_num];  
      
    cout << "Points to clustering: " << endl;  
    for (int i = 0; i < data_num; i++)  
    {  
        data[i * data_dim] = (float)rand()/3.0;  
        data[i * data_dim + 1] = (float)rand()/3.0;  
        cout << data[i * data_dim] << "\t" << data[i * data_dim + 1] << endl;  
    }  
  
    float * init_centers = new float[data_dim * k];  
    cout << "Initial centers: " << endl;  
    for (int i = 0; i < k; i++)  
    {  
        init_centers[i * data_dim] = (float)rand()/3.0;  
        init_centers[i * data_dim + 1] = (float)rand()/3.0;  
        cout << init_centers[i * data_dim] << "\t" << init_centers[i * data_dim + 1] << endl;  
    }  

    VlKMeans * fkmeans = vl_kmeans_new(VL_TYPE_FLOAT, VlDistanceL2);  
    vl_kmeans_set_algorithm(fkmeans, VlKMeansElkan);  
      
    // vl_kmeans_init_centers_plus_plus(fkmeans, data, data_dim, data_num, k);  
    //vl_kmeans_set_centers(fkmeans, (void *)init_centers, data_dim, k);
    vl_kmeans_init_centers_with_rand_data(fkmeans,data,data_dim,data_num,k);
     vl_kmeans_cluster(fkmeans, data, data_dim, data_num, k);  
    // vl_kmeans_set_max_num_iterations(fkmeans, 100);  
    // vl_kmeans_refine_centers(fkmeans, data, data_num);  
    // vl_kmeans_cluster(fkmeans, data, data_dim, data_num, k);  
  
    const float * centers = (float *)vl_kmeans_get_centers(fkmeans);  
  
    cout << "Clustering Centers: " << endl;  
    for (int i = 0; i < k; i++)  
    {  
        cout << centers[i * data_dim] << "\t" << centers[i * data_dim + 1] << endl;  
    }  
}  

void opencv_kmeans()
{
    const string file = "/home/book/git/imageRetrieval/image/1.jpg";
    Mat img = imread(file);
    Mat float_img;
    img.convertTo(float_img,CV_32FC3);

    Mat label,data;
    for (int i = 0; i < img.rows;i++)
        for (int j = 0; j < img.cols; j++)
        {
            Vec3b point = img.at<Vec3b>(i, j);
            Mat tmp = (Mat_<float>(1, 3) << point[0], point[1], point[2]);
            data.push_back(tmp);
        }
    kmeans(data,3,label,TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 
        10, 1.0),3, KMEANS_RANDOM_CENTERS);

    Vec3b colorTab[] =
    {
        Vec3b(0, 0, 255),
        Vec3b(0, 255, 0),
        Vec3b(255, 100, 100),
        Vec3b(255, 0, 255),
        Vec3b(0, 255, 255)
    };

    int n = 0;
    //显示聚类结果，不同的类别用不同的颜色显示
    for (int i = 0; i < img.rows; i++)
    for (int j = 0; j < img.cols; j++)
    {
        int clusterIdx = label.at<int>(n);
        img.at<Vec3b>(i, j) = colorTab[clusterIdx];
        n++;
    }

    imshow("Cluster",img);
    waitKey();
}

void vl_kmeans_image()
{
    const string file = "/home/book/git/imageRetrieval/image/1.jpg";
    Mat img = imread(file);

    int dims = 3; // rows * cols * depth
    int data_num = img.rows * img.cols;

    float* data = new float[data_num * dims];
    float* data_inx = data;
    for(int i = 0 ; i < img.rows; i ++ ){
        Vec3b* ptr = img.ptr<Vec3b>(i);
        for(int j = 0; j < img.cols; j ++) {
            *data_inx = ptr[j][0];
            *(data_inx+1) = ptr[j][1];
            *(data_inx+2) = ptr[j][2];
            data_inx += 3;
        }
    }

    VlKMeans* vl_kmeans = vl_kmeans_new(VL_TYPE_FLOAT,VlDistanceL2);
    vl_kmeans_set_algorithm(vl_kmeans,VlKMeansLloyd);

    int k = 5;
    vl_kmeans_init_centers_with_rand_data(vl_kmeans,data,dims,data_num,k);
        // Run at most 100 iterations of cluster refinement using Lloyd algorithm
    //vl_kmeans_set_max_num_iterations(vl_kmeans,100);
    //vl_kmeans_refine_centers(vl_kmeans,img.data,data_num);
    vl_kmeans_cluster(vl_kmeans,data,dims,data_num,k);

    float* center = new float[k * dims];
    center = (float*)vl_kmeans_get_centers(vl_kmeans);

    for(int i = 0 ; i < k ; i ++) {
        for(int j = 0; j < dims; j ++) {
            cout << *(center + i + j*dims) << endl; 
        }
    }
}
int main(){
    
    vl_kmeans_image();
    return 0;
}