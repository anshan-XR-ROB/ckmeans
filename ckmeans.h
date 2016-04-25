#ifndef __CKMEANS__
#define __CKMEANS__

#define EIGEN_USE_MKL_ALL
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include <math.h>
#include <stdio.h>
#include <algorithm>
#include <vector>
#include <utility>
#include <time.h>
#include <float.h>
#include "RedSVD.h"

using Eigen::Matrix;
using Eigen::MatrixXf;
using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::VectorXf;
using Eigen::VectorXd;
using Eigen::EigenSolver;
using Eigen::JacobiSVD;
using Eigen::ComputeThinU;
using Eigen::ComputeThinV;

using namespace std;

class CKmeansModel
{
public:
  CKmeansModel(void);
  ~CKmeansModel(void);

  void loadData(const char *datafile, int num_points, int dim, MatrixXf &dataMat);
  void loadTrainData(const char *datafile, int num_points, 
                     int dim, MatrixXf &dataMat);
  void train(MatrixXf &dataMat, int m_ckmeans, int nbits, int &npca, char *save_dir);
  void quantize(MatrixXf &dataMat, MatrixXf &R, vector<MatrixXf> &D, MatrixXi &assignmentMat,
                int m_ckmeans, int npca);
  void linscanAQD(MatrixXi &assignmentMat, MatrixXf &transformMat, vector<MatrixXf> &D, 
                  MatrixXi &result, int num_database, int num_bits, int num_result);
  void loadMat(char *filename, vector<MatrixXf> &matvec, int rows, int cols);
  void loadMat(char *filename, MatrixXf &mat);
  void printMat(char *filename, MatrixXf &mat, bool line = false);
  void printMat(char *filename, MatrixXi &mat, bool line = false);
 
private:
  void trainCKmeans(MatrixXf &traindata, MatrixXf &pc, MatrixXf &mu, int num_subspaces, 
                    int num_centers_per_subspace, int num_iter, char *save_dir);
  void kmeansIter(MatrixXf &datapoints, MatrixXi &assignments, 
                  int &num_centers, MatrixXf &centermat);
  void euclideanNN(MatrixXf &database, MatrixXf &query, MatrixXi &result);
  void printMat(char *filename, vector<MatrixXf> &matvec, bool line = false);
  void loadMat(char *filename, MatrixXi &mat);
  inline float sqr(float d);
};
#endif
