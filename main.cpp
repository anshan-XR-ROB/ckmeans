
#include "ckmeans.h"

using Eigen::Matrix;
using Eigen::MatrixXf;
using Eigen::MatrixXi;

int main(int argc, char* argv[])
{
  if(argc < 6){
    printf("Usage: ./ckmeans_train feature_file_name feature_num num_bits num_dim output_file_dir\n");
    return 0;
  }
  const char *feature_file_name = argv[1];
  int feature_num = atoi(argv[2]);
  int num_bits = atoi(argv[3]);
  int num_dim = atoi(argv[4]);
  char *output_file_dir = argv[5];

  CKmeansModel ckmeans;

  printf("Loading Train Data!\n");
  MatrixXf trainmat = MatrixXf::Zero(feature_num, num_dim);
  ckmeans.loadTrainData(feature_file_name, feature_num, num_dim, trainmat);
  printf("Load Train Data ready!\n");

  int npca = 0;
  int m_ckmeans = num_bits/8;
  printf("Start Training!\n");
  ckmeans.train(trainmat, m_ckmeans, num_bits, npca, output_file_dir);
  printf("Train Ready!\n");

	return 0;
}

