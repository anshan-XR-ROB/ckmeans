#include "ckmeans.h"

CKmeansModel::CKmeansModel(void)
{
}
CKmeansModel::~CKmeansModel(void)
{
}

void CKmeansModel::loadTrainData(const char *datafile, int num_points, 
                                int dim, MatrixXf &dataMat)
{
  int i = 0;
  int j = 0;
  FILE *pfile = fopen(datafile, "r");

  char imagename[200];
  char fTemp[100];
  for(i = 0; i < num_points; i++)
  {
    if (i % 100 == 0)
      printf("Load data %d\n", i+1);
    fscanf(pfile, "%s", imagename);
    for (j = 0; j < dim; j++)
    {
      fscanf(pfile, "%s", fTemp);
      char *p;
      p = strtok(fTemp, ":");
      p = strtok(NULL, ":");
      float value = atof(p);
      dataMat(i,j) = value;
    }
  }
  fclose(pfile);
}

void CKmeansModel::loadData(const char *datafile, int num_points, 
                            int dim, MatrixXf &dataMat)
{
  int i = 0;
  int j = 0;
  FILE *pfile = fopen(datafile, "r");

  char imagename[200];
  char fTemp[100];
  for(i = 0; i < num_points; i++)
  {
    if (i % 100 == 0)
      printf("Load data %d\n", i+1);
    fscanf(pfile, "%s", imagename);
    for (j = 0; j < dim; j++)
    {
      fscanf(pfile, "%s", fTemp);
      char *p;
      p = strtok(fTemp, ":");
      p = strtok(NULL, ":");
      float value = atof(p);
      dataMat(j,i) = value;
    }
  }
  fclose(pfile);
}

void CKmeansModel::train(MatrixXf &dataMat, int m_ckmeans, int nbits, int &npca, char *save_dir)
{
  int subspace_bits = 8;
  int niter = 100;
  int default_npca_ck = max(384, nbits * 2);
  npca = min(default_npca_ck - fmod((double)default_npca_ck, (double)m_ckmeans),
             dataMat.cols() - fmod((double)dataMat.cols(), (double)m_ckmeans));

  MatrixXf mu = dataMat.colwise().mean();
  MatrixXf tempA = MatrixXf::Zero(dataMat.rows(), dataMat.cols());
  for (int i = 0; i < dataMat.rows(); i++)
    for (int j = 0; j < dataMat.cols(); j++)
      tempA(i,j) = dataMat(i,j) - mu(0,j);
  printf("Compute mean ready!\n");
  //PCA
  MatrixXd pc;
  if (npca == dataMat.cols()){
    pc = MatrixXd::Zero(dataMat.cols(), dataMat.cols());
    pc.setIdentity();
  }
  else{
    MatrixXf covMat = MatrixXf::Zero(tempA.cols(), tempA.cols());
    covMat = tempA.transpose() * tempA / (float) (tempA.rows());
    MatrixXd covd = covMat.cast <double> (); 
    pc.resize(tempA.cols(), npca);
    //RedSVD::RedSymEigen<MatrixXd> es(covd);
    //RedSVD::RedSVD<MatrixXd> svd(covd);
    //JacobiSVD<MatrixXd> svd(covd, ComputeThinU | ComputeThinV);
    //MatrixXd U = svd.matrixU();
    //MatrixXd V = svd.matrixV(); 
    //printf("SVD OK\n");
    printf("Start compute eigen value!\n");
    EigenSolver<MatrixXd> es(covd); //
    VectorXd eigenValues = es.eigenvalues().real();
    MatrixXd eigenVectors = es.eigenvectors().real();	
    printf("Eigen value has been computed!\n");
    
    typedef std::pair<double, int> eigen_pair;
    std::vector<eigen_pair> ep;	
    for (int i = 0 ; i < tempA.cols(); ++i) 
    {
      ep.push_back(std::make_pair(eigenValues(i), i));
    }
    sort(ep.begin(), ep.end()); // Ascending order by default
    // Sort them all in descending order
    MatrixXd eigenVectors_sorted = MatrixXd::Zero(eigenVectors.rows(), eigenVectors.cols());
    VectorXd eigenValues_sorted = VectorXd::Zero(tempA.cols());
    int colnum = 0;
    int q = (int)ep.size() - 1;
    for (; q > -1; q--) 
    {
      eigenValues_sorted(colnum) = ep[q].first;
      eigenVectors_sorted.col(colnum++) += eigenVectors.col(ep[q].second);
    }
    for(int i = 0; i < npca; i++)
    {
      pc.col(i) = eigenVectors_sorted.col(i);
    }
  }
  MatrixXf pcf = pc.cast <float> (); 
  MatrixXf trdata2 = pcf.transpose() * tempA.transpose();
  /*MatrixXf trdata2 = MatrixXf::Zero(npca, dataMat.rows());
  loadMat("F:\\topics\\indexing\\PQ\\ckmeans-master\\trdata2.txt", trdata2);
  printf("load ok!\n");*/
  printf("PCA compute ready!\n");
  trainCKmeans(trdata2, pcf, mu, m_ckmeans, pow(2, subspace_bits), niter, save_dir);
}

void CKmeansModel::trainCKmeans(MatrixXf &traindata, MatrixXf &pc, MatrixXf &mu, 
                                int num_subspaces, int num_centers_per_subspace, 
                                int num_iter, char *save_dir)
{
  int cols = traindata.cols();
  int rows = traindata.rows();
  MatrixXi centersMat = MatrixXi::Ones(num_subspaces,1) * num_centers_per_subspace;
  int nbits = 0;
  for (int i = 0; i < num_subspaces; i++)
    nbits += log(centersMat(i,0))/log(2);
  MatrixXi lengthMat = MatrixXi::Ones(num_subspaces,1) * floor(rows/num_subspaces);
  int plusnum = (int)fmod((double)rows, (double)num_subspaces);
  for (int i = 0; i < plusnum; i++)
    lengthMat(i,0) += 1;

  MatrixXi lengthMat0 = MatrixXi::Zero(num_subspaces,1);
  for (int i = 0; i < num_subspaces-1; i++)
    lengthMat0(i+1, 0) = lengthMat0(i, 0) + lengthMat(i, 0);
  for (int i = 0; i < num_subspaces; i++)
    lengthMat0(i, 0) = lengthMat0(i, 0) + 1;

  MatrixXi lengthMat1 = MatrixXi::Zero(num_subspaces,1);
  lengthMat1(0, 0) = lengthMat(0, 0);
  for (int i = 1; i < num_subspaces; i++)
    lengthMat1(i, 0) = lengthMat1(i-1, 0) + lengthMat(i, 0);

  int step = num_subspaces;
  MatrixXi a = MatrixXi::Zero(1, rows);
  for (int i = 0; i < rows; i++)
    a(0, i) = 1 + fmod((double)(i * step), double(rows-1));
  a(0, rows-1) = rows;
  MatrixXf Rt = MatrixXf::Zero(rows, rows);
  for (int i = 0; i < rows; i++)
    Rt(i, a(0,i)-1) = 1;
  MatrixXf R = Rt.transpose();
  MatrixXf RX = Rt * traindata;

  //inializing D by random selection of subspace centers (after rotation)
  printf("Inializing D!\n");
  srand((unsigned)time(NULL));
  vector<MatrixXf> D;
  D.reserve(num_subspaces);
  //FILE *fp = fopen("F:\\topics\\indexing\\PQ\\ckmeans-master\\perm.txt", "r");
  for (int i = 0; i < num_subspaces; i++){ 
    vector<int> perm;
    perm.resize(centersMat(i,0));
    for (int j = 0; j < centersMat(i,0); j++){
      perm[j] = rand() % cols;
      /*int temp = 0;
      fscanf(fp, "%d", &temp);
      perm[j] = temp - 1;*/
    }
    MatrixXf Dtemp = MatrixXf::Zero(lengthMat1(i, 0) - lengthMat0(i, 0) + 1, 
                     centersMat(i,0));
    for (int j = 0; j < lengthMat1(i, 0) - lengthMat0(i, 0) + 1; j++)
      for (int m = 0; m < centersMat(i,0); m++)
        Dtemp(j, m) = RX(j + lengthMat0(i, 0) - 1, perm[m]);
    D.push_back(Dtemp);
  }
  //fclose(fp);
  printf("Compute DB!\n");
  MatrixXf DB = MatrixXf::Zero(rows,cols);
  MatrixXi Btemp = MatrixXi::Zero(cols, 1);
  vector<MatrixXi> B;
  B.reserve(num_subspaces);
  for (int i = 0; i < num_subspaces; i++){
    MatrixXf RXtemp = RX.block(lengthMat0(i, 0) - 1, 0,
                      lengthMat1(i, 0) - lengthMat0(i, 0) + 1, RX.cols());
    euclideanNN(D[i], RXtemp, Btemp);
    B.push_back(Btemp);
    for (int m = lengthMat0(i, 0)-1; m < lengthMat1(i, 0); m++)
      for (int n = 0; n < cols; n++)
        DB(m, n) = D[i](m-lengthMat0(i, 0)+1, Btemp(n, 0));
  }

  //Iteration
  printf("Start iteration!\n");
  double objerror = DBL_MAX;
  double objerror1 = 0.0;
  double objlast = 0.0;
  for (int i = 0; i < num_iter; i++){
    printf("Iteration: %d\n", i+1);
    if((int)fmod((double)i, 10.0) == 0){
      objlast = objerror;
      MatrixXf temp = R * DB;
      temp = temp - traindata;
      for (int m = 0; m < rows; m++)
        for (int n = 0; n < cols; n++)
          temp(m,n) = temp(m,n) * temp(m,n);
      MatrixXf tempsum = temp.colwise().sum();
      MatrixXd tempsumd = tempsum.cast <double> (); 
      objerror = tempsumd.mean();
      if (i == 0){
        objerror1 = objerror;
        printf("objerror1 = %f\n", objerror1);
      }
    }
    printf("objlast = %f, objerror = %f\n", objlast, objerror);
    if(objlast - objerror < objerror1 * 0.00001){
      printf("not enough improvement in the objective... breaking.\n");
      break;
    }
    //updata R
    JacobiSVD<MatrixXf> svd(traindata * DB.transpose(), ComputeThinU | ComputeThinV);
    //RedSVD::RedSVD<MatrixXf> svd(traindata * DB.transpose());
    MatrixXf U = svd.matrixU();
    MatrixXf V = svd.matrixV();
    R = U * V.transpose();
    printf("SVD ready!\n");
    //loadMat("F:\\topics\\indexing\\PQ\\ckmeans-master\\R.txt", R);
    //printf("load R ok!\n");   
    //updata R*X
    RX = R.transpose() * traindata;
    for(int j = 0; j < num_subspaces; j++){
      //updata D
      MatrixXf RXtemp = RX.block(lengthMat0(j, 0) - 1, 0,
                        lengthMat1(j, 0) - lengthMat0(j, 0) + 1, RX.cols());
      kmeansIter(RXtemp, B[j], centersMat(j,0), D[j]); 
      //updata B
      euclideanNN(D[j], RXtemp, B[j]);
      //updata D*B
      for (int m = lengthMat0(j, 0)-1; m < lengthMat1(j, 0); m++)
        for (int n = 0; n < cols; n++)
          DB(m, n) = D[j](m-lengthMat0(j, 0)+1, B[j](n, 0));
    }
    printf("Update B and DB ready!\n");
  }

  R = pc * R;
  R = R.transpose();
  MatrixXf Rmu = R * mu.transpose();
  for(int i = 0; i < num_subspaces; i++){
    for (int m = 0; m < D[i].rows(); m++)
      for (int n = 0; n < D[i].cols(); n++)
        D[i](m, n) = D[i](m, n) + Rmu(m - lengthMat0(i, 0) + 1, 0);
  }

  //save results
  char *base1 = "/lengthMat0.txt";
  char *base2 = "/lengthMat1.txt";
  char *base3 = "/R.txt";
  char *base4 = "/D.txt";
  char str1[200], str2[200], str3[200], str4[200];
  strcpy(str1, save_dir);
  strcat(str1, base1);
  strcpy(str2, save_dir);
  strcat(str2, base2);
  strcpy(str3, save_dir);
  strcat(str3, base3);
  strcpy(str4, save_dir);
  strcat(str4, base4);
  printMat(str1, lengthMat0);
  printMat(str2, lengthMat1);
  printMat(str3, R);
  printMat(str4, D);
}

void CKmeansModel::quantize(MatrixXf &dataMat, MatrixXf &R, vector<MatrixXf> &D,
                            MatrixXi &assignmentMat, int m_ckmeans, int npca)
{
  int p = dataMat.rows();
  int n = dataMat.cols();
  MatrixXi lengthMat0 = MatrixXi::Zero(m_ckmeans, 1);
  MatrixXi lengthMat1 = MatrixXi::Zero(m_ckmeans, 1);
  loadMat("lengthMat0.txt", lengthMat0);
  loadMat("lengthMat1.txt", lengthMat1);

  for(int i = 0; i < m_ckmeans; i++){
    MatrixXi ind = MatrixXi::Zero(n, 1);
    MatrixXf Rtemp = R.block(lengthMat0(i, 0) - 1, 0,
                      lengthMat1(i, 0) - lengthMat0(i, 0) + 1, R.cols());
    MatrixXf RXtemp = Rtemp * dataMat;
    euclideanNN(D[i], RXtemp, ind);
    for (int j = 0; j < n; j++)
      assignmentMat(i, j) = ind(j, 0);
  }
}

void CKmeansModel::linscanAQD(MatrixXi &assignmentMat, MatrixXf &transformMat, 
                              vector<MatrixXf> &D, MatrixXi &result, int num_database,
                              int num_bits, int num_result)
{
  int num_query = transformMat.cols(); //10000
  int dim1codes = assignmentMat.rows(); //128
  int dim1queries = transformMat.rows(); //2048
  int subdim = D[0].rows(); //16*256 //D.size = 128
  int B_over_8 = num_bits / 8; //128

  vector<float> dis_from_q(B_over_8 * 256);

  for (int i = 0; i < num_query; i++) {
    for (int k = 0; k < B_over_8; k++) {
      for (int r = 0; r < 256; r++) {
        int t = k * 256 + r;
        dis_from_q[t] = 0;
        for (int s = 0; s < subdim; s++)
          dis_from_q[t] += sqr(D[k](s, r) - transformMat(k * subdim + s, i));
      }
    }
    int buffer_size = (int)1e7;
    int from = 0;
    int npairs = min(num_database, buffer_size + num_result);
    pair<float, int> *pairs = new pair<float, int>[npairs];
    int p = 0;
    while (from < num_database) {
      int offset = 0;
      if (from > 0)
        offset = num_result;
      for (int j = 0 + offset;
        j < min(num_database, from + buffer_size + (num_result - offset)) - from + offset;
        j++, p++) {
          pairs[j].first = 0;
          for (int k = 0; k < B_over_8; k++)
            pairs[j].first += dis_from_q[k * 256 + assignmentMat(k, p)];
          pairs[j].second = j + from - offset;
      }
      from = min(num_database, from + buffer_size + (num_result - offset));
      partial_sort(pairs, pairs + num_result, pairs + npairs);
    }
    for (int j = 0; j < num_result; j++)
      result(i, j) = pairs[j].second + 1;
  }
}

void CKmeansModel::kmeansIter(MatrixXf &datapoints, MatrixXi &assignments, 
                              int &num_centers, MatrixXf &centermat)
{
  int d = datapoints.rows();
  int n = datapoints.cols();
  vector<float>  nCp(num_centers);

  for (int i = 0; i < n; i++){
    int c = assignments(i, 0);
    nCp[c]++;
    for (int j = 0; j < d; j++){
      centermat(j, c) += datapoints(j, i);
    }
  }
 
  for (int i = 0; i < num_centers; i++) {
    if(nCp[i] != 0){
      for (int j = 0; j < d; j++)
        centermat(j, i) /= nCp[i];
    }
  }
}

void CKmeansModel::euclideanNN(MatrixXf &database, MatrixXf &query, 
                               MatrixXi &result)
{
  int p = database.rows();
  int n = database.cols();
  int nq = query.cols();
  for (int i = 0; i < nq; i++){
    double min_dis = DBL_MAX;
    int best_ind = -1;
    for (int j = 0; j < n; j++){
      double dist = 0.0;
      for (int k = 0; k < p; k++){
        dist += (database(k, j) - query(k, i)) * 
                (database(k, j) - query(k, i));
      }
      if (dist < min_dis) {
        min_dis = dist;
        best_ind = j;
      }
    }
    result(i, 0) = best_ind;
  }
}

void CKmeansModel::printMat(char *filename, vector<MatrixXf> &matvec, bool line)
{
  FILE* fp = fopen(filename, "w");
  for (int k = 0; k < matvec.size(); k++)
  {
    for (int i = 0; i < matvec[k].rows(); i++)
    {
      for (int j = 0; j < matvec[k].cols(); j++)
      {
        fprintf(fp, "%f ", matvec[k](i, j));
      }
      if (line)
        fprintf(fp, "%\n");
    }
  }
  fclose(fp);
}

void CKmeansModel::printMat(char *filename, MatrixXf &mat, bool line)
{
  FILE* fp = fopen(filename, "w");
  for (int i = 0; i < mat.rows(); i++)
  {
    for (int j = 0; j < mat.cols(); j++)
    {
      fprintf(fp, "%f ", mat(i, j));
    }
    if (line)
      fprintf(fp, "\n");
  }
  fclose(fp);
}

void CKmeansModel::printMat(char *filename, MatrixXi &mat, bool line)
{
  FILE* fp = fopen(filename, "w");
  for (int i = 0; i < mat.rows(); i++)
  {
    for (int j = 0; j < mat.cols(); j++)
    {
      fprintf(fp, "%d ", mat(i, j));
    }
    if (line)
      fprintf(fp, "\n");
  }
  fclose(fp);
}

void CKmeansModel::loadMat(char *filename, MatrixXf &mat)
{
  FILE* fp = fopen(filename, "r");
  float value = 0.0;
  for (int i = 0; i < mat.rows(); i++)
  {
    for (int j = 0; j < mat.cols(); j++)
    {
      fscanf(fp, "%f", &value);
      mat(i,j) = value;
    }
  }
  fclose(fp);
}

void CKmeansModel::loadMat(char *filename, MatrixXi &mat)
{
  FILE* fp = fopen(filename, "r");
  int value = 0;
  for (int i = 0; i < mat.rows(); i++)
  {
    for (int j = 0; j < mat.cols(); j++)
    {
      fscanf(fp, "%d", &value);
      mat(i,j) = value;
    }
  }
  fclose(fp);
}

void CKmeansModel::loadMat(char *filename, vector<MatrixXf> &matvec, 
                           int rows, int cols)
{
  FILE* fp = fopen(filename, "r");
  float value = 0.0;
  for (int k = 0; k < matvec.size(); k++){
    for (int i = 0; i < rows; i++)
    {
      for (int j = 0; j < cols; j++)
      {
        fscanf(fp, "%f", &value);
        matvec[k](i,j) = value;
      }
    }
  }
  fclose(fp);
}

inline float CKmeansModel::sqr(float d)
{
  return d*d;
}
