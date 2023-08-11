#include <stdio.h>
#include <ctime>
#include <vector>
#include <math.h>
#include <iostream>
#include <fstream>
#include <Eigen>
#include <random>
using namespace std;
using namespace Eigen;



//leer y printear archivos
vector<vector<double>> leerCSV(ifstream& data);
pair<MatrixXd, VectorXd> pasarEigenTrain (vector<vector<double>> &mat);
MatrixXd pasarEigenTest (vector<vector<double>> &mat);
void fprintVector (vector<int> v, ofstream& fout);


//knn
int knn(VectorXd& v, pair<MatrixXd, VectorXd>&tageados,int k);
vector<int> kminimos (vector<pair<double,int >> &tageados, uint k);
int minimo (vector<pair<double,int >> &tageados);
int moda (vector<int> v);
vector<int> correrKnn (MatrixXd &test, pair<MatrixXd,VectorXd>&tageados,int k);

//pca
pair<VectorXd, double>  metodoPotencia (MatrixXd &m, float eps, uint niter);
vector<pair<VectorXd, double>> potDeflacion (MatrixXd &m, uint cantalfas, float eps, uint niter);
MatrixXd crearMatrizCovarianza (MatrixXd& mat);
MatrixXd matCambioBase(MatrixXd& mat, float eps, uint niter, uint alfa);
vector<int> correrPCA (MatrixXd &test, pair<MatrixXd,VectorXd>&train,int k,float eps, uint niter, uint alfas);
