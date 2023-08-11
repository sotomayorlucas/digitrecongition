#include "funciones.h"
#include <ctime>    


int main(){
    bool PCA;
    cout << "Escriba 0 para correr sin PCA, cualquier otro para correr con PCA" <<endl;
    cin >> PCA;
    int k;
    cout << "Elija un valor de k (cantidad de vecinos)" << endl;
    cin >> k;
    uint niter = 5000;
    float eps = 1e-10;
    string csvTrain;
    string csvTest;
    cout << "Escriba el nombre del archivo train. Debe estar en la carpeta test" << endl;
    cin >> csvTrain;
    cout << "Escriba el nombre del archivo test. Debe estar en la carpeta test" << endl;
    cin >> csvTest;
    string path = "../test/";
    ifstream dataTrain("../test/" + csvTrain);
    ifstream dataTest ("../test/" + csvTest);
    vector<vector<double>> matTrain = leerCSV(dataTrain);
    vector<vector<double>> matTest = leerCSV(dataTest);
    pair<MatrixXd, VectorXd> mateigenTrain = pasarEigenTrain(matTrain);
    MatrixXd mateigenTest = pasarEigenTest(matTest);
    vector<int> res;
    unsigned t0 = clock();
    if (!PCA){
        res = correrKnn(mateigenTest, mateigenTrain, k);
    } else {
        uint alfas;
        cout << "Elija valor de alfa"<<endl;
        cin >> alfas;
        res = correrPCA(mateigenTest,mateigenTrain,k,eps,niter,alfas);
    }
    cout << "El archivo de salida se llamará resultados.csv y estará en la carpeta test" <<endl;
    ofstream fout;
    fout.open("../test/resultados.csv");
    unsigned t1 = clock();
    double time = (double(t1-t0)/CLOCKS_PER_SEC);
    cout << "Tiempo de ejecución: " <<time<< " Segundos" << endl;
    fprintVector(res, fout);
    fout.close();
    return 0;
}
