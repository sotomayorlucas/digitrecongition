#include "funciones.h"
vector<int> correrPCA (MatrixXd &test, pair<MatrixXd,VectorXd>&train,int k,float eps, uint niter, uint alfas){
    //Genero el cambio de base
     MatrixXd matCB = matCambioBase(train.first ,eps, niter,alfas);
    train.first = train.first * matCB;
    pair<MatrixXd, VectorXd> A (train.first,train.second);
    test = test * matCB;
    vector<int> res = correrKnn(test, A, k);
    return res;
}
pair<VectorXd, double>  metodoPotencia (MatrixXd &m, float eps, uint niter){
    
    int n = m.row(0).size();
    VectorXd v = VectorXd::Random(n); //N será el tamaño del vector;
    v.normalize(); //normaliza in-place
    double aval;
    VectorXd prev = v;
    for (uint i=0; (i < niter); i++){
        v = m * v;
        v.normalize();
        if (((prev - v ).norm()) < eps ) break;
        prev = v;
    }
    aval =( (v.transpose()) * (m * v))(0,0); 
    pair<VectorXd, double> res = make_pair(v, aval);
    return res;
}

MatrixXd matCambioBase(MatrixXd& mat, float eps, uint niter, uint alfa){
    MatrixXd matCov = crearMatrizCovarianza(mat);
    vector<pair<VectorXd, double>> vecval = potDeflacion(matCov, alfa, eps, niter);
    MatrixXd matAvecMx ( vecval[0].first.size(),alfa); 
    for (uint i = 0; i < alfa; i++){
        matAvecMx.col(i) = vecval[i].first;
    }
    return  matAvecMx; 
}


vector<pair<VectorXd, double>> potDeflacion (MatrixXd &m, uint cantalfas, float eps, uint niter) {
 int n = m.row(0).size();
 vector<pair<VectorXd, double>> res(n);
 for (uint i = 0; i < cantalfas; i++){
    pair<VectorXd, double> temp = metodoPotencia(m,eps,niter);
    res[i] = temp;
    MatrixXd mtemp = (temp.first * temp.first.transpose());
    m = m -  (mtemp * temp.second);
 }
 return res;   
}

vector<int> correrKnn (MatrixXd &test,pair<MatrixXd,VectorXd>&tageados,int k){
    vector<int> knnTags;
    for (uint i = 0; i < test.rows(); i++)
    {
        VectorXd v =  test.row(i);
        int tag = knn(v,tageados,k);
        knnTags.push_back(tag);
    }
    return knnTags;
}

int knn(VectorXd &v, pair<MatrixXd,VectorXd>&tageados,int k) {
    vector<pair<double, int>> normas(tageados.second.size());
    for (uint i = 0; i < tageados.second.size(); i++){
        VectorXd temp;
        temp = (tageados.first.row(i));
        temp -= v;
        normas[i] = make_pair(temp.norm(),tageados.second[i]);
    }
    vector<int> kmins = kminimos(normas, k);
    int res = moda(kmins);
    return res;
}


int moda (vector<int> v){
    int mod = v[0];
    int cantidadMax = 0;
    for (uint i = 0; i < v.size(); i++){
        int contador = 0;
        int elemento = v[i];
        long unsigned int j = i;
        while (j < v.size()){
            if (v[j] == elemento){
                contador++;  
            }
            j++;
        }
        if (cantidadMax < contador){
            mod = v[i];
            cantidadMax = contador;
        }
    }
    return mod;
}


vector<int> kminimos (vector<pair<double,int >> &tageados, uint k){
    vector<int> res(k);
    for (uint i = 0; i < k; i++){
        int index = minimo(tageados);
        res[i] = tageados[index].second;
        tageados.erase(tageados.begin()+index);
    }
    return res;
    }
int minimo (vector<pair<double,int >> &tageados){
    int min=0;
    for (uint i = 0; i <tageados.size(); i++){
        if(tageados[min].first > tageados[i].first){
            min = i;
        }
    }
    return min;
}
vector<vector<double>> leerCSV(ifstream& data){
    string line;
    vector<vector<double>> parsedCsv;
    getline(data,line);
    while(getline(data,line)){
        stringstream lineStream(line);
        string cell;
        vector<double> parsedRow;
        while(getline(lineStream,cell,',')){
            parsedRow.push_back(stod(cell));
        }
        parsedCsv.push_back(parsedRow);
    }
    return parsedCsv;
}

//Debido al no poder traducir el csv a matrix de eigens, debimos hacer un paso que implica hacer una doble copia
pair<MatrixXd, VectorXd> pasarEigenTrain (vector<vector<double>> &mat){

    MatrixXd mateig (mat.size(),mat[0].size()-1); 
    VectorXd tags (mat.size());
    for (uint i = 0; i < mat.size(); i++){
        for (uint j = 1; j < mat[0].size(); j++){
            mateig(i,j-1) = mat[i][j]; 
        }
        tags(i) = mat[i][0]; //coloco en cada iteracion un tag en el vector, y evito hacer otro for de paso
    }
    pair<MatrixXd, VectorXd>  res = make_pair(mateig,tags);

return res;
}

MatrixXd pasarEigenTest (vector<vector<double>> &mat){

    MatrixXd mateig (mat.size(),mat[0].size()); 
    VectorXd tags (mat.size());
    for (uint i = 0; i < mat.size(); i++){
        for (uint j = 0; j < mat[0].size(); j++){
            mateig(i,j) = mat[i][j]; 
        }
    }
return mateig;
}


MatrixXd crearMatrizCovarianza (MatrixXd& mat) {
    VectorXd muVect = mat.colwise().mean(); //creo vector mu
    //modifico la matriz para crear el X con la resta de los promedios
    Eigen::MatrixXd X = mat;
    for(int i = 0; i < X.cols(); i++) 
        X.col(i) -= Eigen::VectorXd::Constant(X.rows(), muVect[i]);
    Eigen::MatrixXd Mx = (X.transpose() * X) / (X.rows() - 1); 

    return Mx;
}


void fprintVector (vector<int> v, ofstream& fout){
    fout << "ImageId,Label"<<endl;
    for(uint i=0; i<v.size(); i++){
        fout<<i+1 << ","<<v[i]<<endl;
    }
    fout<<endl;
}
