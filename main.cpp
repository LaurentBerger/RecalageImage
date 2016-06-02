#include <opencv2/opencv.hpp> 
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;


map <int ,vector<Point > > arrayNeighbour;
vector<vector<Point> > contours;
vector<Vec4i> hierarchy;
int valRef=255;

vector<Point2d> z;
vector<Point2d> Z;

// Construction d'un tableau associatif avec en entrée le carré d'un distance et en sortie la liste des pixels situés à cette 
// distance. Les coordonnées des pixels sont coordonnées relatives : le pixel central est en coordonnée (0,0)
void InitMapDist(Mat src)
{
    for (int i=-src.rows/2;i<src.rows/2;i++)
        for (int j = -src.cols / 2; j < src.cols / 2; j++)
        {
            if (arrayNeighbour.find(i*i + j*j) == arrayNeighbour.end())
            {
                vector<Point> v;
                arrayNeighbour.insert(make_pair(i*i+j*j,v));
            }
            else
                arrayNeighbour[i*i + j*j].push_back(Point(i,j));
        }

}
//
// Recherche des nb plus proche voisins avec une valeur égale à valRef
//
int LookForNearest(Mat imgThresh, Mat src, int l, int c, int nb)
{
    int nbFound=0;
    vector<int> pixel;
    int dist=0;
    map<int,vector<Point> >::iterator it=arrayNeighbour.begin();
    float mean=0;
    while (nbFound < nb && it!=arrayNeighbour.end())
    {
        for (int i = 0; i < it->second.size(); i++)
        {
            Point p=Point(l,c)+it->second[i];
            if (p.x >= 0 && p.x < imgThresh.cols && p.y >= 0 && p.y < imgThresh.rows)
            {
                if (imgThresh.at<uchar>(p.y, p.x) == valRef)
                {
                    nbFound++;
                    mean+=src.at<uchar>(p.y, p.x);
                }

            }

        }
        it++;
    }
return mean/nbFound;
}

// Recherche du plus proche voisin (origine de la recherche en haut à gauche en cas d'égalité)
Point LookForNearestPos(Mat imgThresh,int l, int c)
{
    int nbFound=0;
    vector<int> pixel;
    int dist=1;
    map<int,vector<Point> >::iterator it=arrayNeighbour.begin();
    float mean=0;
    while (nbFound < 1 && it!=arrayNeighbour.end())
    {
        for (int i = 0; i < it->second.size(); i++)
        {
            Point p=Point(c,l)+it->second[i];
            if (p.x >= 0 && p.x < imgThresh.cols && p.y >= 0 && p.y < imgThresh.rows)
            {
                if (imgThresh.at<uchar>(p.y, p.x) == valRef)
                {
                    nbFound++;
                    return(Point(p));
                }

            }
        
        }
        it++;
    }
return Point(-1,-1);
}

// Recheche du pixel le plus proche de p dans un contour ctr
int LookForNearestPos(vector<Point> &ctr,Point p)
{
    int idx=-1;
    int dist=1000;
    for (int i = 0; i<ctr.size() && dist!=0;i++)
    {
        float d = norm(p-ctr[i]);
        if (d<dist)
        {
            dist=d;
            idx=i;
        }
    }
return idx;
}

//
// Calcul du laplacien des images tx et ty. Résultat dans lx et ly
void Laplacien(Mat mask, Mat tx, Mat ty, Mat &lx, Mat &ly)
{
    if (lx.rows==0)
        lx = Mat::zeros(tx.rows,tx.cols,CV_32FC1);
    if (ly.rows==0)
        ly = Mat::zeros(tx.rows,tx.cols,CV_32FC1);
    if (!lx.isContinuous())
    {
        cout << "PB mat is not continuous";
        return;
    }
    if (!ly.isContinuous())
    {
        cout << "PB mat is not continuous";
        return;
    }
    for (int i = 1; i < mask.rows - 1; i++)
    {   
        uchar *ptr = mask.ptr(i);
        float *txPtr = (float*)tx.ptr(i);
        float *tyPtr = (float*)ty.ptr(i);
        float *lxPtr = (float*)lx.ptr(i);
        float *lyPtr = (float*)ly.ptr(i);

        ptr++;
        txPtr++;
        tyPtr++;
        lxPtr++;
        lyPtr++;
        for (int j = 1; j < mask.cols - 1; j++)
        {
            if (*ptr == 255)
            {
                *lxPtr = *(txPtr - 1) + *(txPtr + 1) - 4 * (*txPtr) + *(txPtr+mask.cols)+*(txPtr-mask.cols);
                *lyPtr = *(tyPtr - 1) + *(tyPtr + 1) - 4 * (*tyPtr) + *(tyPtr+mask.cols)+*(tyPtr-mask.cols);
            }
            ptr++;
            txPtr++;
            tyPtr++;
            lxPtr++;
            lyPtr++;

        }
    }

}
//
//
void MatchSide(vector <Point> &newCtr,vector <Point> &ctrRec, Point p1, Point p2,int idx1,int idx2)
{
    float st=0;
    float s=0;

    // First side
    Point pRef1=p1;
    Point pRef2=p2;
    Point uRef=pRef2-pRef1; //
    int step=1;
    st=0;
    s=0;
	// Détermination du sens de parcours
    if (idx1<idx2)
        if(idx2-idx1<newCtr.size()/2)
            step=1;
        else
            step=-1;
    else if (idx1>idx2 && idx1-idx2<newCtr.size()/2)
        step =-1;
    else
        step=1;
    int ind=idx1+step;
    int pIdx=idx1;
    ctrRec[idx1]=p1;
	// Détermination de la longueur de la courbe entre pRef1 et pRef2
	// La longueur est dans st
    while( ind !=idx2 )
    {
        st += norm(newCtr[pIdx]-newCtr[ind]);
        pIdx=ind;
        ind+=step;
        if (ind>=static_cast<int>(newCtr.size()))
            ind=0;
        else if (ind<0)
            ind = newCtr.size()-1;

    }
    ind=idx1+step;
    pIdx=idx1;
	// A chaque point du contour entre pRef1 et pRef2 un point est associé sur le segment pref1 et pRef2
	// l'abscisse curviligne est constante des deux points est constante
	// Les points sont rangés dans ctrRec
    while( ind !=idx2 )
    {
        s += norm(newCtr[pIdx]-newCtr[ind]);
        Point p=pRef1+uRef*s/st;
        ctrRec[ind]=p;
        pIdx=ind;
        ind+=step;
        if (ind>=static_cast<int>(newCtr.size()))
            ind=0;
        else if (ind<0)
            ind = newCtr.size()-1;
    }
}


int main(int argc, char **argv)
{
Mat mTest,mThresh,mConnected;
// m  : orignal image grayscale
// mThresh : m threshold 
// idx : indiece of selected contours
// contours : mThresh contours
// mc : contours in an image
// approx : contours[idx] approximated with 4 corners
// rC: bounding rectangle of contours[idx]

Mat  m;
int idx=15;
int thresh=55;
int choix =7;
string nomFichier;
string nomDossier1("");
string nomDossier2("");


switch (choix){
case 0:
    nomFichier=nomDossier1+"plaqueDeforme3.jpg";
    m=imread(nomFichier,CV_LOAD_IMAGE_GRAYSCALE);
    idx=15;
    thresh=55;
    break;
case 1:
    nomFichier=nomDossier1+"VFgm6.jpg";
    m=imread(nomFichier,CV_LOAD_IMAGE_GRAYSCALE);
    idx=115;
    thresh=120;
    break;
case 2:
    nomFichier=nomDossier2+"echecTest1Paint.jpg";
    m=imread(nomFichier,CV_LOAD_IMAGE_GRAYSCALE);
    GaussianBlur(m,m,Size(5,5),1.0,1.0);
    idx=24;
    thresh=71;
    break;
case 3:
    nomFichier=nomDossier2+"echecTest2PaintR-22.jpg";
    m=imread(nomFichier,CV_LOAD_IMAGE_GRAYSCALE);
    GaussianBlur(m,m,Size(5,5),1.0,1.0);
    idx=6;
    thresh=83;
    break;
case 4:
    nomFichier=nomDossier2+"echecTest3PaintR-29.jpg";
    m=imread(nomFichier,CV_LOAD_IMAGE_GRAYSCALE);
    GaussianBlur(m,m,Size(5,5),1.0,1.0);
    idx=26;
    thresh=120;
    break;
case 5:
    nomFichier=nomDossier1+"YcRzN.jpg";
    m=imread(nomFichier,CV_LOAD_IMAGE_GRAYSCALE);
    GaussianBlur(m,m,Size(5,5),1.0,1.0);
    idx=0;
    thresh=30;
    break;
case 6:
    nomFichier=nomDossier1+"gpQlI.jpg";
    m=imread(nomFichier,CV_LOAD_IMAGE_GRAYSCALE);
    GaussianBlur(m,m,Size(5,5),1.0,1.0);
    idx=0;
    thresh=30;
    break;
case 7:
    nomFichier=nomDossier1+"5rmER.jpg";
    m=imread(nomFichier,CV_LOAD_IMAGE_GRAYSCALE);
    GaussianBlur(m,m,Size(5,5),1.0,1.0);
    idx=47;
    thresh=10;
    break;

}

Mat mc;
// Seuillage de l'image
threshold(m,mThresh,thresh,255,THRESH_BINARY);
// Recherche des contours
findContours(mThresh,contours,hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE, cv::Point(0,0));
mc = Mat::zeros(m.size(),CV_8UC3);
// Le contour idx est un contour de référence défini manuellement
drawContours(mc,contours,idx,Scalar(255,0,0),-1);
imshow("mc",mc);
vector<Point> approx;
// Recherche du plus petit rectangle R contenant le contour idx
Rect rC=boundingRect(contours[idx]);
vector<Point> c;
vector<vector<Point> > newCtr;
// Déplacement des points de contours (à voir?)
Point offset(rC.x-1,rC.y-1);
for (int i = 0; i<contours[idx].size();i++)
    c.push_back(contours[idx][i]-offset);
newCtr.push_back(c);
c.clear();
// mask d'une taille équivalent à R (+2 pixels)
Mat mask=Mat::zeros(rC.height+2, rC.width+2,CV_8UC1);
// Valeur de translation en x à résoudre à l'aide du laplacien 
Mat tx=Mat::zeros(rC.height+2, rC.width+2,CV_32FC1);
// Valeur de translation en y à résoudre à l'aide du laplacien 
Mat ty=Mat::zeros(rC.height+2, rC.width+2,CV_32FC1);
// Initialisation de la carte des distances.
InitMapDist(mask);
rC.x=1;rC.y=1;
mask=Mat::zeros(rC.height+2, rC.width+2,CV_8UC1);
// Contour plot of shape
drawContours(mask,newCtr,0,Scalar(255),1);

// Recherche des points les plus proches des sommets du rectangle contenu dans le masque
vector<Point> nearestRectToCtr;
vector<vector<Point>> ctrRec;
int nbPtCtr=0,nbPtRect=0;
Point pRect[] = { Point(rC.x,rC.y),Point(rC.x,rC.y+rC.height-1),Point(rC.x+rC.width-1,rC.y+rC.height-1),Point(rC.x+rC.width-1,rC.y) };
Point p1=LookForNearestPos(mask,pRect[0].y,pRect[0].x);
Point p2=LookForNearestPos(mask,pRect[1].y,pRect[1].x);
Point p3=LookForNearestPos(mask,pRect[2].y,pRect[2].x);
Point p4=LookForNearestPos(mask,pRect[3].y,pRect[3].x);
// longueur des cotés
float dp12 = norm(p1-p2);
float dp23 = norm(p2-p3);
float dp34 = norm(p3-p4);
float dp41 = norm(p4-p1);

// Recherche des index des points du contour les plus proches des sommets du rectangle 
int idx1 = LookForNearestPos(newCtr[0],p1);
int idx2 = LookForNearestPos(newCtr[0],p2);
int idx3 = LookForNearestPos(newCtr[0],p3);
int idx4 = LookForNearestPos(newCtr[0],p4);

ctrRec.resize(1);
ctrRec[0].resize(newCtr[0].size());


float st=0;
float s=0;

// First side
// A chaque point du contour est associé un point du rectangle afin de calculer un vecteur translation initial
ctrRec[0][idx1]=Point (rC.x,rC.y);
MatchSide(newCtr[0],ctrRec[0], pRect[0], pRect[1],idx1,idx2);
MatchSide(newCtr[0],ctrRec[0], pRect[1], pRect[2],idx2,idx3);
MatchSide(newCtr[0],ctrRec[0], pRect[2], pRect[3],idx3,idx4);
MatchSide(newCtr[0],ctrRec[0], pRect[3], pRect[0],idx4,idx1);



Mat zoomMask;
resize(mask, zoomMask, Size(0,0),2,2);
// La translation en X et Y à l'intérieur du contour est initialisé à 254
drawContours(tx,newCtr,0,Scalar(254),-1);
drawContours(ty,newCtr,0,Scalar(254),-1);
float mx=0,my=0;
int nbPt=0;
// Condition aux limites définies en utilisant les points du contour
for (int i = 0; i < newCtr[0].size(); i++)
{
    tx.at<float>(newCtr[0][i])=ctrRec[0][i].x-newCtr[0][i].x;
    ty.at<float>(newCtr[0][i])=ctrRec[0][i].y-newCtr[0][i].y;
}
//
// Vérification
// Les points du contours ne doivent plus être à la valeur 254 
for (int i = 0; i < newCtr[0].size(); i +=1)
{
    if (tx.at<float>(newCtr[0][i]) == 254)
        cout << "Pb at " << newCtr[0][i] << "\t"<<tx.at<float>(newCtr[0][i])<<"\n";
}
//
// Lissage du vecteur translation de la frontière en fonction des  translations des points voisins
//
for (int i = 0; i < newCtr[0].size(); i++)
{
    float mTx=0;
    float mTy=0;
    int l=1;
    for (int j = -l; j <= l; j++)
    {
        int p=i+j;
        if (p<0)
            p+=newCtr[0].size();
        p = p % newCtr[0].size();
        mTx += (l+1-abs(j))*tx.at<float>(newCtr[0][p]);
        mTy += (l+1-abs(j))*ty.at<float>(newCtr[0][p]);
    }
    tx.at<float>(newCtr[0][i]) = mTx / ((l+1)*(l+1));
    ty.at<float>(newCtr[0][i]) = mTy / ((l+1)*(l+1));
}
// Sauvegarde des valeurs frontières dans un fichier
ofstream fs("txty.txt");
for (int i = 0; i < newCtr[0].size(); i +=1)
{
        fs<<newCtr[0][i].x<<"\t"<<newCtr[0][i].y<<"\t" << tx.at<float>(newCtr[0][i]) << "\t" << ty.at<float>(newCtr[0][i]) << "\n";
}
imshow("mask",zoomMask);
imwrite("mask.png",mask);
waitKey();
// Remise à zéro du masque
mask=0*Mat::ones(rC.height+2, rC.width+2,CV_8UC1);
// Les points à l'intérieur du contour dans le masque sont mis à la valeur 255 (points à résoudre en utilisant le laplacien)
drawContours(mask,newCtr,0,Scalar(255),-1);
// Les points sur le contour dans le masque sont mis à 128 (point défini une condition au limite)
drawContours(mask,newCtr,0,Scalar(128),1);
imshow("mask",mask);
// Initialisation des points à l'intérieur de la surface aléatoire
for (int i=0;i<mask.rows;i++)
    for (int j = 0; j < mask.cols; j++)
    {
        if (mask.at<uchar>(i,j) ==255)
        {
            tx.at<float>(i,j)=float(rand())/RAND_MAX;
            ty.at<float>(i,j)=float(rand())/RAND_MAX;
        }
    }
// Résolution de l'équation de Laplace
Mat lx,ly;
double t=0,dx2=1;
double diffusivity=1000,dt=dx2/(4*diffusivity);
int iAfter=0;
Mat xSol;
Mat txc,tyc,mt;
{
    cv::FileStorage fsx("txini.yml", cv::FileStorage::WRITE);
	fsx<<"Image"<<tx;
	cv::FileStorage fsy("tyini.yml", cv::FileStorage::WRITE);
	fsy<<"Image"<<ty;
}
    m=imread(nomFichier,CV_LOAD_IMAGE_COLOR);
int nbIter=0;
while (t<200000*dt)
{
    Laplacien(mask,tx,ty,lx,ly);
    tx =tx +(diffusivity*dt)*lx;
    ty =ty +(diffusivity*dt)*ly;
    t=t+dt;
    nbIter++;
    if (nbIter % 100 == 0)
    {
        mt = Mat::zeros(m.size(),CV_8UC3);
        int nbPtErr=0;
        for (int i=0;i<mask.rows;i++)
            for (int j = 0; j < mask.cols; j++)
            {
                uchar v=mask.at<uchar>(i,j);
                if (mask.at<uchar>(i,j) >0)
                {
                    float offsetx = tx.at<float>(i,j),offsety=ty.at<float>(i,j);
                    int lig=i+offset.y+offsety*1,col=j+offset.x+offsetx*1;
                    if (lig>=0 && lig<mt.rows && col>=0 && col<mt.cols)
                        mt.at<Vec3b>(lig,col) = m.at<Vec3b>(i+offset.y,j+offset.x);
                    else 
                    {
                        nbPtErr++;
                    }
                }
            }
        imshow("RESULTAT FINAL",mt);
        cout<<nbPtErr<<"\t"<<nbIter<<"\n";
        waitKey(1);
    }
   
}
cv::FileStorage fsx("tx.yml", cv::FileStorage::WRITE);
fsx<<"Image"<<tx;
cv::FileStorage fsy("ty.yml", cv::FileStorage::WRITE);
fsy<<"Image"<<ty;
imwrite("result.jpg",mt);
waitKey();



return 0;
}




int main_ver1(int argc, char **argv)
{
Mat mTest,mThresh,mConnected;
// m  : orignal image grayscale
// mThresh : m threshold 
// idx : indiece of selected contours
// contours : mThresh contours
// mc : contours in an image
// approx : contours[idx] approximated with 4 corners
// rC: bounding rectangle of contours[idx]

//Mat  m=imread("C:/Users/Laurent.PC-LAURENT-VISI/Downloads/plaqueDeforme3.jpg",CV_LOAD_IMAGE_GRAYSCALE);
Mat  m=imread("YcRzN.jpg",CV_LOAD_IMAGE_GRAYSCALE);
//m=imread("C:/Users/Laurent.PC-LAURENT-VISI/Downloads/rectangle.png",CV_LOAD_IMAGE_GRAYSCALE);
Mat mc;
threshold(m,mThresh,55,255,THRESH_BINARY);
findContours(mThresh,contours,hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE, cv::Point(0,0));
// Image contour extracted in contours
imshow("Image",m);
mc = Mat::zeros(m.size(),CV_8UC3);
int idx=15;
// There is only one contour 0
drawContours(mc,contours,idx,Scalar(255,0,0),-1);
imshow("mc",mc);
vector<Point> approx;
// Looking for bounding Rect of contours idx
Rect rC=boundingRect(contours[idx]);
vector<Point> c;
vector<vector<Point> > newCtr;
// All contour points  are translated 
Point offset(rC.x-1,rC.y-1);
for (int i = 0; i<contours[idx].size();i++)
    c.push_back(contours[idx][i]-offset);
newCtr.push_back(c);
c.clear();

Mat mask=Mat::zeros(rC.height+2, rC.width+2,CV_8UC1);
Mat tx=Mat::zeros(rC.height+2, rC.width+2,CV_32FC1);
Mat ty=Mat::zeros(rC.height+2, rC.width+2,CV_32FC1);
InitMapDist(mask);
rC.x=1;rC.y=1;
//drawContours(mask,newCtr,newCtr.size()-1,Scalar(255),1);
rectangle(mask,rC,Scalar(255),1);
vector<vector<Point> > ctrRec;
// ctrRect contour point of bounding rect
findContours(mask,ctrRec,hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE, cv::Point(0,0));
mask=Mat::zeros(rC.height+2, rC.width+2,CV_8UC1);
// Contour plot of shape
drawContours(mask,newCtr,0,Scalar(255),1);
vector<Point> nearestRectToCtr;
int nbPtCtr=0,nbPtRect=0;
for (int i = 0; i < ctrRec[0].size(); i ++)
{
    Point p=LookForNearestPos(mask,ctrRec[0][i].y,ctrRec[0][i].x);
    if (p.x==-1)
        cout << " Error " << i << " "<<ctrRec[0][i]<<endl;
    else
        nbPtRect++;
    nearestRectToCtr.push_back(p);

}
mask=Mat::zeros(rC.height+2, rC.width+2,CV_8UC1);
// Contour plot of shape
drawContours(mask,ctrRec,0,Scalar(255),1);
vector<Point> nearestCtrToRect;
for (int i = 0; i < newCtr[0].size(); i ++)
{
    Point p=LookForNearestPos(mask,newCtr[0][i].y,newCtr[0][i].x);
    if (p.x==-1)
        cout << " Error " << i << " "<<newCtr[0][i]<<endl;
    else 
        nbPtCtr++;
    nearestCtrToRect.push_back(p);

}
cout << "Rectangle : " << nbPtRect << "/" << ctrRec[0].size()<<" to Ctr"<<endl;
cout << "Ctr : " << nbPtCtr << "/" << newCtr[0].size()<<" to Rect"<<endl;
mask=Mat::zeros(rC.height+2, rC.width+2,CV_8UC3);
drawContours(mask,newCtr,0,Scalar(255,255,255),1);
for (int i = 0; i < ctrRec[0].size(); i +=10)
{
    if (nearestRectToCtr[i].x!=-1)
        line(mask, ctrRec[0][i], nearestRectToCtr[i],Scalar(255,0,0),1);
}

for (int i = 0; i < newCtr[0].size(); i +=10)
{
    if (nearestCtrToRect[i].x!=-1)
        line(mask, newCtr[0][i], nearestCtrToRect[i],Scalar(0,0,255),1);
}

Mat zoomMask;
resize(mask, zoomMask, Size(0,0),2,2);
drawContours(tx,newCtr,0,Scalar(100),-1);
drawContours(ty,newCtr,0,Scalar(100),-1);
float mx=0,my=0;
int nbPt=0;

for (int i = 0; i < newCtr[0].size(); i +=1)
{
    if (nearestCtrToRect[i].x!=-1)
    {
        vector<Point>::iterator it;
        tx.at<float>(newCtr[0][i])=nearestCtrToRect[i].x-newCtr[0][i].x;
        ty.at<float>(newCtr[0][i])=nearestCtrToRect[i].y-newCtr[0][i].y;
        cout << "(tx,ty) " << tx.at<float>(newCtr[0][i]) << "\t" << ty.at<float>(newCtr[0][i]) << "\n";
    }

}

for (int i = 0; i < ctrRec[0].size(); i +=1)
{
    if (nearestRectToCtr[i].x != -1)
    {
        if (tx.at<float>(nearestRectToCtr[i]) != 100)
        {
            float v1=tx.at<float>(nearestRectToCtr[i]);
            float v2=ctrRec[0][i].x-nearestRectToCtr[i].x;
                tx.at<float>(nearestRectToCtr[i]) = v2;
            mx+=tx.at<float>(nearestRectToCtr[i]);
            v1=ty.at<float>(nearestRectToCtr[i]);
            v2=ctrRec[0][i].y-nearestRectToCtr[i].y;
                ty.at<float>(nearestRectToCtr[i]) = v2;
            nbPt++;
        }
    }
}
for (int i = 0; i < newCtr[0].size(); i +=1)
{
    if (tx.at<float>(newCtr[0][i]) == 100)
        cout << "Pb at " << newCtr[0][i] << "\n";
}
for (int i = 0; i < newCtr[0].size(); i++)
{
    float mTx=0;
    float mTy=0;
    int l=4;
    for (int j = -l; j <= l; j++)
    {
        int p=i+j;
        if (p<0)
            p+=newCtr[0].size();
        p = p % newCtr[0].size();
        mTx += (l+1-abs(j))*tx.at<float>(newCtr[0][p]);
        mTy += (l+1-abs(j))*ty.at<float>(newCtr[0][p]);
    }
    tx.at<float>(newCtr[0][i]) = mTx / ((l+1)*(l+1));
    ty.at<float>(newCtr[0][i]) = mTy / ((l+1)*(l+1));
}
ofstream fs("txty.txt");
for (int i = 0; i < newCtr[0].size(); i +=1)
{
        fs<<newCtr[0][i].x<<"\t"<<newCtr[0][i].y<<"\t" << tx.at<float>(newCtr[0][i]) << "\t" << ty.at<float>(newCtr[0][i]) << "\n";
}
imshow("mask",zoomMask);
imwrite("mask.png",mask);
waitKey();
mask=Mat::zeros(rC.height+2, rC.width+2,CV_8UC1);
drawContours(mask,newCtr,0,Scalar(255),-1);
drawContours(mask,newCtr,0,Scalar(128),1);
imshow("mask",mask);

for (int i=0;i<mask.rows;i++)
    for (int j = 0; j < mask.cols; j++)
    {
        if (mask.at<uchar>(i,j) ==255)
        {
            tx.at<float>(i,j)=mx/nbPt;
            ty.at<float>(i,j)=my/nbPt;
        }
    }

Mat lx,ly;
double t=0,dx2=1;
double diffusivity=100000000,dt=dx2/(4*diffusivity);
int iAfter=0;
Mat xSol;
Mat txc,tyc,mt;
{
    cv::FileStorage fsx("txini.yml", cv::FileStorage::WRITE);
	fsx<<"Image"<<tx;
	cv::FileStorage fsy("tyini.yml", cv::FileStorage::WRITE);
	fsy<<"Image"<<ty;
}
m=imread("C:/Users/Laurent/Downloads/plaqueDeforme3.jpg",CV_LOAD_IMAGE_COLOR);
while (t<1000*dt)
{
    Laplacien(mask,tx,ty,lx,ly);
    tx =tx +(diffusivity*dt)*lx;
    ty =ty +(diffusivity*dt)*ly;
    t=t+dt;
    mt = Mat::zeros(m.size(),CV_8UC3);
    int nbPtErr=0;
    for (int i=0;i<mask.rows;i++)
        for (int j = 0; j < mask.cols; j++)
        {
            uchar v=mask.at<uchar>(i,j);
            if (mask.at<uchar>(i,j) >0)
            {
                float offsetx = tx.at<float>(i,j),offsety=ty.at<float>(i,j);
                int lig=i+offset.y+offsety*1,col=j+offset.x+offsetx*1;
                if (lig>=0 && lig<mt.rows && col>=0 && col<mt.cols)
                    mt.at<Vec3b>(lig,col) = m.at<Vec3b>(i+offset.y,j+offset.x);
                else 
                {
                    nbPtErr++;
                }
            }
        }
    imshow("RESULTAT FINAL",mt);
    cout<<nbPtErr<<endl;
    waitKey(10);
}
	cv::FileStorage fsx("tx.yml", cv::FileStorage::WRITE);
	fsx<<"Image"<<tx;
	cv::FileStorage fsy("ty.yml", cv::FileStorage::WRITE);
	fsy<<"Image"<<ty;
    imwrite("result.jpg",mt);
    waitKey();



return 0;
}
