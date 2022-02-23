import pandas as pd
import numpy as np
import pickle
import cv2

from skimage import measure
from skimage.transform import radon
from scipy import interpolate
from scipy import stats


def cal_den(x):
    return 100*(np.sum(x==2)/np.size(x))  

def find_regions(x):
    rows, cols= x.shape
    ind1=np.arange(0,rows,rows//5)
    ind2=np.arange(0,cols,cols//5)
    
    reg1=x[ind1[0]:ind1[1],:]
    reg3=x[ind1[4]:,:]
    reg4=x[:,ind2[0]:ind2[1]]
    reg2=x[:,ind2[4]:]

    reg5=x[ind1[1]:ind1[2],ind2[1]:ind2[2]]
    reg6=x[ind1[1]:ind1[2],ind2[2]:ind2[3]]
    reg7=x[ind1[1]:ind1[2],ind2[3]:ind2[4]]
    reg8=x[ind1[2]:ind1[3],ind2[1]:ind2[2]]
    reg9=x[ind1[2]:ind1[3],ind2[2]:ind2[3]]
    reg10=x[ind1[2]:ind1[3],ind2[3]:ind2[4]]
    reg11=x[ind1[3]:ind1[4],ind2[1]:ind2[2]]
    reg12=x[ind1[3]:ind1[4],ind2[2]:ind2[3]]
    reg13=x[ind1[3]:ind1[4],ind2[3]:ind2[4]]
 
    fea_reg_den = np.array([cal_den(reg1),cal_den(reg2),cal_den(reg3),cal_den(reg4),cal_den(reg5),cal_den(reg6),cal_den(reg7),cal_den(reg8),cal_den(reg9),cal_den(reg10),cal_den(reg11),cal_den(reg12),cal_den(reg13)])
    return fea_reg_den


def change_val(img):
    img[img==1] =0  
    return img


def cubic_inter_mean(img):
    theta = np.linspace(0., 180., max(img.shape), endpoint=False)
    sinogram = radon(img, theta=theta, preserve_range=True)
    xMean_Row = np.mean(sinogram, axis = 1)
    x = np.linspace(1, xMean_Row.size, xMean_Row.size)
    y = xMean_Row
    f = interpolate.interp1d(x, y, kind = 'cubic')
    xnew = np.linspace(1, xMean_Row.size, 20)
    ynew = f(xnew)/100   # use interpolation function returned by `interp1d`
    return ynew

def cubic_inter_std(img):
    theta = np.linspace(0., 180., max(img.shape), endpoint=False)
    sinogram = radon(img, theta=theta, preserve_range=True)
    xStd_Row = np.std(sinogram, axis=1)
    x = np.linspace(1, xStd_Row.size, xStd_Row.size)
    y = xStd_Row
    f = interpolate.interp1d(x, y, kind = 'cubic')
    xnew = np.linspace(1, xStd_Row.size, 20)
    ynew = f(xnew)/100   # use interpolation function returned by `interp1d`
    return ynew  


def cal_dist(img,x,y):
    dim0=np.size(img,axis=0)    
    dim1=np.size(img,axis=1)
    dist = np.sqrt((x-dim0/2)**2+(y-dim1/2)**2)
    return dist  


def fea_geom(img):
    norm_area=img.shape[0]*img.shape[1]
    norm_perimeter=np.sqrt((img.shape[0])**2+(img.shape[1])**2)
    
    img_labels = measure.label(img, connectivity=1, background=0)

    if img_labels.max()==0:
        img_labels[img_labels==0]=1
        no_region = 0
    else:
        info_region = stats.mode(img_labels[img_labels>0], axis = None)
        no_region = info_region[0][0]-1       
    
    prop = measure.regionprops(img_labels)
    prop_area = prop[no_region].area/norm_area
    prop_perimeter = prop[no_region].perimeter/norm_perimeter 
    
    prop_cent = prop[no_region].local_centroid 
    prop_cent = cal_dist(img,prop_cent[0],prop_cent[1])
    
    prop_majaxis = prop[no_region].major_axis_length/norm_perimeter 
    prop_minaxis = prop[no_region].minor_axis_length/norm_perimeter  
    prop_ecc = prop[no_region].eccentricity  
    prop_solidity = prop[no_region].solidity  
    
    return np.array([prop_area,prop_perimeter,prop_majaxis,prop_minaxis,prop_ecc,prop_solidity])

    
def manual_feature_extraction(x):
    den = find_regions(x)
    radon_mean = cubic_inter_mean(change_val(x))
    radon_std = cubic_inter_std(change_val(x))
    geom = fea_geom(change_val(x))
    
    return np.concatenate((den,radon_mean,radon_std,geom))


if __name__ == '__main__':
    mapping_type={'Center':0,'Donut':1,'Edge-Loc':2,'Edge-Ring':3,'Loc':4,'Random':5,'Scratch':6,'Near-full':7,'none':8}

    df = pd.read_pickle('../data/LSWMD.pkl')
    df=df.replace({'failureType':mapping_type})
    df.drop(['lotName', 'waferIndex', 'trianTestLabel'],axis=1,inplace=True)

    df_withlabel = df[(df['failureType']>=0) & (df['failureType']<=8)]
    df_withlabel = df_withlabel[df_withlabel['dieSize'] > 100]

    X = df_withlabel['waferMap'].values
    y = df_withlabel['failureType'].values.astype(np.int64)
    
    manual_features = np.array([manual_feature_extraction(x).astype(np.float32) for x in X])
    pickle.dump(y, open('../data/y.pickle', 'wb'))
    pickle.dump(manual_features, open(f'../data/X_MFE.pickle', 'wb'))

    X_binary = np.array([np.where(x==2, 1, 0).astype('uint8') for x in X], dtype=object)
    X_resized = np.array([cv2.resize(x*255, (64, 64), interpolation=2) for x in X_binary])

    pickle.dump(y, open('../data/y.pickle', 'wb'))
    pickle.dump(X_resized, open(f'../data/X_CNN.pickle', 'wb'))

    