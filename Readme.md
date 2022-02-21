# Wafer map pattern classification by Combining Convolutioal and Handcrafted Features 

Wafer map defect pattern classification with Multi-Input Neural Network of Convolutioal and Handcrafted Features 

Proposed by H.Kang and S.Kang

Hyungu Kang, Seokho Kang* (2021), "A stacking ensemble classifier with handcrafted and convolutional features for wafer map pattern classification", Computers in Industry 129: 103450 (https://www.sciencedirect.com/science/article/pii/S0166361521000579?via%3Dihub)

## Methodology

### Multi-input neural network (MultiNN)

![](https://github.com/DMkelllog/WMPC_MultiNN/blob/main/MultiNN%20flow?raw=true)

* Input:    wafer map
  * resized to 64x64
* Output: predicted score
* Model:  CNN (based on VGG16)
* handcrafted features (59-dim) are concatenated to CNN features (512-dim)


## Data

* WM811K
  * 811457 wafer maps collected from 46393 lots in real-world fabrication

  * 172950 wafers were labeled by domain experts.

  * 9 defect classes (Center, Donut, Edge-ring, Edge-local, Local, Random, Near-full, Scratch, None)

  * provided by MIR Lab (http://mirlab.org/dataset/public/)

  * .pkl file downloaded from Kaggle dataset (https://www.kaggle.com/qingyi/wm811k-wafer-map)

  * directory: /data/LSWMD.pkl

## Dependencies

* Python 3.8
* Pytorch 1.9.1
* Pandas 1.3.2
* Scikit-learn 1.0.2
* OpenCV-python 4.5.3

## References

* WM-811K(LSWMD). National Taiwan University Department of Computer Science Multimedia Information Retrieval LAB http://mirlab.org/dataSet/public/
* Nakazawa, T., & Kulkarni, D. V. (2018). Wafer map defect pattern classification and image retrieval using convolutional neural network. IEEE Transactions on Semiconductor Manufacturing, 31(2), 309-314.
* Shim, J., Kang, S., & Cho, S. (2020). Active learning of convolutional neural network for cost-effective wafer map pattern classification. IEEE Transactions on Semiconductor Manufacturing, 33(2), 258-266.
* Kang, S. (2020). Rotation-Invariant Wafer Map Pattern Classification With Convolutional Neural Networks. IEEE Access, 8, 170650-170658.
