# Deep_Learning_SKin

# Introduction

Skin cancer can be simply defined as the growth in the skin
which happens due to damage to the skin cells. This research is
based on the detection of various skin diseases both non
melanoma and melanoma, using deep learning techniques known
as Convolutional Neural Network (CNN) to classify the image
dataset. The image dataset comprises of nine classes of skin
disease which is the input to the algorithm. Many research have
been performed on skin cancer detection and they all focus on a
particular class which is melanoma, this is because it is a major
causes of skin cancer, but it is important to note that other nonmelanoma may result into challenging issues if not detected and
treated early. The aim of this research is to encourage the
accurate detection of different nine classes of skin cancer which
may be difficult during diagnosis. This is achieved using
Convolutional Neural Network models. The CNN used includes
custom CNN, VGG16 and ResNet18. Custom CNN has 5
Conv2d and MaxPool2d, 8 activation function (ReLu) and 4 fully
connected layers. The collected dataset has 2239 training
samples and 118 test samples. 15% of the training samples were
taken for validation. 

# Dataset
The dataset contains 2,239 samples while the
test data contains 118 samples, the validation dataset was created
by taking 15% of samples from each class in the training dataset.
The task has 9 classes of skin cancer which are actinic keratosis,
basal cell carcinoma, dermatofibroma, melanoma, nevus,
pigmented benign keratosis, squamous cell carcinoma, and
vascular lesion. Four classes have the highest number of
samples, while the other 5 are underrepresented. 
To prevent overfitting due to the limited number of data
available in the training data and also to improve the quality of
the training data, it is important that data augmentation is
performed (Shorten et al., 2019). The augmentation employed
includes random horizontal flip, random rotation to a degree of
10, color jitter - tuning the brightness, saturation and contrast.
The images of in the training data were reduced to a uniform size
of 224 x 224. 
![image](https://github.com/user-attachments/assets/a43c12ff-2392-4132-9d70-9142d9fd5a8e)
