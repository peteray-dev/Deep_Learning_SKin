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

Due to the limited available images, the model was discovered to be biased in its classification, so to tackle class imbalance which is the main reason for this bias, data balancing approach was employed, augmenting the training data images in each class to 500, then using a 15% splitting approach to set out the images required for validation (Figure 3).  
 
![image](https://github.com/user-attachments/assets/3ddd7b26-6f09-42b7-a088-d4d0850fedf7)

# METHODOLOGY
For a robust classification of skin cancer based on images, three models were employed, the custom CNN, ResNet18 and VGG19.
CNN is a special type of multilayer neural network or deep learning architecture that is inspired by  the visual system of living beings (Ghosh et al., 2020). CNN model learn different features in the image in three dimensions (RGB) which are the input data, the input data passes through the convolution layer which contains kernel that convolve the images to generate an output feature. The kernel is also known as filter having discrete values which are wight, this kernel learns and extract the important features in the image as an output matrix.
Input matrix = 𝑛ℎ × 𝑛w  
Kernel = 𝑘h × 𝑘w
output matrix = (𝑛ℎ − 𝑘h + 1) × (𝑛𝑤 − 𝑘w + 1)

When the convolution operation is then applied with stride which is the pace at which the kernel moves across the input matrix and padding which give border size to the input matrix inorder to keep the feature on the border, the output feature is then calculated as: 
Output shape=[(𝑛ℎ−𝑘ℎ+2×𝑝ℎ)/𝑠ℎ+1) ×(𝑛w−𝑘w+2×𝑝w)/𝑠w+1) ] 
In order to reduce the size and also preserve the important feature, max pooling is employed, the larger image size is shrink to a lower size and at the same time retaining the dominant features in each steps. When this layer is used the output is calculated as:
Output shape=[(𝑛ℎ−𝑘ℎ+2×𝑝ℎ)/𝑠ℎ+1) ×(𝑛w−𝑘w+2×𝑝w)/𝑠w+1) ]
In each layer of the CNN architedture, the activation function is utilized to ensure that there is no linearity so that the model can learn complexly, the activation fnnction mostly used is Relu, this is because of its ability to covert all the input values to positive
f(x)ReLU = max(0, x)
After the convolutional layer, the fully connected layer is created which follows the principle of Multilayer Perceptron (MLP) architecture, this layer ensures that each neuron is connected to another neuron. The input which is the output from the convolutional layer is flattened to form a vector that enters into the fully connected layer

![image](https://github.com/user-attachments/assets/b1b59910-c48f-4b38-a8b3-2450a4fd4d06)

At the output, the loss function is calculated which is the difference between the actual and the predicted output, at this stage the loss function is to be kept to its minimal level inorder to have a higher accuracy. In the case of higher error, the process is optimized using an optimizer, this helps to balance the weight in a process called back propagation. The optimizer utilizes learning rate which ensure the reduction of the error at a lower rate, a higher learning rate will cause the reduction not to come to it’s optimal state, the widely used learning rate is Adaptive Momentum Estimation (Adam), which puts into consideration the momentum, learning rate and RMSprop, others include Stochastic Gradient Descent, Batch Gradient Descent, Root Mean Square Propagation (RMSProp) etc. The loss function used is Cross-Entropy because it uses the softmax activation at the output layer so as to generate a probability distribution of the output.
In this project, to ensure the selection of a perfect learning rate, a learning rate finder was used, also a strategy of early stopping was employed that’s helps to stop the training process when the performance is getting worse and the difference between the training and validation loss is high 
# A.	Custom CNN:
The developed architecture for this research includes 4 convolutional layer and 3 fully connected layers and the output (figure)
![image](https://github.com/user-attachments/assets/a4d60ae4-e929-4b36-8fc1-180ffeb5533a)
Figure: the custom CNN architecture
# B.	ResNet18
This residual network has 18 layers with a 7x7 kernel at the first layer, it has 4 convolutional layers, the uniqueness in this model is skip connection to the output with a ReLu (Sai and Allena, 2022). This is aim to train deeper networks and then compared to the custom architecture to improve classification accuracy
![image](https://github.com/user-attachments/assets/0d46bb56-4038-4ded-968c-bd79c840b670)
Figure: ResNet18 architecture

# C.	VGG 16
This classification algorithm has 13 convolutional layers, 5 max pooling layers and 3 fully connected layers. This algorithm is known for its simplicity and image classification, and it is used to learn intricate features to further analyze its characteristics among others.

#EXPERIMENTAL RESULTS AND DISCUSSION
In the dataset, some image classes were underrepresented which resulted into overfitting and refusal to predict the classes with less images, to overcome this data augmentation was employed to generate more images and ensure that all classes have almost the same image present which led to the increment of the training samples from 2,239 to 4,500. The test samples were left unmodified. Since deep learning techniques required a large memory, batch size of 32 was used which processed only a small subset of the entire dataset for training at a time for parallel computation.
To choose an optimal learning rate, a learning rate finder package was used which suggested an optimal learning rate, this help to lessen the amount of guess work that would be employed to choose a learning rate. Adam optimizer was used because it is robust compare with others, it shows a marginal improvement over SDG with momentum (Diederik and Jimmy, 2014)
A.	Evaluation Metrics
For this classification problem, confusion matrix is utilized because this provides a prediction summary in matrix form thereby offering a depiction of the model performance. This helps to measure recall, precision, specificity, and accuracy.
		Actual Values
Predicted Values		Positive	Negative
	Positive	TP	FP
	Negative	FN	TN

Figure: Confusion Matrix
Recall	 = 	    TP
    TP + FN
precision = 	    TP
    TP + FP
Accuracy = 	TP + TN
TP + TN +FP+ FN
F1 Score =   2 * 	Precision x Recall
Precision + Recall

Result and Discussions
The performance of the custom CNN was evaluated by training it
over 100 epochs, but stopped training at 79 epoch through the
usage of early stopping. As expected, training and validation loss
decreased while the accuracy increased. However, the test
accuracy of 41% which is significantly lower when compared
with training and validation accuracies, which were both above
70%. This discrepancy is as a result of smaller images in the test
dataset, to address this, more test image data and sufficient
computational resources is needed. 
![image](https://github.com/user-attachments/assets/d04ab42d-6a18-41c7-96ba-11531b9a1669)

![image](https://github.com/user-attachments/assets/011fb680-fdf4-4867-8304-0d2141eb9985)

Figure 8: Performance of custom CNN (Loss and Accuracy)
Resnet18 which had a deeper architecture had its training and
validation loss lower, while the accuracy score is between 80%
and 97% for validation and training when ran on 100 epochs, the
test accuracy was 53% (Figure 9).

![image](https://github.com/user-attachments/assets/51ca29c2-b07b-4324-9a74-86bf04fa3ad4)

![image](https://github.com/user-attachments/assets/f8813813-1460-49ea-b00f-e436a5ca5030)

Figure 9: Performance of ResNet18 (Loss and Accuracy)
VGG16 stopped at epoch 87 using early stopping when the
difference between the training and validation loss increases, this
helps to avoid overfitting. The test accuracy score is 50%
![image](https://github.com/user-attachments/assets/7ad52241-13c3-41bb-afdc-ad1d97c470b2)
![image](https://github.com/user-attachments/assets/891794ce-2064-4d22-b71a-7d085a512d7a)

Figure 10: Performance of VGG16 (Loss and Accuracy)
There is generally no concern about overfitting between the
training and validation sets because they have a sufficient amount
of image data. However, the small size of the test set led to
underfitting on the unseen data. While early stopping helps to
prevent overfitting on the training data, it doesn’t guarantee good
performance on the limited test data.
Model Accuracy Precision Recall F1-score
Custom 41% 38% 42% 37%
ResNet18 53% 57% 55% 52%
VGG16 50% 54% 50% 50%
Table 2 : Result Summary
The table above explains the performance metrics for the three
models. ResNet18 has a higher accuracy and made more true
positive prediction than any of the other models.
From the confusion matrix analysis, it is seen that ResNet18
predicted a higher number of skin cancer accurately (True
Positive), ResNet18 has fewer misclassification as compared with
other CNN models.
VI. CONCLUSION
The study investigates skin cancer prediction using a custom
CNN and two transfer learning models (ResNet18 and VGG16).
ResNet18 achieved superior performance, making it the preferred
choice for this classification task. This is likely due to the deeper
architecture of transfer learning models compared to the custom
CNN. However, the research has limitations. Firstly, the lack of a
GPU hinders training on a larger number of epochs (iterations).
Secondly, the test dataset size was limited, which may have
impacted the model's ability to generalize well or make a better
test predictions. If this limitations are addressed, the model’s
performance will be improved. 
REFERENCES
N. Rezaoana, M. S. Hossain and K. Andersson, "Detection and Classification of Skin Cancer by Using a Parallel CNN Model," 2020 IEEE International Women in Engineering (WIE) Conference on Electrical and Computer Engineering (WIECON-ECE), Bhubaneswar, India, 2020, pp. 380-386, doi: 10.1109/WIECON-ECE52138.2020.9397987. keywords: {Transfer learning;Neural networks;Skin;Lesions;Task analysis;Sun;Skin cancer;Skin cancer;CNN;Data Augmentation;Deep learning;Transfer learning},
Naqvi, M.; Gilani, S.Q.; Syed, T.; Marques, O.; Kim, H.-C. Skin Cancer Detection Using Deep Learning—A Review. Diagnostics 2023, 13, 1911. https://doi.org/10.3390/diagnostics13111911
Kalouche, Simon. “Vision-Based Classification of Skin Cancer using Deep Learning.” (2016).
Hassan, Hafiz & Jan, Bismillah & Ahmad, Zara & Tahira, Fatima & Khan, Muhammad. (2019). Skin Lesion Classification Using Deep Learning Techniques.
A, Praveen & V, Kanishk & K, Vineesh & Ayothi, Senthilselvi. (2023). Skin Cancer Classification using Multiple Convolutional Neural Networks. Journal of Soft Computing Paradigm. 5. 327-346. 10.36548/jscp.2023.4.001.
Shorten, Connor & Khoshgoftaar, Taghi. (2019). A survey on Image Data Augmentation for Deep Learning. Journal of Big Data. 6. 10.1186/s40537-019-0197-0.
Ghosh, Anirudha & Sufian, A. & Sultana, Farhana & Chakrabarti, Amlan & De, Debashis. (2020). Fundamental Concepts of Convolutional Neural Network. 10.1007/978-3-030-32644-9_36.
Sai Abhishek, Allena Venkata. (2022). Resnet18 Model With Sequential Layer For Computing Accuracy On Image Classification Dataset. 10. 2320-2882.
Kingma, Diederik & Ba, Jimmy. (2014). Adam: A Method for Stochastic Optimization. International Conference on Learning Representations.


