# flower-bouquet-classification

## Project Explained

In this assignment, we have a dataset of images of 8 flower species from a flower shop and are given the two tasks of:

1. Classifying the images according to a flower type
2. Recommend 10 flower images in the dataset which is similar to the input flower image

The first task belongs to the categories of multiclass image classification problem and supervised learning. The second task is an unsupervised learning problem

In this project, I work mainly on the task 1 including data processing, modelling and fintuning the Image classifier and documenting. While my friend, `Hieu Le works mainly on task 2 including data cleaning, moddelling an Image Recommender as well as documenting`. 

## Exploratory Data Analysis

### Imbalanced classes

> When plotting out the data, a major observation can be seen in which there exists significant class imbalances with the data(see  Appendix A). Certain flower species, namely rosy and ping pong, contain much less data than the other species and possibly make the resulting model less reliable at classifying them. The inverse is also true with babi and lisianthus, and could cause majority class bias.

### Data irregularities
> 1. We also discovered numerous irregularities regarding the images within the datasets. These irregularities are:
Wrongly labelled data. Example being Pingpong_293, which is a bundle of rosy flowers wrongly labelled as pingpongs.

> 2. Ambiguous flower image. Example being pingpong_46, which is a bundle of both pingpongs and tana flowers, but they are labeled as pingpongs. The model could potentially misclassify it as tana.

> 3. Duplicated images. Certain images, with babi_623 and babi_624 being the exact same image. This could lead to overfitting in the model.

## Data Preparation

### Handling Duplication

We think that duplication handling was needed for our project because the presence of duplicate images could introduce bias towards the duplicated image into the resulting model, as Zhu et al.[1] found that duplicate data inside a dataset “will trap the training of the neural networks into the problem of over-fitting”.

For this purpose, we chose to use the difPy library, which implements a function that finds and removes duplicate images in a dataset. The methodology behind this function is that the tensor of the images in the folder are computed, and then the computed mean squared error(MSE) of the images are compared with each other. If the MSE is lower than a defined threshold, it is considered a duplicated image and the user can choose to remove the image. In our case, the MSE was set to 0 as we only want to look for exact duplicates. Another reason why difPy was chosen is because of its robust ability to detect rotated duplicate, in which an image is a duplicate from the original, but is instead rotated(see Appendix B) and thus won’t be detected using a normal hashing method. For an example portion of the discovered duplicates, refer to Appendix C.

### Manual cleansing

While manual cleaning of data is usually avoided in machine learning, with frameworks such as ImageDC[2] paving the way for automated data cleaning, we still felt that manual cleansing was needed for our given dataset. The reasons are that first is that the relatively small dataset makes it a possibility, and second is because we felt that training a neural network to account for all the data irregularities would be challenging and time consuming. During our manual cleansing, we iterate through all the images in each flower species and delete any images that are either wrongly labelled or are ambiguous.

## Data Preprocessing

### Data Augmentation

Due to insufficient amount of training data, we considered the option of using data augmentation. This is because a paper by Poojary et al.[3] studied the effect of data augmentation on CNN models and found that test accuracy gets better when data augmentation is applied, with an accuracy increase of 8% being observed in their study. Thus, we believe that our data situation would benefit from some data augmentation.

The data augmentations that we use are randomly flipping horizontally, and randomly rotating from 0 to 20 degrees max, and this is all done through the keras processing library.

### Class Re-Weighting

The differences in the amount of flowers in each flower type creates a big challenge for the neural network since it might construct a bias towards the flower folder that contains the most amount of flowers. To falsely predict a large number of predictions towards those folders having most flowers, even though the accuracy can be high but when seeing other metrics like precision, recall or f1 score, they will be very low. At first, our group applied this method to improve the overall performance of the neural network but we quickly found out that it doesn't improve much so we stopped using it. 

## Modelling & Fine Tuning

### Task 1: image classification

This dataset carries some of the most stressful challenges ranging from dirty data points to imbalanced dataset so that the use of a simple CNN will not be helpful. Therefore, my team decided to leverage the great architecture of the strong existing models like ResNet50, Xception and DenseNet121 to use them as the base models then fine tuning it to make it work well with the given dataset. 

Imbalanced dataset is the biggest problem of this data. And to deal with that problem, our team used the above models because on average, each of them has around 100 layers. Having a good enough base model will generate the best result on the data if we chose to use the weaker models like VGG16 or VGG19, the performance of the test set will be ahead that of the train set. That would cause underfitting. On the other hand, using too powerful models like ResNet101 or DenseNet169 will make the train set learn faster than the test set which will cause overfitting. The use of too powerful models will also be a waste of computing power. 

### Task 2: image recommendation

Even though the more popular approach for doing recommendation systems are content-based recommendation and collaborative filtering recommendation, the lack of the detailed information and description about each photo in the dataset. Therefore the approach that we chose to do is similarity-base recommendation. 

In this approach, firstly, we built a CNN model to make predictions later to recommend 10 similar images. After that, we transformed the image from resizing to perform dimensionality reduction to extract the image features. Then we applied a k-mean clustering method to cluster similar data points and created 5 clusters. After inputting an image to recommend, the CNN model will extract the feature from the input image and use cosine similarity to recommend 10 similar images for the users.


### Model Evaluation & Ultimate Judgement

Model                 | Test Accuracy
ResNet50 Based CNN    | 77.3%
DenseNet121 Based CNN | 70%
Xception Based CNN    | 72%
Traditional CNN       | 37%

We found that the ResNet50 base model works best for task 1 as it had the highest test accuracy out of all the models. And using a similarity based approach for task 2 is a good idea as the recommender doesn’t recommend merely on the flowers type but instead based on image similarity, which is what was required for task 2.

To compare our model from task 1 to those from existing literature, we picked a CNN transfer learning model from Narvekar and Rao[4] and a CNN model with stochastic pooling strategy from Prasad et al.[5]	. Both of these models perform image classification on flower species, similar to ours.

Model                     | Test Accuracy
Chosen ResNet50 Based CNN | 77%
Narvekar and Rao[4]       | 91%
Prasad et al.[5]          | 93.98%

As shown above, the models from existing literature vastly outperforms ours in testing accuracy.

### References

[1]	E. Zhu, Y. Ju, Z. Chen, F. Liu, and X. Fang, “DTOF-ANN: An Artificial Neural Network phishing detection model based on Decision Tree and Optimal Features,” Appl. Soft Comput., vol. 95, p. 106505, Oct. 2020, doi: 10.1016/j.asoc.2020.106505.
[2]	Y. Zhang, Z. Jin, F. Liu, W. Zhu, W. Mu, and W. Wang, “ImageDC: Image Data Cleaning Framework Based on Deep Learning,” in 2020 IEEE International Conference on Artificial Intelligence and Information Systems (ICAIIS), Mar. 2020, pp. 748–752. doi: 10.1109/ICAIIS49377.2020.9194803.
[3]	R. Poojary, R. Raina, and A. Kumar Mondal, “Effect of data-augmentation on fine-tuned CNN model performance,” IAES Int. J. Artif. Intell. IJ-AI, vol. 10, no. 1, p. 84, Mar. 2021, doi: 10.11591/ijai.v10.i1.pp84-92.
[4]	C. Narvekar and M. Rao, “Flower classification using CNN and transfer learning in CNN- Agriculture Perspective,” in 2020 3rd International Conference on Intelligent Sustainable Systems (ICISS), Dec. 2020, pp. 660–664. doi: 10.1109/ICISS49785.2020.9316030.
[5]	M. V.D. Prasad et al., “An efficient classification of flower images with convolutional neural networks,” Int. J. Eng. Technol., vol. 7, no. 1.1, p. 384, Dec. 2017, doi: 10.14419/ijet.v7i1.1.9857.
