# Age Detection Classifier

<div style="text-align: center;">
<li> Shoaib Ahmed Khan </li>
<li> Iretioluwa Olawuyi </li>
<li> Esther Yu </li>
<li> Pratheep </li>
</div>

## Introduction
In the era of rapidly advancing technology, deep learning and artificial intelligence have revolutionized various industries, and computer vision is no exception. Age detection, a fundamental task in computer vision, plays a crucial role in various domains, including surveillance, targeted advertising, and personalized user experiences. The ability to accurately estimate a person's age from images has far-reaching implications, making it an essential problem to solve.

This report presents the development and evaluation of an age detection classification
model utilizing Convolutional Neural Networks (CNNs). CNNs have proven to be highly
effective in image-related tasks due to their ability to learn hierarchical features directly from the raw pixel data. The proposed model leverages the power of CNNs, along with MaxPooling and Dense layers, to automatically extract relevant features and predict the age of subjects depicted in images.

The primary objective of this project is to design a robust and accurate age detection
system that can be deployed in real-world scenarios. Throughout the report, we will
delve into the details of the dataset used for training and testing, the architecture of the CNN model, the preprocessing steps undertaken to enhance model performance, and
the evaluation metrics employed to assess the model's accuracy.

## Section 1: Related Work
The field of age estimation and classification has garnered significant attention from
researchers and practitioners over the years. A myriad of approaches has been
proposed, ranging from traditional statistical methods to more recent deep
learning-based techniques. In this section, we present an overview of some of the
seminal works and recent advancements in the domain of age detection.

## 1.1 Traditional Approaches:
Early age detection methods predominantly relied on handcrafted features and
traditional machine learning algorithms. These methods often involved extracting facial
landmarks, texture patterns, and statistical features, followed by employing techniques
such as Support Vector Machines (SVMs) or Random Forests for classification. While
these methods served as valuable foundations for age estimation, they were limited by the need for expert domain knowledge to handcraft features and struggled to handle
complex variations in facial appearance.

## 1.2 Deep Learning-based Approaches:
The advent of deep learning has dramatically transformed the landscape of age
detection. Convolutional Neural Networks (CNNs), in particular, have emerged as a
dominant force in image-related tasks. CNNs can automatically learn hierarchical
features from raw pixel data, thus alleviating the need for manual feature engineering.
The pioneering work of Krizhevsky et al. with the AlexNet architecture in ImageNet
challenge demonstrated the potential of CNNs in large-scale image classification tasks,
sparking a new era of research in computer vision.

##  1.3 Recent Advancements:
Recent research in age detection has explored novel techniques like attention
mechanisms, multi-task learning, and generative adversarial networks (GANs). Attention
mechanisms help focus on crucial facial regions, while multi-task learning
simultaneously predicts age and other facial attributes, leading to better generalization.
GANs have been utilized for generating synthetic age-progressed images to augment
datasets and improve model robustness.

## 1.4 Objective of This Study:
This study aims to contribute to the ongoing progress in age detection by proposing a
deep learning-based age classification model. The proposed model leverages the power
of CNNs to automatically learn relevant facial features and accurately predict age.
Through extensive experimentation and evaluation, we endeavor to demonstrate the
effectiveness and potential applications of our age detection system.

# Section 2: Dataset
The foundation of any machine learning model lies in the quality and diversity of the
dataset used for training and evaluation. In this section, we provide a comprehensive
overview of the dataset employed in this study, including its collection, annotation, and preprocessing steps.
## 2.1 Data Collection and Sources:
The dataset used in this research effort is a custom-created collection of facial images, curated specifically for age detection purposes. The images were obtained from various sources, including publicly available datasets, social media platforms, and image repositories. To ensure a diverse representation of age groups and facial variations, images were collected from different demographics and geographical regions.
## 2.2 Data Annotation and ID Assignment:
To facilitate supervised learning, each image in the dataset was manually annotated
with the corresponding age label. The age labels were assigned based on the subjects'
actual ages at the time the images were captured. To ensure privacy and ethical
considerations, all identifying information was anonymized, and subjects were assigned
unique identifiers (IDs) instead of using their real names or personal information.
## 2.3 Data Preprocessing:
Data preprocessing is a crucial step in preparing the dataset for training a deep learning
model. Several preprocessing techniques were applied to enhance the model's
performance and robustness:
### 2.3.1 Data Augmentation: To mitigate overfitting and expand the dataset's diversity, data
augmentation techniques were employed. Random rotations, horizontal flips, and
brightness adjustments were applied to generate variations of the original images.
### 2.3.2 Normalization: Pixel values of the images were normalized to a range suitable for
the chosen activation function. This step ensures that the model converges faster
during training.
## 2.4 Train-Test Split:
The dataset consisting of 19,906 images for training and 6,636 images for testing was
split in a stratified manner to ensure an even distribution of age groups in both sets. The
training set was used to train the model, and the testing set was reserved to evaluate
the model's generalization and performance on unseen data.
## 2.5 Data Imbalance:
One common challenge in age detection datasets is class imbalance, where certain age
groups may be underrepresented. To mitigate this issue, the dataset was carefully
examined for imbalances, and appropriate techniques, such as class weighting or
oversampling, were employed during training to prevent bias towards dominant age
groups.

In conclusion, this section has provided a detailed account of the dataset used for
training and evaluating the age detection classification model. The combination of
diverse facial images, appropriate annotation, and careful preprocessing lays the
groundwork for the subsequent sections, where we elaborate on the model architecture,
training process, and evaluation results. By leveraging this well-curated dataset, our aim
is to build a robust and accurate age detection system capable of making meaningful
predictions on unseen facial images.

# Section 3: Model Architecture and Training (Experiment 1)
In this section, we will delve into the architecture of the age detection classification
model used in Experiment 1.
## 3.1 Model Architecture:
The age detection model is implemented using the Sequential API from Keras, a
high-level neural networks API running on top of TensorFlow. The model consists of
three layers:
### 3.1.1 Input Layer: 
The input layer defines the shape of the input data. In this case, the
input_shape parameter is set to input_num_units, which represents the dimensions of
the input image. The model expects a flattened input, as indicated by the subsequent
Flatten layer.
### 3.1.2 Flatten Layer: 
The Flatten layer is used to convert the two-dimensional image data
into a one-dimensional array. It reshapes the input data into a vector, preparing it for the
subsequent fully connected layers.
### 3.1.3 Dense Layers: 
Dense layers, also known as fully connected layers, play a critical
role in learning complex patterns from the flattened input. In Experiment 1, the model
has two Dense layers:
- First Dense Layer: The first Dense layer has hidden_num_units units/neurons
and uses the ReLU activation function. ReLU (Rectified Linear Unit) introduces
non-linearity into the model and helps the network learn more complex
relationships within the data.
- Second Dense Layer: The second Dense layer has output_num_units
units/neurons, which corresponds to the number of age categories in the
classification task. It utilizes the softmax activation function, which transforms
the output scores into a probability distribution, allowing the model to make age
predictions for each image.

## 3.2 Model Compilation:
After defining the architecture, the model is compiled with the following parameters:
Optimizer: The model uses Stochastic Gradient Descent (SGD) as the optimizer. SGD is
a widely used optimization algorithm that updates the model's parameters in the
direction that reduces the loss function during training.
Loss Function: The categorical cross-entropy loss function
('categorical_crossentropy') is used for multi-class classification problems like age
detection. It measures the dissimilarity between the predicted probability distribution
and the true label distribution.
Metrics: During training, the model's performance is evaluated using the accuracy
metric. The accuracy metric measures the percentage of correctly classified samples
out of the total samples.
## 3.3 Model Training:
The model.fit() function is used to train the age detection model with the following
parameters:
Training Data: The training data (train_x) contains the flattened images, and the
corresponding age labels are one-hot encoded (train_y).
Batch Size: The batch size determines the number of samples processed before the
model's parameters are updated. It controls the trade-off between computation
efficiency and memory usage during training.
Epochs: The number of epochs represents the number of times the entire training
dataset is passed through the model during training. Each epoch helps the model refine
its parameters and improve its accuracy.
Verbose: The verbose parameter is set to 1, which means that progress updates will be
displayed during training.
The model will undergo multiple epochs of training, gradually improving its ability to
predict the correct age categories for the input images. After training, the model's
performance will be evaluated on the test dataset to assess its generalization
capabilities and accuracy on unseen data.
Section 3: Model Architecture and Training
(Experiment 2)
In Experiment 2, we explore an alternative age detection classification model with a
Convolutional Neural Network (CNN) architecture.
## 3.4 Model Architecture:
The age detection model for Experiment 2 is implemented using the Sequential API
from Keras, with a CNN architecture. The model consists of multiple layers:
### 3.4.1 Convolutional Layers: 
Convolutional layers are designed to automatically learn
relevant features from the input images. In this experiment, the model comprises two
Conv2D layers:
- First Conv2D Layer: This layer has 32 filters, each with a 3x3 kernel size. The
activation function used is ReLU, which introduces non-linearity and helps the
model capture complex patterns.
- MaxPooling2D Layer: After the first Conv2D layer, a MaxPooling2D layer with a
2x2 pool size is applied. MaxPooling reduces the spatial dimensions of the
feature maps, leading to computational efficiency and some translation
invariance.
- Second Conv2D Layer: This layer has 64 filters, again with a 3x3 kernel size and
ReLU activation. It further extracts higher-level features from the already
processed feature maps.
- MaxPooling2D Layer: Following the second Conv2D layer, another MaxPooling2D
layer with a 2x2 pool size is applied, continuing the process of reducing spatial
dimensions.
### 3.4.2 Flatten Layer: The Flatten layer is used to convert the two-dimensional feature
maps into a one-dimensional vector. It prepares the data for the subsequent fully
connected layers.
### 3.4.3 Dense Layers: Similar to Experiment 1, the model includes two Dense layers:
- First Dense Layer: This Dense layer has 128 units/neurons and uses the ReLU
activation function. It introduces non-linearity and helps the network learn
complex relationships in the extracted features.
- Second Dense Layer: The output Dense layer remains unchanged from
Experiment 1, consisting of output_num_units units with softmax activation for
age category predictions.
## 3.5 Model Compilation:
The model is compiled with the following parameters:
- Optimizer: The Adam optimizer is used for this experiment. Adam is an adaptive
learning rate optimization algorithm that combines the benefits of both AdaGrad and
RMSProp, providing fast convergence and improved performance.
- Loss Function: The categorical cross-entropy loss function
('categorical_crossentropy') is again used for multi-class classification, measuring
the dissimilarity between the predicted probability distribution and the true label
distribution.
- Metrics: The model's performance will be evaluated using the accuracy metric,
measuring the percentage of correctly classified samples out of the total samples
during training and evaluation.
## 3.6 Model Training:
The model is trained using the model.fit() function, with the same parameters as
### Experiment 1:
Training Data: The training data (train_x) contains the images, and the corresponding
age labels are one-hot encoded (train_y).
- Batch Size: The batch size determines the number of samples processed before the
model's parameters are updated, impacting computation efficiency and memory usage.
- Epochs: The number of epochs represents the number of times the entire training
dataset is passed through the model during training.
- Verbose: The verbose parameter is set to 1, providing progress updates during training.

After training, the model's performance will be assessed on the test dataset, allowing us
to analyze the model's accuracy and generalization capabilities on unseen data.

# Section 4: Implementation and Training Analysis
In this section, we present an in-depth analysis of the implementation and training
process of Experiment 1 for the age detection classification model. We will explore the
hyperparameters used, monitor the training progress, and assess the model's
performance on both the training and validation datasets.
## 4.1 Hyperparameters:
The success of a deep learning model heavily depends on selecting appropriate
hyperparameters. In Experiment 1, the following hyperparameters were chosen:
- batch_size: The batch size determines how many samples are processed in
each training iteration before updating the model's parameters. A suitable batch
size balances the trade-off between computational efficiency and memory
utilization. The specific value of batch_size used in this experiment will be noted
here.
- epochs: The number of epochs defines the number of times the entire training
dataset is passed through the model during training. A sufficient number of
epochs are required to allow the model to converge and learn meaningful
features from the data. The specific number of epochs used in this experiment
will be mentioned.
- validation_split: The validation split parameter indicates the proportion of the
training data used for validation during training. This parameter helps us monitor
the model's generalization and detect potential overfitting.
## 4.2 Training Progress and Performance Evaluation:
During the training process, we will monitor the following key metrics:
- Training Loss and Accuracy: The training loss and accuracy are computed during
each epoch and indicate how well the model is learning from the training data.
- Validation Loss and Accuracy: The validation loss and accuracy are calculated
using the validation data during each epoch. They provide insights into the
model's generalization performance on unseen data.
## 4.3 Early Stopping:
To prevent overfitting, early stopping may be implemented. Early stopping monitors the
validation metrics and stops training if there is no significant improvement or if the
performance starts to deteriorate. This ensures that the model does not continue to
memorize the training data and, instead, learns generalizable patterns.
## 4.4 Training Analysis:
The training process will be analyzed to determine the following:
- Convergence: We will observe whether the training loss and accuracy reach a
stable state, indicating that the model has converged and learned meaningful
representations from the data.
- Overfitting: We will examine the training and validation metrics to check for signs
of overfitting. Overfitting occurs when the model performs well on the training
data but poorly on unseen data.
## 4.5 Training Evaluation:
Upon completion of training, we will evaluate the final performance of the model on the
testing dataset (6636 images) to assess its ability to generalize to completely new data.
We will compute the accuracy on the test set to measure the model's performance in
correctly predicting the age categories of unseen images.
# Section 5a: Results and Discussion
In this section, we present the results obtained from Experiment 1, including the training
progress, performance metrics, and analysis of the age detection classification model.
Let's examine the output provided for each epoch during training:
## 5.1 Training Progress:
From the training progress, we observe that the model was trained for 5 epochs
(epochs=5). Each epoch represents one pass through the entire training dataset. During
training, we can observe the following trends:
- The training accuracy steadily increases from approximately 63.38% (Epoch 1) to
64.59% (Epoch 5).
- The validation accuracy fluctuates between approximately 64.19% (Epoch 1) and
64.94% (Epoch 3) before dropping to 62.58% (Epoch 5).
5.2 Model Performance:
The model's accuracy on the training data increases with each epoch, indicating that the
model is learning from the training dataset. However, the validation accuracy
experiences slight fluctuations and then drops after Epoch 3. This discrepancy between
training and validation accuracy suggests a potential issue of overfitting.
## 5.3 Overfitting Analysis:
Overfitting occurs when the model performs well on the training data but fails to
generalize to unseen data. In Experiment 1, we observe a slight decline in validation
accuracy after Epoch 3, which may indicate the onset of overfitting. To address
overfitting, techniques like regularization, dropout layers, or early stopping can be
employed in future iterations of the model.
## 5.4 Model Evaluation on Test Data:
To gain a comprehensive understanding of the model's performance, it is essential to
evaluate it on unseen data. The model should be tested on the test dataset (6636
images), and the accuracy on this dataset will provide an estimate of the model's
generalization capabilities.
## 5.5 Future Improvements:
Based on the analysis, we can identify potential areas for improvement in the model and
training process. Experiment 1 can serve as a starting point to iteratively refine the
model architecture, hyperparameters, and regularization techniques to achieve better
performance and generalization.
# Section 5b: Results and Discussion (Experiment 2)
In this section, we present the results obtained from Experiment 2, which utilizes a
Convolutional Neural Network (CNN) architecture for the age detection classification
model. Let's examine the output provided for each epoch during training:
## 5.6 Training Progress:
From the training progress, we observe that the model was trained for 5 epochs
(epochs=5) with a CNN architecture. Each epoch represents one pass through the entire
training dataset. During training, we observe the following trends:
- The training accuracy increases from approximately 60.84% (Epoch 1) to 73.64%
(Epoch 5).
- The validation accuracy improves significantly from approximately 67.03%
(Epoch 1) to 72.05% (Epoch 3) before stabilizing at 71.77% (Epoch 4 and Epoch
5).
# 5.7 Model Performance:
The model exhibits notable improvements in both training and validation accuracy over
the course of 5 epochs. The training accuracy shows consistent growth, indicating that
the model is effectively learning from the training data. Moreover, the validation
accuracy shows significant progress up to Epoch 3 before stabilizing, suggesting that
the model is generalizing well to unseen data.
## 5.8 Overfitting Analysis:
From the training and validation accuracy trends, we observe that the model's accuracy
on the validation data is consistently close to the training accuracy, indicating a
reasonable generalization capability. The absence of significant divergence between the
two metrics suggests that the model is not overfitting.
## 5.9 Model Evaluation on Test Data:
To obtain a comprehensive evaluation, it is essential to assess the model on unseen
data, namely the test dataset (6636 images). Evaluating the model on the test data will
provide a reliable estimate of its real-world performance.
## 5.10 Future Improvements:
Considering the promising performance of Experiment 2, further exploration can be
conducted to fine-tune the CNN architecture, optimize hyperparameters, and employ
additional regularization techniques. By doing so, the model's accuracy and robustness
can potentially be further enhanced.

# Conclusion
In this study, we conducted two experiments to build an age detection classification
model using deep learning techniques. Experiment 1 utilized a simple architecture with
Dense layers, while Experiment 2 employed a more sophisticated Convolutional Neural
Network (CNN) architecture. Both experiments aimed to predict age categories from
facial images in a custom dataset comprising 19,906 training images and 6,636 testing
images, all anonymized with unique IDs.

Experiment 1 demonstrated reasonable performance, achieving a training accuracy of
approximately 64.59% and a validation accuracy of around 62.58%. However, the
validation accuracy exhibited fluctuations and a potential sign of overfitting, which calls
for further optimization and regularization to enhance generalization.

In contrast, Experiment 2, utilizing a CNN-based architecture, delivered more promising
results. The model's training accuracy rose from 60.84% to 73.64% over five epochs,
while the validation accuracy improved from 67.03% to a stable 71.77%. The CNN model
showed superior performance compared to the simpler architecture used in Experiment
1, successfully capturing intricate facial patterns and demonstrating better
generalization capabilities.
While both experiments provided valuable insights, Experiment 2 emerged as the
preferred choice for age detection tasks due to its higher accuracy and potential for
real-world applications. Further optimizations and fine-tuning can be explored in future
iterations to achieve even better results