# Speech-Emotion-Recognition-using-GMM-and-KNN
Speech Emotion Recognition (SER) refers to the automatic recognition of emotions using audios that are imported as dataset. Speech Emotion Recognition have received a lot attention in past history due to its wide range of applications in human computer interactions that affects computing and healthcare sectors. It develops intuitive and empathetic human machine interfaces; it is imperative and comprehend and interpret
speech emotions more accurately. Notwithstanding SER’s technology advancements, there are several challenges in language sectors using multilingual situations where existing systems find it difficult to achieve high levels of accuracy. 
## Speech Emotion Recognition
Speech Emotion Recognition (SER) is an important tool for enhancing humancomputer interaction and to revolutionize various fields such as healthcare, education, and customer service. Speech emotion recognition recognizes human emotions using Gaussian Mixture Model (GMM) and K-Nearest Neighbours (KNN) algorithms to understand emotion variations that are imported in dataset as audio files. Through the thorough analysis and practical verification, this project provides an important basis for further research work in this field. It collectively contributes to the understanding of spectral features and machine learning techniques in speech emotion recognition system.
### Problem Definition
The problem addresses in this project is the low accuracy using SVM classifier that is 72% in monolingual environment. The existing system does not use diverse multi language dataset for training the model, it simply uses English language for training the model which does help model to recognize emotions in other languages. SVM Model does not generalize across diverse linguistic and cultural contexts. The model
should be robust and ensure fairness and reduce discrimination in SER system. 
### Objectives
The objectives of this project are to enhance the accuracy of speech emotion recognition especially in multilingual environment. To achieve satisfactory accuracy levels that can be used in various fields such as human-computer interaction and affective computing. The project focuses on integrating Mel-Frequency Cepstral Coefficient (MFCC), Zero-Crossing Rate (ZCR), Teager Energy Operator (TEO) and Harmonic to Noise Ratio (HNR) with Gaussian Mixture Model (GMM) and KNearest Neighbour (KNN) as Machine Learning model. The project aims to develop a methodology that significantly enhances accuracy of speech emotion recognition
systems particularly in multilingual environment. The project develops techniques to reduce biases in system and ensure fairness in emotion detection across different demographic groups that minimizes risk of discrimination. It also develops strategies to improve generalization of speech emotion recognition models across diverse linguistic, cultural, and situational contexts, enabling reliable emotion recognition. Also, to investigate various methods to recognize the complex emotional expressions. 
## Algorithm
### Step 1: Data Preprocessing and Feature Extraction
1.1 Convert video samples from .avi format to .wav format for audio-based analysis.
1.2 Segregate the dataset into different emotion labels: happiness (HA), sadness (SA), anger (AN), fear (FE), surprise (SU), and disgust (DI).
1.3 Perform feature extraction on audio samples:
    Compute 39 Mel-frequency cepstral coefficients (MFCC).
    Calculate harmonic-to-noise ratio (HNR).
    Determine zero-crossing rate (ZCR).
    Compute Teager energy operator (TEO).
Resulting in a total of 42 features per sample.
### Step 2: Feature Normalization
2.1 Normalize the extracted features to ensure consistency and facilitate model training.
2.2 Normalize each feature to have zero mean and unit variance.
### Step 3: Hybrid Model Construction
3.1 Train Gaussian Mixture Models (GMM) for each emotion class using the normalized feature vectors.
3.2 Use GMM to predict the probabilities of each emotion class for the feature vectors.
### Step 4: Feature Engineering for KNN
4.1 Use the predicted probabilities from GMM as additional features for KNN.
### Step 5: K-Nearest Neighbours Classification
5.1 Train K-Nearest Neighbours (KNN) classifier using the augmented feature vectors.
5.2 Use the Euclidean distance metric to identify the nearest neighbours of a given feature vector.
5.3 Assign the majority class label among its neighbours as the predicted emotion.
### Step 6: Model Evaluation
6.1 Evaluate the trained hybrid GMM-KNN model on a separate test dataset.
6.2 Assess the performance metrics such as accuracy, precision, recall, and F1-score.
6.3 Analyse the model's performance in accurately recognizing emotions from speech samples.
## Gaussian Mixture Models (GMM):
GMM Training: GMMs are statistical models that represent a distribution of data points as a sum of several Gaussian distributions (bell curves). For emotion recognition, a separate GMM is trained for each emotion category (e.g., Happy, Sad, Angry, Fear, Disgust, Surprise) using speech data labelled with those emotions.
Emotion Classification: When presented with new speech data, the system calculates the probability of that data belonging to each of the emotionspecific GMMs. The emotion with the highest probability is considered the recognized emotion.
## K-Nearest Neighbour (KNN):
Training: A large dataset of speech recordings with corresponding text transcripts is used for training. Each recording is converted into a feature vector representing its characteristics.
Recognition: When presented with new speech, KNN finds the k nearest neighbour (most similar speech samples) in the training data based on feature distance (often Euclidean distance). 
## Dataset:
Dataset link: https://www.kaggle.com/datasets/ryersonmultimedialab/ryerson-emotion-database
The Dataset consists of all the videos formats .avi (audio and video format) there. These audio recordings often include individuals speaking with varying emotional states, which serve as the basis for emotion recognition. The data is collected from the Ryerson Multimedia Lab database. The dataset consists of emotions, the emotions are generally Anger, Disgust, Fear, Happiness, Sadness, Surprise. It Has 6 motions based on human audio. There emotions are different from each other because of their Frequency, pitch etc. The RML emotion database is a diverse collection of audio-visual samples collected from eight subjects speaking six different languages, including English, Mandarin, Urdu, Punjabi, Persian, and Italian. The dataset also includes various accents of English and Mandarin, adding to its diversity. Some subjects have facial hair, further increasing variability. To ensure accurate expression of human emotion, samples were selected based on listening tests by multiple human subjects unfamiliar with the corresponding languages, adding rigor to the dataset's emotional authenticity. The dataset consists of around 500 samples for training and testing, each portraying one of the six principal emotions: happiness, sadness, anger, fear, surprise, and
disgust. Each sample has an average duration of around 5 seconds, with audio and visual components synchronized in time. The audio was recorded at a sampling rate of 22050 Hz, while the video was recorded at a frame rate of 30 frames per second (fps), ensuring high-quality recordings. The dataset provides a rich resource for developing and evaluating speech emotion recognition systems, with its diverse linguistic, cultural, and emotional content.
## Performance Evaluation Parameters
The evaluation parameters for the KNN classifier are given below:
1. Precision: For a specific emotion class, precision tells you the proportion of samples identified as that emotion that are actually correct. It's calculated as True Positives divided by (True Positives + False Positives).
2. Recall: Also, for a specific emotion class, recall reflects how well the system identifies all the relevant cases. It's calculated as True Positives divided by (True Positives + False Negatives).
3. F1-Score: This metric combines precision and recall into a single value, providing a balanced view of the system's performance. It's calculated as the harmonic mean of precision and recall: 2 * (Precision * Recall) / (Precision + Recall).
4. Confusion Matrix: This is a table that shows how many samples from each emotion class were actually classified into each category. It provides a visual breakdown of the system's performance for different emotion types.
## Software and Hardware Setup
The software and hardware requirements for the project are as follows:
### Hardware Requirements:
1. CPU Processor (Intel i3 or above)
2. Memory (RAM) (4GB or above)
3. Graphics Processing Unit (GPU)
4. Network Connectivity
### Software Requirements:
1. Windows Operating System
2. Programming Language (Python)
3. Feature Extraction Tools (MFCC, HNR, ZCR, TEO)
4. Flask server
5. Reactjs

## Result and Analysis
The outcome of the proposed system is a substantial improvement in the accuracy of Speech Emotion Recognition (SER).

 ![image](https://github.com/AbhayShukla57/Speech-Emotion-Recognition-using-GMM-and-KNN/assets/110044415/6560267e-7ceb-4106-8599-a39a49989b57)

Figure 6.1: Confusion Matrix

The confusion matrix provides a visual representation of the performance of a classification model. From the confusion matrix, it is evident that the system performs well in correctly classifying instances across most emotion categories, with relatively few misclassifications. Notably, the majority of instances are correctly classified along the diagonal, indicating strong performance. It is observed that the model performs well in correctly identifying instances of 'Angry', 'Disgust', and 'Happiness', as indicated by the high diagonal values in these rows. However, it struggles more with classifying 'Fear' and 'Sadness', as evidenced by the off-diagonal values in these rows.

Table 6.1: Comparison between Existing System (SVM) and Proposed System (GMM & KNN)

![image](https://github.com/AbhayShukla57/Speech-Emotion-Recognition-using-GMM-and-KNN/assets/110044415/e65fdeb9-21df-4225-b831-f97e2722bcf4)


Table 6.2: Classification Report

![image](https://github.com/AbhayShukla57/Speech-Emotion-Recognition-using-GMM-and-KNN/assets/110044415/777a8cd0-a23c-4abd-bfea-e5f2a251261c)

The classification report provides a summary of the classification performance across different emotion classes. It includes metrics such as precision, recall, and F1-score, which assess the model's accuracy in classifying each emotion class. Additionally, it reports the overall accuracy of the model across all classes.
Examining the precision scores, it is observed that the system performs well in identifying the 'Disgust' and 'Happiness' emotions, with precision scores of 0.92 and 0.82, respectively. This suggests that when the system predicts these emotions, it is highly likely to be correct. However, the precision for 'Sadness' is relatively lower at 0.67, indicating that the system may sometimes misclassify other emotions as 'Sadness'.
Similarly, the recall scores provide insight into the system's ability to correctly identify instances of each emotion class. Notably, 'Angry' and 'Happiness' have high recall scores of 0.92 and 0.90, respectively, indicating that the system effectively captures most instances of these emotions. However, the recall for 'Fear' is relatively lower at 0.67, indicating that the system may miss some instances of this emotion.
The F1-score, which is the harmonic mean of precision and recall, provides a balanced measure of the system's performance. Overall, the system achieves a macro-average F1-score of 0.82, indicating good overall performance across all emotion classes. Further analysis reveals that while the model demonstrates robust performance, there may be room for improvement in correctly classifying 'Fear' and 'Sadness' emotions. This enhanced accuracy underscores the system's capability to achieve more nuanced and precise recognition of complex and blended emotions within spoken languages.
