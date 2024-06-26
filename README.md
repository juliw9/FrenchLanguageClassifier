# CamemBERT Model
## Introduction
The goal of the project is to predict the level of difficulty of a sentence in French language. Difficulty levels include A1, A2, B1, B2, C1 and C2. In order to navigate the github page, there are three sections, including code, data, and streamlit app. The code sections contains the final version of the code in its significant parts. Every part represents one ML model. All units combined contributed to the accuracy of 0.618 which placed team Hublot at the fifth place. The data section contains two files of training data (what the model was trained on) and the unlabelled test data (what model was tested on to achieve the end accuracy). The folder StreamlitApp contains the remainders of our first try to create a streamlit app.

There is some training data given in order to train the model which is done with 80% of this data. The leftover randomly chosen 20% is used for testing. Additionally, there is some unlabelled test data that the model was tested on to achieve some general metrics associated with the testing. These metrics include precision, recall, F1 score and accuracy.

## Pre-developed classifiers
Before considering deeper analysis of this project, pre-developed classifiers were implemented into prediction, including KNN, Decision Tree, Random Forest and Logistic Regression. Metric results of these classifiers are included in the table below. KNN performs the worst among the classifiers in this scenario. KNN classifier relies on the local structure of the data. It may not capture complex relationships between sentences and difficulties, leading to poorer performance compared to others. Decision Tree performs similarly to KNN but slightly better. Decision Trees can overfit easily while dealing with high-dimensional data such as text. This likely caused poorer generalization and performance on unlabelled data. Logistic Regression performs better than KNN and Decision Tree with an accuracy of 0.45. Logistic Regression is well-suited for multiclass classification (six difficulty levels) when used with multinomial logistic regression. One-vs-Rest (OvR) technique within logistic regression did not perform better than multinomial due to worse smoothness of decision boundaries and imbalanced class distributions for individual binary classifiers. The CEFR levels have a natural ordinal relationship, indicating a progression in language proficiency from A1 to C2. Multinomial logistic regression considers the ordinal relationship among the classes and can model the probability of each class directly, capturing the inherent order. The smoothness of the boundaries is beneficial when the classes are not well-separated, as is often the case in ordinal classification with smaller number of classes (only six in this case). A single softmax function created linear decision boundary which turned out to be effective in capturing the patterns in the data. However, Random Forest performed slightly outperformed Logistic Regression with 0.47 accuracy. The reason is because it can handle non-linear relationships and feature interactions better due to its ensemble nature. The ability to aggregate multiple decision trees reduces overfitting and improves generalization, which is beneficial for complex datasets like text data. This reduces overfitting, which individual decision trees can suffer from, leading to better performance on unseen data and higher F1 score. Random Forest can identify and prioritize important features, which helps in making more accurate predictions. This can be particularly useful in text data, where certain words or phrases might be strong indicators of sentence difficulty. Although Random Forest gave quite satisfying results for such a short amount of code, it was only the motivation and the beginning of journey to achieve higher accuarcy. 

![Classifiers](https://github.com/juliw9/FrenchLanguageClassifier/assets/161482444/1c371c34-7259-4097-a0ab-e91b4a82672f)

Results from the table are optimized using grid search function for the best optimal parameters.
Here is a list of optimization parameters for each classifier:

Logistic Regression:

C: Inverse of regularization strength. <br>
solver: Algorithm to use in the optimization problem. <br>
penalty: Type of regularization ('l2', 'none'). <br>

Random Forest:

n_estimators: Number of trees in the forest. <br>
max_depth: Maximum depth of the tree. <br>
min_samples_split: Minimum number of samples required to split an internal node. <br>
min_samples_leaf: Minimum number of samples required to be at a leaf node. <br>
max_features: Number of features to consider when looking for the best split. <br>

K-Nearest Neighbors (KNN):

n_neighbors: Number of neighbors to use. <br>

Decision Tree:

max_depth: Maximum depth of the tree. <br>
min_samples_split: Minimum number of samples required to split an internal node. <br>
min_samples_leaf: Minimum number of samples required to be at a leaf node. <br>
criterion: Function to measure the quality of a split ('entropy').

## About the Model
The model that team Hublot created uses CamemBERT model, specifically designed for classifying difficulty levels of French language. The CamemBERT model, introduced by Louis Martin et al., is a state-of-the-art French language model based on Facebook’s RoBERTa, trained on 138GB of French text. It addresses the limitations of existing models that are predominantly trained on English or multilingual data, offering superior performance in French-specific NLP tasks such as part-of-speech tagging, dependency parsing, named-entity recognition, and natural language inference. CamemBERT sets new benchmarks in these areas and is publicly available to support further research and applications in French NLP. 

![Camembert_Model](https://github.com/juliw9/FrenchLanguageClassifier/assets/161482444/35073009-6d72-4a9f-a219-f9e25af28953)

## Final Version of the Model
The final version of the actual model contains features that improve the simple model. This editing required significant number of libraries. Two utility functions were defined for converting between difficulty scores and numeric representations and vice versa. This was done to make the comparison easier. The data augmentation process in the code generates augmented versions of a given text by applying various operations that include synonym replacement, random deletion, random swap, and random insertion of words. The generate_augmented_sentences function combines these methods to create multiple augmented sentences, enhancing the diversity of the training data and potentially improving the generalization and performance of the natural language model. New function is introduced to evaluate the CamemBERT model. This function takes a configuration file and a model file of a CamemBERT model. It uses these files to load a pre-trained model. Then, it evaluates the model's prediction on a given input text. This involves processing the text using the model's tokenizer, passing it through the model, and obtaining the predicted class index based on the model's output. A custom dataset class called 'LanguageLevelDataset' is defined and used to prepare data for training a model. The class takes input texts, corresponding labels, a tokenizer, and a maximum length parameter. It determines the length of the dataset and retrieves individual data samples. Each data sample consists of the input text encoded using the provided tokenizer, along with its corresponding label, both formatted as tensors. Custom loss function was used to adjust the standard cross-entropy loss by giving more weight to hard-to-classify examples. It calculates focal loss based on the input predictions and target labels, allowing for customization of alpha and gamma parameters to control the loss function's behavior. The loss can be computed as a mean, sum, or without reduction, depending on the specified reduction parameter. Once these functions have been created, the actual running code begins. It begins by setting up hyperparameters, loading training and test data, and initializing the model and optimizer. Hyperparameter modification played a key role in increasing the accuracy of the model. Larger number of epochs seemed to improve the accuracy. However, it does so up to certain treshold where it starts decreasing later on. Data augmentation increases the diversity of the training data. The augmented and original data were combined. Each training epoch consisting of batches of sentences passed through the model. Batches were also optimized based on the number of epochs for better performance. The number of iterations was set to be large but on purpose since not all of them were used. The reason behind this is that the focal loss function was used as the optimization criterion, aiming to minimize the loss between predicted and true labels. During training, if the loss improved, the model was evaluated on the test data. If the performance surpassed a certain threshold, the model parameters were saved. This process continued until a predefined number of iterations or until the loss stops improving. Finally, the predicted language levels for the test data were exported to a CSV file for submission. This model had an accuracy of 0.65 during the test train split, which is far beyond the pre-developed models initially used. The submission accuracy on unlabelled test data was 0.618.

## External Links
The link to our streamlit app is provided here:
https://frenchlanguageclassifier-ugs2o2engy3ge2zahjymwn.streamlit.app

The link to our YouTube video is provided here: https://www.youtube.com/watch?v=zJF8yJVpZNc

The link to our huggingface: https://huggingface.co/juliw9/FrenchModel/tree/main

## Splitting the Tasks
This project was realized by Lazar Aleksic and Julian Wirtz as part of the course "Data Science and Machine Learning" by Michalis Vlachos. We jointly designed all codes evaluating the difficulty levels, with Lazar focussing more on the fine-tuning of the more first four models, while Julian focussed more on the fine-tuning of the CamemBERT model and its implementation to Streamlit. Lazar took care of the design of this github and the creation of the video. 
