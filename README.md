[![Python](https://img.shields.io/badge/Python-3.8.10-%233776AB?logo=Python)](https://www.python.org/)

# Framework for fairness analysis and mitigation for Graph Neural Network-based user profiling models
Repository of the the Master Thesis Design and development of a standardised framework for fairness analysis and mitigation for Graph Neural Network-based user profiling models.

![Framework_final](https://user-images.githubusercontent.com/45569039/213778147-d8488eb0-965d-46ec-a526-1f1f942dfb18.jpg)

## Abstract
User profiling classification has been a very popular problem in the last year in regards to machine learning, which concentrates on classifying users into a specific category.
After the introduction of Graph Neural Networks (GNNs) in last years, user profiling classification has been represented as a node classification problem to better understand and account for the relationship between users.
This paved the way to many new state of the art GNN models structures to solve the user profiling problem while concentrating on many different aspects, which gave the user so many options to consider from.
Additionally, most of these models only concentrate on evaluating how good is the model prediction, while neglecting the model fairness.
In this work we design and develop a novel framework for fairness analysis and mitigation based on user profiling classification.
The framework goal is allow users to better analyze and compare different user profiling models at the same time, making it easier for users to choose the best suitable model for them. 
Since every model requires a different input data type structure, we overcome this problem by designing a standardised pre-processing approach which makes it easier for the user to train several state of the art GNN models sequentially using only a single data type structure.
To this end, we also conducted a series of preliminary experiments to compare the fairness of several state of the art GNN models using several pre-processing debiasing approaches.
We evaluate the model fairness using disparate impact and disparate mistreatment metrics and observe that it is possible to achieve fairer GNN models predictions using some of the debiasing approaches in some cases.

## Requirements
TBA
