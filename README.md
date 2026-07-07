# movie-recommendation-system
A Python-based movie recommendation system built to analyse viewer data and deliver personalized movie suggestions as the model learns user preferences from past ratings using collaborative filtering with matrix factorization.
# Features
•	Data Pipeline: Custom PyTorch Dataset loader mapping MovieLens IDs to sequential indices.

•	Deep Learning Model: PyTorch Matrix Factorization leveraging custom nn.Embedding layers.

•	Unsupervised Learning: Scikit-Learn KMeans implementation grouping movie latent variables into thematic clusters.
# Technologies Used

•	Python

•	PyTorch

•	Pandas

•	NumPy

•	Scikit-learn

•	MovieLens Dataset

# Dataset
The project uses the MovieLens Latest Small dataset, which contains:
•	Movie information
•	User ratings
•	Thousands of interactions between users and movies

# How It Works
Load and clean the MovieLens dataset.
Convert user IDs and movie IDs into numerical indices.
Train a matrix factorization model using user and movie embeddings.
Optimize the model with Adam and Mean Squared Error loss.
Extract movie embeddings after training.
Apply K-Means clustering to group similar movies.
# Project Structure
MovieRecommendationSystem/

│

├── MovieRecommenderSystem.ipynb

├── movies.csv

├── ratings.csv

├── ml-latest-small.zip

└── README.md
# How to Run
Ensure your MovieLens .csv data files are present in the directory.
Launch Jupyter Notebook or open the notebook file inside VS Code.
Select your active Python interpreter environment kernel.
Execute all cells sequentially.
# What I Learned
Collaborative Filtering: Built and optimized recommendation engines designed to learn latent user-item preferences.

PyTorch Embeddings: Leveraged PyTorch embedding layers to map discrete user and movie IDs into continuous, low-dimensional vector spaces.

Deep Learning Pipelines: Trained neural network architectures on real-world datasets, handling custom data loaders and feature engineering.

Unsupervised Clustering: Applied K-Means clustering to analyze and interpret the learned embedding spaces, grouping similar items together based on learned characteristics.

Data Engineering & Evaluation: Designed robust data preprocessing workflows and implemented proper evaluation metrics to measure recommendation performance.
# Future Improvements

•	Generate personalized Top-N movie recommendations

•	Add a simple web interface using Streamlit

•	Compare different recommendation algorithms

•	Improve model accuracy through hyperparameter tuning

# Academic Project
Developed as a university semester project, this repository explores collaborative filtering recommendation systems using PyTorch. The project provides hands-on implementation of data preprocessing, embedding-based deep learning models, and unsupervised learning techniques
