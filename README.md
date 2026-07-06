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
Building recommendation systems with collaborative filtering,
Working with embedding layers in PyTorch,
Training deep learning models on real-world data,
Using K-Means clustering to analyze learned embeddings,
Data preprocessing and model evaluation.
# Future Improvements

•	Generate personalized Top-N movie recommendations

•	Add a simple web interface using Streamlit

•	Compare different recommendation algorithms

•	Improve model accuracy through hyperparameter tuning

# Academic Project
This project was developed as part of a university semester course to explore recommendation systems using collaborative filtering and PyTorch. It provided hands-on experience with data preprocessing, deep learning, embedding-based models, and unsupervised learning techniques.
