import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("jobstreet.csv")

# Vectorize 'categoriesName'
vectorizer = TfidfVectorizer(stop_words='english')
categories_vectors = vectorizer.fit_transform(data['categoriesName'])

# Find the optimal number of clusters using the Elbow Method
inertia = []
K_range = range(1, 11)  # Testing up to 10 clusters; adjust as needed

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(categories_vectors)
    inertia.append(kmeans.inertia_)

# Plot the Elbow graph
plt.plot(K_range, inertia, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method to Determine Optimal K')
plt.show()

# Once the plot is reviewed, choose the optimal K (e.g., 4 as an example here)
optimal_k = 4  # Replace with the observed optimal K from the Elbow plot

# Perform KMeans clustering with the optimal K
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
data['Cluster'] = kmeans.fit_predict(categories_vectors)

# Export the clustered data to a new CSV
data_sorted = data.sort_values(by='Cluster')
data_sorted.to_csv('jobstreet_tfidf.csv', index=False)

print("Data telah diekspor ke 'jobstreet_tfidf.csv'")

