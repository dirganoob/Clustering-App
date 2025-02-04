import requests
import json
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Mendapatkan data pekerjaan
base_url = "https://id.jobstreet.com/api/jobsearch/v5/search?siteKey=ID-Main&sourcesystem=houston&facets=title&userqueryid=3d6a1753652ffdc7d67afa62d5ad93a4-4642551&userid=5b7c167c-03f7-4b7f-a24f-1bf60f7feb3f&usersessionid=5b7c167c-03f7-4b7f-a95e-50b9b6aa85ed&where=Sulawesi+Selatan&pageSize=100&include=seodata,relatedsearches,joracrosslink,gptTargeting&locale=id-ID"

# Mengumpulkan data dari beberapa halaman
jobs_data = []
for page in range(1, 6):
    url = f"{base_url}&page={page}"
    response = requests.get(url)
    if response.status_code == 200:
        data = json.loads(response.content)
        for job in data["data"]:
            job_dict = {
                "Company": job.get("companyName", ""),
                "Job": job["title"],
                "Listing Date": job["listingDateDisplay"],
                "Location": job["locations"][0]["label"],
                "Work Type": job["workTypes"][0],
                "Teaser job": job["teaser"],
                "Job ID": job["id"],
                "Classification": job["classifications"][0]["classification"]["description"],
                "Sub Classification": job["classifications"][0]["subclassification"]["description"],
                "Job_URL": f"https://id.jobstreet.com/id/job/{job['id']}",  
            }
            jobs_data.append(job_dict)
    else:
        print(f"Error: Failed to retrieve data for page {page}. Status code: {response.status_code}")

# Menyimpan data ke dalam DataFrame
df = pd.DataFrame(jobs_data)

# Fungsi untuk memplot Elbow Method
def plot_elbow_method(data):
    distortions = []
    K = range(1, 4)  # Jumlah cluster dari 1 hingga 10
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        distortions.append(kmeans.inertia_)
    
    # Plot hasil WCSS untuk setiap nilai k
    plt.figure(figsize=(8, 5))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('Jumlah Cluster (k)')
    plt.ylabel('WCSS')
    plt.title('Metode Elbow untuk Menentukan Jumlah Cluster Optimal')
    plt.xticks(K)
    plt.grid()
    plt.show()

    return distortions

# Fungsi clustering
def perform_clustering(df):
    if df is not None and not df.empty:
        # Label Encoding untuk kolom kategorikal
        le = LabelEncoder()
        df['Classification_encoded'] = le.fit_transform(df['Classification'])

        features = df[['Classification_encoded']]

        # Plot Elbow Method
        distortions = plot_elbow_method(features)

        # Tentukan jumlah cluster optimal (contoh sederhana: cari siku)
        optimal_k = distortions.index(min(distortions)) + 1
        print(f"Optimal number of clusters: {optimal_k}")

        # Clustering menggunakan jumlah cluster optimal
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        df['Cluster'] = kmeans.fit_predict(features)

        # Simpan hasil clustering
        df_sorted = df.sort_values(by='Cluster')
        df_sorted.to_csv('job_data_with_clusters.csv', index=False)
        df_sorted.to_excel('job_data_with_clusters.xlsx', index=False)
        print("Clustering completed and data saved with clusters.")
    else:
        print("DataFrame is empty or None, clustering aborted.")

# Lakukan clustering
perform_clustering(df)
