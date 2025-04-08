# Silabus Metodologi Penelitian Informatika (14 Pertemuan)

Berikut adalah materi perkuliahan metodologi penelitian informatika untuk 14 pertemuan, termasuk praktik kode untuk beberapa topik yang relevan:

## Pertemuan 1: Pengantar Metodologi Penelitian Informatika
- Konsep dasar penelitian ilmiah
- Perbedaan penelitian informatika dengan bidang lain
- Jenis-jenis penelitian dalam informatika
- Etika penelitian dan integritas akademik
- Pengenalan struktur proposal dan laporan penelitian informatika

## Pertemuan 2: Identifikasi Masalah dan Perumusan Pertanyaan Penelitian
- Teknik mengidentifikasi masalah penelitian
- Cara merumuskan pertanyaan penelitian yang baik
- Menentukan ruang lingkup penelitian
- Latihan: Mengembangkan pertanyaan penelitian dari masalah di bidang informatika
- Praktik: Evaluasi dan perbaikan rumusan pertanyaan penelitian

## Pertemuan 3: Tinjauan Pustaka dan Literature Review
- Strategi pencarian literatur ilmiah
- Database publikasi ilmiah bidang informatika
- Teknik membaca dan menganalisis paper ilmiah
- Cara mensintesis informasi dari berbagai sumber
- Praktik: Pencarian literatur menggunakan Google Scholar, IEEE Xplore, ACM Digital Library

## Pertemuan 4: Metodologi Penelitian Kuantitatif dalam Informatika
- Desain eksperimen dalam penelitian informatika
- Metode pengambilan sampel
- Teknik analisis data kuantitatif
- Praktik kode: Pengenalan analisis statistik dengan Python

```python
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Contoh data performa algoritma
data = {
    'Algoritma A': np.random.normal(100, 15, 30),
    'Algoritma B': np.random.normal(110, 15, 30)
}

df = pd.DataFrame(data)

# Visualisasi perbandingan
df.boxplot()
plt.ylabel('Waktu Eksekusi (ms)')
plt.title('Perbandingan Performa Algoritma')
plt.show()

# Uji statistik (t-test)
t_stat, p_value = stats.ttest_ind(df['Algoritma A'], df['Algoritma B'])
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")
if p_value < 0.05:
    print("Terdapat perbedaan signifikan antara dua algoritma")
else:
    print("Tidak ada perbedaan signifikan antara dua algoritma")
```

## Pertemuan 5: Metodologi Penelitian Kualitatif dalam Informatika
- Teknik wawancara dan observasi
- Metode analisis data kualitatif
- Studi kasus dalam penelitian informatika
- Grounded theory dalam konteks informatika
- Praktik: Merancang protokol wawancara untuk penelitian UX

## Pertemuan 6: Desain Eksperimen dalam Penelitian Informatika
- Jenis-jenis desain eksperimen
- Variabel penelitian: dependen, independen, dan kontrol
- Validitas internal dan eksternal
- Menghindari bias dalam eksperimen
- Praktik kode: Simulasi eksperimen dengan Python

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Simulasi eksperimen untuk perbandingan algoritma machine learning
np.random.seed(42)

# Simulasi dataset
n_samples = 1000
X = np.random.randn(n_samples, 5)  # 5 fitur
y = 3*X[:,0] + 2*X[:,1] - X[:,2] + 0.5*np.random.randn(n_samples)  # Target dengan noise

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Jumlah data training: {X_train.shape[0]}")
print(f"Jumlah data testing: {X_test.shape[0]}")

# Visualisasi distribusi data
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.hist(y_train, bins=30)
plt.title('Distribusi Data Training')
plt.subplot(1, 2, 2)
plt.hist(y_test, bins=30)
plt.title('Distribusi Data Testing')
plt.tight_layout()
plt.show()
```

## Pertemuan 7: Metode Pengumpulan dan Analisis Data
- Teknik pengumpulan data dalam informatika
- Preprocessing dan cleaning data
- Metode validasi data
- Analisis data menggunakan R atau Python
- Praktik kode: Data preprocessing dengan Pandas

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Simulasi dataset dengan missing values dan outliers
np.random.seed(0)
n = 100

# Buat data dengan masalah umum
data = {
    'user_id': range(1, n+1),
    'age': np.random.randint(18, 70, n),
    'gender': np.random.choice(['M', 'F', None], n, p=[0.48, 0.48, 0.04]),
    'usage_time': np.random.normal(120, 30, n),
    'satisfaction': np.random.randint(1, 6, n)
}

# Tambahkan outliers
data['usage_time'][0] = 500  # outlier
data['usage_time'][1] = -50  # invalid value
data['usage_time'][2:5] = np.nan  # missing values

# Buat dataframe
df = pd.DataFrame(data)
print("Data awal:")
print(df.head())
print("\nStatistik deskriptif:")
print(df.describe())
print("\nMissing values:")
print(df.isnull().sum())

# Cleaning dan preprocessing
# 1. Handle missing values
imputer = SimpleImputer(strategy='mean')
df['usage_time'] = imputer.fit_transform(df[['usage_time']])

# 2. Handle outliers dengan metode IQR
Q1 = df['usage_time'].quantile(0.25)
Q3 = df['usage_time'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df['usage_time_cleaned'] = df['usage_time'].clip(lower_bound, upper_bound)

# 3. Encode categorical variables
df['gender'] = df['gender'].fillna('Unknown')
encoder = OneHotEncoder(sparse=False)
gender_encoded = encoder.fit_transform(df[['gender']])
gender_df = pd.DataFrame(gender_encoded, columns=encoder.get_feature_names_out(['gender']))

# Gabungkan hasil
df_processed = pd.concat([df, gender_df], axis=1)

print("\nData setelah preprocessing:")
print(df_processed.head())

# Visualisasi hasil cleaning
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.boxplot(df['usage_time'])
plt.title('Usage Time Sebelum Cleaning')
plt.subplot(1, 2, 2)
plt.boxplot(df['usage_time_cleaned'])
plt.title('Usage Time Setelah Cleaning')
plt.tight_layout()
plt.show()
```

## Pertemuan 8: Evaluasi Algoritma dan Benchmark
- Teknik evaluasi algoritma
- Metrik evaluasi untuk berbagai jenis penelitian
- Cross-validation dan holdout methods
- Benchmark tools dan frameworks
- Praktik kode: Evaluasi model machine learning

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model training dan evaluasi
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True)
}

results = {}

for name, model in models.items():
    # Train model
    model.fit(X_train, y_train)
    
    # Prediksi
    y_pred = model.predict(X_test)
    
    # Evaluasi
    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "Cross Val Score": np.mean(cross_val_score(model, X, y, cv=5))
    }

# Tampilkan hasil
results_df = pd.DataFrame(results).T
print("Hasil evaluasi model:")
print(results_df)

# Visualisasi perbandingan model
plt.figure(figsize=(12, 6))
results_df.plot(kind='bar', figsize=(12, 6))
plt.title('Perbandingan Performa Model')
plt.ylabel('Score')
plt.xlabel('Model')
plt.xticks(rotation=0)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# Confusion matrix untuk model terbaik
best_model_name = results_df['Accuracy'].idxmax()
best_model = models[best_model_name]
y_pred = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix - {best_model_name}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()
```

## Pertemuan 9: Penulisan Proposal Penelitian
- Komponen utama proposal penelitian
- Teknik penulisan latar belakang dan signifikansi
- Merumuskan tujuan dan kontribusi penelitian
- Format penulisan referensi (IEEE, ACM)
- Praktik: Workshop penyusunan proposal penelitian

## Pertemuan 10: Metode Pengembangan Software dalam Penelitian
- Model pengembangan software untuk penelitian
- Prototyping dan proof of concept
- Dokumentasi kode dan reproduksibilitas
- Praktik kode: Implementasi proof of concept dengan Python

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Implementasi proof of concept: Sistem rekomendasi sederhana berbasis clustering

# 1. Generate data simulasi preferensi pengguna
np.random.seed(42)
n_users = 200
n_features = 5  # Misal: rating kategori buku/film

# Simulasi preferensi pengguna
user_preferences = np.random.randint(1, 6, size=(n_users, n_features))
user_ids = [f"user_{i}" for i in range(n_users)]
feature_names = [f"category_{i}" for i in range(n_features)]

# Buat dataframe
df = pd.DataFrame(user_preferences, index=user_ids, columns=feature_names)
print("Data preferensi pengguna:")
print(df.head())

# 2. Preprocessing
scaler = StandardScaler()
scaled_preferences = scaler.fit_transform(user_preferences)

# 3. Implementasi clustering untuk segmentasi pengguna
# Tentukan jumlah cluster optimal dengan elbow method
inertia = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_preferences)
    inertia.append(kmeans.inertia_)

# Visualisasi elbow method
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method untuk Jumlah Cluster Optimal')
plt.xlabel('Jumlah Cluster')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

# Pilih jumlah cluster berdasarkan elbow method
optimal_k = 4  # Anggap kita pilih 4 dari elbow method
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
cluster_labels = kmeans.fit_predict(scaled_preferences)

# Tambahkan label cluster ke dataframe
df['cluster'] = cluster_labels
print("\nData dengan label cluster:")
print(df.head())

# 4. Visualisasi hasil clustering dengan PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_preferences)

plt.figure(figsize=(12, 8))
for cluster in range(optimal_k):
    plt.scatter(pca_result[cluster_labels == cluster, 0], 
                pca_result[cluster_labels == cluster, 1],
                label=f'Cluster {cluster}')

plt.title('Visualisasi Segmentasi Pengguna dengan PCA')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.grid(True)
plt.show()

# 5. Analisis karakteristik cluster
cluster_profiles = df.groupby('cluster').mean()
print("\nProfil rata-rata setiap cluster:")
print(cluster_profiles)

# Visualisasi profil cluster
plt.figure(figsize=(12, 8))
cluster_profiles.T.plot(kind='bar')
plt.title('Profil Preferensi Rata-rata per Cluster')
plt.ylabel('Rating Rata-rata')
plt.xlabel('Kategori')
plt.legend(title='Cluster')
plt.grid(True)
plt.tight_layout()
plt.show()

# 6. Implementasi fungsi rekomendasi sederhana
def recommend_for_user(user_id, df, cluster_profiles, top_n=2):
    """
    Fungsi untuk memberikan rekomendasi kategori berdasarkan cluster pengguna
    """
    # Ambil cluster pengguna
    user_cluster = df.loc[user_id, 'cluster']
    
    # Ambil preferensi pengguna
    user_prefs = df.loc[user_id, feature_names]
    
    # Ambil rata-rata preferensi cluster
    cluster_prefs = cluster_profiles.loc[user_cluster]
    
    # Cari kategori yang belum disukai oleh pengguna tapi disukai oleh cluster
    potential_recommendations = []
    
    for category in feature_names:
        # Jika pengguna memberi rating rendah tetapi cluster memberi rating tinggi
        if user_prefs[category] <= 2 and cluster_prefs[category] >= 4:
            potential_recommendations.append((category, cluster_prefs[category]))
    
    # Urutkan rekomendasi berdasarkan preferensi cluster
    recommendations = sorted(potential_recommendations, key=lambda x: x[1], reverse=True)
    
    return recommendations[:top_n]

# Demo rekomendasi
test_user = 'user_42'
recommendations = recommend_for_user(test_user, df, cluster_profiles)

print(f"\nRekomendasi untuk {test_user}:")
print(f"Preferensi pengguna: {df.loc[test_user, feature_names].to_dict()}")
print(f"Cluster pengguna: {df.loc[test_user, 'cluster']}")
print(f"Rekomendasi kategori: {recommendations}")
```

## Pertemuan 11: Teknik Visualisasi Data dalam Penelitian
- Prinsip-prinsip visualisasi data efektif
- Tools visualisasi data (Matplotlib, ggplot, D3.js)
- Visualisasi untuk berbagai jenis data
- Praktik kode: Visualisasi data penelitian dengan Python

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import learning_curve

# Simulasi data hasil eksperimen
np.random.seed(42)

# 1. Buat data simulasi: perbandingan beberapa algoritma pada berbagai ukuran dataset
algorithms = ['Algorithm A', 'Algorithm B', 'Algorithm C', 'Algorithm D']
dataset_sizes = [1000, 2000, 5000, 10000, 20000]

# Simulasi runtime (ms)
runtime_data = {
    'Algorithm A': [10, 22, 53, 105, 210],  # O(n)
    'Algorithm B': [20, 80, 500, 2000, 8000],  # O(n²)
    'Algorithm C': [50, 60, 75, 90, 110],  # O(log n)
    'Algorithm D': [30, 30, 35, 35, 40]  # O(1) dengan overhead
}

# Simulasi akurasi (%)
accuracy_data = {
    'Algorithm A': [85, 87, 89, 90, 91],
    'Algorithm B': [88, 90, 92, 94, 95],
    'Algorithm C': [82, 84, 85, 86, 86],
    'Algorithm D': [80, 80, 81, 81, 82]
}

# 2. Visualisasi kompleksitas runtime
plt.figure(figsize=(12, 6))
for algo in algorithms:
    plt.plot(dataset_sizes, runtime_data[algo], marker='o', linewidth=2, label=algo)

plt.title('Perbandingan Runtime Algoritma', fontsize=14)
plt.xlabel('Ukuran Dataset', fontsize=12)
plt.ylabel('Runtime (ms)', fontsize=12)
plt.xscale('log')
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# 3. Visualisasi trade-off runtime vs akurasi
plt.figure(figsize=(10, 8))
for i, size in enumerate(dataset_sizes):
    plt.scatter([runtime_data[algo][i] for algo in algorithms],
               [accuracy_data[algo][i] for algo in algorithms],
               s=100, alpha=0.7, label=f'n={size}')
    
    # Tambahkan label algoritma
    for j, algo in enumerate(algorithms):
        plt.annotate(algo, 
                   (runtime_data[algo][i], accuracy_data[algo][i]),
                   xytext=(5, 5), textcoords='offset points')

plt.title('Trade-off Runtime vs Akurasi', fontsize=14)
plt.xlabel('Runtime (ms)', fontsize=12)
plt.ylabel('Akurasi (%)', fontsize=12)
plt.xscale('log')
plt.grid(True, alpha=0.3)
plt.legend(title='Ukuran Dataset')
plt.tight_layout()
plt.show()

# 4. Visualisasi perbandingan performa dengan heatmap
# Buat dataframe untuk heatmap performa algoritma pada berbagai kriteria evaluasi
criteria = ['Akurasi', 'Runtime', 'Memori', 'Skalabilitas', 'Stabilitas']
performance_scores = pd.DataFrame({
    'Algorithm A': [4, 5, 3, 4, 4],
    'Algorithm B': [5, 2, 4, 2, 5],
    'Algorithm C': [3, 4, 5, 4, 3],
    'Algorithm D': [2, 5, 5, 5, 3]
}, index=criteria)

plt.figure(figsize=(10, 8))
sns.heatmap(performance_scores, annot=True, cmap='viridis', linewidths=0.5, fmt='d')
plt.title('Perbandingan Performa Algoritma (Skala 1-5)', fontsize=14)
plt.tight_layout()
plt.show()

# 5. Visualisasi learning curve
# Fungsi helper untuk membuat learning curve sintetis
def plot_learning_curve(train_sizes, train_scores, test_scores, title, ylim=None, figsize=(10, 6)):
    plt.figure(figsize=figsize)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    
    plt.title(title, fontsize=14)
    plt.xlabel("Training examples", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    
    if ylim is not None:
        plt.ylim(*ylim)
        
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    
    return plt

# Simulasi data learning curve untuk dua algoritma
train_sizes = np.linspace(0.1, 1.0, 10)

# Algoritma 1: Underfitting
train_scores_1 = np.array([np.random.normal(0.60 + i*0.03, 0.02, 3) for i in range(len(train_sizes))])
test_scores_1 = np.array([np.random.normal(0.55 + i*0.03, 0.03, 3) for i in range(len(train_sizes))])

# Algoritma 2: Overfitting
train_scores_2 = np.array([np.random.normal(0.70 + i*0.03, 0.02, 3) for i in range(len(train_sizes))])
test_scores_2 = np.array([np.random.normal(0.65 + i*0.01, 0.03, 3) for i in range(len(train_sizes))])
test_scores_2[:5] = test_scores_2[:5] - 0.15  # Tambahkan gap untuk ilustrasi overfitting

# Plot learning curve
plot_learning_curve(train_sizes, train_scores_1, test_scores_1, "Learning Curve: Algoritma 1", ylim=(0.5, 1.01))
plt.show()

plot_learning_curve(train_sizes, train_scores_2, test_scores_2, "Learning Curve: Algoritma 2", ylim=(0.5, 1.01))
plt.show()
```

## Pertemuan 12: Teknik Validasi Penelitian
- Keabsahan dan keandalan penelitian
- Triangulasi metode
- Validasi silang (Cross-validation)
- Teknik peer review dan evaluasi pakar
- Praktik: Implementasi validasi silang untuk model prediktif

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import KFold, cross_val_score, learning_curve
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# Standardisasi fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 1. Implementasi K-Fold Cross Validation
models = {
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "SVR": SVR(kernel='rbf', C=1.0, epsilon=0.1)
}

# Jumlah fold
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

results = {}
cv_results = {}

for name, model in models.items():
    # Cross validation menggunakan MSE
    cv_scores = -cross_val_score(model, X_scaled, y, cv=kf, scoring='neg_mean_squared_error')
    
    # Cross validation menggunakan R2
    r2_scores = cross_val_score(model, X_scaled, y, cv=kf, scoring='r2')
    
    results[name] = {
        'MSE': cv_scores.mean(),
        'MSE_std': cv_scores.std(),
        'R2': r2_scores.mean(),
        'R2_std': r2_scores.std()
    }
    
    # Simpan hasil per fold untuk visualisasi
    cv_results[name] = {
        'MSE': cv_scores,
        'R2': r2_scores
    }

# Tampilkan hasil
results_df = pd.DataFrame({
    model: {
        'MSE': f"{values['MSE']:.2f} ± {values['MSE_std']:.2f}",
        'R2': f"{values['R2']:.3f} ± {values['R2_std']:.3f}"
    }
    for model, values in results.items()
})

print("Hasil Cross-Validation:")
print(results_df)

# 2. Visualisasi hasil cross-validation
plt.figure(figsize=(14, 6))

# Plot MSE
plt.subplot(1, 2, 1)
for i, (name, scores) in enumerate(cv_results.items()):
    plt.boxplot(scores['MSE'], positions=[i], widths=0.6)
plt.xticks(range(len(models)), models.keys(), rotation=45)
plt.title('MSE per Model (Cross-Validation)')
plt.ylabel('Mean Squared Error')
plt.grid(True, alpha=0.3)

# Plot R2
plt.subplot(1, 2, 2)
for i, (name, scores) in enumerate(cv_results.items()):
    plt.boxplot(scores['R2'], positions=[i], widths=0.6)
plt.xticks(range(len(models)), models.keys(), rotation=45)
plt.title('R² per Model (Cross-Validation)')
plt.ylabel('R² Score')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 3. Learning curve untuk model terbaik
# Pilih model dengan R2 tertinggi
best_model_name = max(results, key=lambda x: results[x]['R2'])
best_model = models[best_model_name]

# Generate learning curve
train_sizes, train_scores, test_scores = learning_curve(
    best_model, X_scaled, y, cv=5, n_jobs=-1, 
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='r2')

# Plot learning curve
plt.figure(figsize=(10, 6))
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.grid()
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
plt.title(f"Learning Curve ({best_model_name})")
plt.xlabel("Training samples")
plt.ylabel("R² Score")
plt.legend(loc="best")
plt.ylim(0, 1)
plt.show()

print(f"\nModel terbaik berdasarkan R²: {best_model_name}")
print(f"R² Score: {results[best_model_name]['R2']}")
```

## Pertemuan 13: Penulisan Naskah Ilmiah dan Publikasi
- Struktur paper ilmiah (IEEE, ACM format)
- Teknik penulisan abstrak dan pendahuluan
- Menyajikan hasil dan diskusi
- Proses review dan publikasi ilmiah
- Praktik: Workshop penulisan naskah ilmiah

## Pertemuan 14: Presentasi Hasil Penelitian
- Teknik presentasi ilmiah yang efektif
- Persiapan dan pelaksanaan presentasi
- Menjawab pertanyaan dan diskusi
- Presentasi poster ilmiah
- Praktik: Simulasi presentasi hasil penelitian
