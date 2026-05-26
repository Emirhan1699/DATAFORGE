# DATAFORGE
## Tarımsal Üretim Tahmin Modeli (XGBoost & Optuna)

Bu proje, Dataforge yarışması kapsamında özel olarak oluşturulmuş sentetik (yapay) bir tarım veri seti kullanılarak tarımsal üretim miktarını (Production) tahmin etmek amacıyla geliştirilmiş bir makine öğrenmesi modelidir. Model, uç değerlerin etkisini azaltmak ve gerçeğe daha yakın sonuçlar almak adına **MAE (Mean Absolute Error)** metriğine odaklanarak optimize edilmiştir.

### Proje Adımları ve Kullanılan Teknikler

**1. Veri Ön İşleme ve Temizlik (Data Preprocessing):**
* Hedef değişkeni (Production) eksik olan satırların veri setinden tamamen çıkarılması.
* **Zehirli Veri (Poison Data) Temizliği:** Veri setini bozan spesifik aykırı değerin (`65659968.137`) eğitim setinden temizlenmesi.
* `Crop_Value_Index` sütunundaki eksik değerlerin, eğitim setinin medyan değeri ile doldurulması.
* Hedef değişkendeki (Production) çarpıklığı gidermek için logaritmik dönüşüm (`np.log1p`) uygulanması.

**2. Özellik Mühendisliği (Feature Engineering):**
* `Year` (Yıl) değişkeninden başlangıç yılı (`Year_Start`), bitiş yılı (`Year_End`) ve aralarındaki farkın (`Year_Diff`) türetilmesi.
* `Area` (Ekim Alanı) değişkenine, aşırı büyük değerleri dengelemek için logaritmik dönüşüm (`np.log1p`) uygulanması.
* Kategorik değişkenler (`State`, `District`, `Crop`) için **Frekans Kodlaması (Frequency Encoding)** yapılarak yeni özellikler eklenmesi.
* Tüm kategorik verilerin, Train ve Test setlerinde ortak bir kategori havuzunda birleştirilerek XGBoost'un doğal kategorik desteğine (`enable_categorical=True`) uygun hale getirilmesi.

**3. Model Optimizasyonu (MAE Odaklı):**
* **Optuna** kütüphanesi kullanılarak XGBoost algoritması için 15 denemelik (trial) hiperparametre optimizasyonu gerçekleştirilmiştir.
* Önceki modellerin aksine, hedef fonksiyon olarak doğrudan **`reg:absoluteerror`** kullanılmış ve Optuna'nın hiperparametreleri **MAE** değerini en aza indirecek şekilde seçmesi sağlanmıştır.

**4. Model Değerlendirme (5-Fold Cross-Validation):**
* Aşırı öğrenmeyi (overfitting) engellemek ve model tutarlılığını ölçmek için 5-Fold Cross Validation uygulanmıştır.
* Tahmin sonuçları `np.expm1` ile orijinal ölçeğine geri döndürülmüş ve negatif çıkan üretim tahminleri sıfıra (0) eşitlenerek mantıksal hataların önüne geçilmiştir.
* Her bir fold için Eğitim (Train) ve Doğrulama (Test) **MAE (Ortalama Mutlak Hata)** metrikleri hesaplanarak raporlanmış, sonuçlar `submission_xgb_mae.csv` dosyasına kaydedilmiştir.

### ⚠️ Bilinen Durumlar ve Kısıtlamalar (Known Issues)
* **Aşırı Öğrenme (Overfitting):** Mevcut ön işleme ve Optuna hiperparametre optimizasyonu adımlarına rağmen, sentetik veri setindeki karmaşık veya ezberlemeye müsait yapılar nedeniyle modelde **overfitting (aşırı öğrenme)** gözlemlenmektedir. Eğitim (Train) ve Doğrulama (Test/Validation) MAE skorları arasında belirgin bir fark bulunmaktadır. İlerleyen aşamalarda L1/L2 (alpha/lambda) düzenlileştirme (regularization) değerlerinin artırılması, veri sayısının çoğaltılması veya daha katı ağaç derinliği limitleri uygulanarak bu durumun hafifletilmesi hedeflenmektedir.

### 🛠️ Kullanılan Teknolojiler ve Kütüphaneler
* **Dil:** Python 3.14
* **Veri İşleme:** Pandas, NumPy
* **Makine Öğrenmesi & Doğrulama:** Scikit-Learn (`KFold`, `train_test_split`, `mean_absolute_error`)
* **Algoritma:** XGBoost (`XGBRegressor`)
* **Hiperparametre Optimizasyonu:** Optuna
