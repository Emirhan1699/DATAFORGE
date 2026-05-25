# DATAFORGE
XGBoost ve Optuna kullanılarak sentetik tarım verisi üzerinde geliştirilmiş üretim tahmini modeli.

# Tarımsal Üretim Tahmin Modeli (Crop Production Prediction)

Bu proje, Dataforge yarışması kapsamında özel olarak oluşturulmuş **sentetik (yapay) bir tarım veri seti** kullanılarak tarımsal üretim miktarını (Production) tahmin etmek amacıyla geliştirilmiş bir makine öğrenmesi modelidir. Sentetik veri setleri üzerinde çalışmak, modelin ezberleme (overfitting) yapmasını zorlaştırdığı için veri ön işleme, özellik mühendisliği (feature engineering) ve optimizasyon becerilerini ön plana çıkarmayı gerektirir.

## Proje Adımları ve Kullanılan Teknikler

* **Veri Ön İşleme (Data Preprocessing):**
  * Eksik hedef değişkenlerin veri setinden temizlenmesi.
  * `Crop_Value_Index` sütunundaki eksik değerlerin medyan ile doldurulması.
  * Hedef değişkendeki (Production) çarpıklığı gidermek ve model performansını artırmak için logaritmik dönüşüm (`np.log1p`) uygulanması.

* **Özellik Mühendisliği (Feature Engineering):**
  * `Year` (Yıl) değişkeninden başlangıç yılı, bitiş yılı ve aradaki farkın (Year_Diff) türetilmesi.
  * `Area` (Ekim Alanı) değişkenine logaritmik dönüşüm uygulanması.
  * Kategorik değişkenler (`State`, `District`, `Crop`) için Frekans Kodlaması (Frequency Encoding) yapılması.
  * Kategorik verilerin XGBoost'un doğal kategorik desteğine (`enable_categorical=True`) uygun hale getirilmesi.

* **Model Optimizasyonu:**
  * **Optuna** kütüphanesi kullanılarak XGBoost algoritması için (n_estimators, learning_rate, max_depth, subsample vb.) hiperparametre optimizasyonu yapılmıştır.

* **Model Değerlendirme (Cross-Validation):**
  * Sentetik verideki karmaşık örüntüleri yakalarken aşırı öğrenmeyi (overfitting) engellemek ve modelin tutarlılığını ölçmek için **5-Fold Cross Validation** uygulanmıştır. 
  * Her bir fold için Train ve Validation MSE/RMSE metrikleri hesaplanarak tutarlılık raporlanmıştır.

## 🛠️ Kullanılan Teknolojiler ve Kütüphaneler
* Python 3.14.2
* Pandas & NumPy
* Scikit-Learn (KFold, mean_squared_error)
* XGBoost
* Optuna
