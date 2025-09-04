# Ads Analytics

Sebuah purwarupa untuk menganalisis demografi audiens di depan layar (iklan, etalase, dll.) secara real-time menggunakan Computer Vision.

## ✅ Fitur Utama

- **Deteksi Kehadiran (Impressions)**: Menghitung jumlah wajah yang terdeteksi menggunakan Haar Cascades.
- **Deteksi Atensi (Views)**: Mengukur atensi audiens. Sebuah "view" dihitung jika wajah terdeteksi secara terus-menerus melebihi ambang batas (`DWELL_THRESHOLD`).
- **Analisis Demografi**:
  - **Estimasi Usia**: Mengelompokkan usia ke dalam beberapa kategori (`13-17`, `18-24`, `25-34`, dll.). Memerlukan model ONNX.
  - **Ekspresi Sederhana**: Mendeteksi ekspresi `smile` atau `neutral`.
- **Metrik Agregat**:
  - **Dwell Time**: Waktu rata-rata audiens melihat ke arah layar.
  - **View Rate**: Rasio antara jumlah views dan impressions.
- **Output Fleksibel**:
  - **Laporan Konsol**: Menampilkan statistik agregat secara periodik.
  - **Ekspor CSV**: Menyimpan data statistik harian di direktori `exports/`.
  - **Dasbor Web**: (Opsional) Menampilkan laporan real-time melalui antarmuka web.
- **Input Beragam**: Dapat menggunakan input dari kamera, file video, atau data sintetis untuk development tanpa kamera.
- **Mode Headless**: Berjalan di server atau environment tanpa GUI.

---

## 📦 Instalasi & Konfigurasi

### 1. Dependensi Utama
Pastikan Python 3.9+ terpasang.

```bash
# Instal dependensi utama (OpenCV dan NumPy)
pip install opencv-python numpy
```

### 2. Dependensi Opsional

- **Untuk Dasbor Web**:
  ```bash
  pip install flask
  ```
- **Untuk Estimasi Usia**:
  Membutuhkan model klasifikasi usia dalam format ONNX.
  ```bash
  # Instal ONNX Runtime
  pip install onnxruntime

  # Unduh atau siapkan file model .onnx Anda.
  # Contoh: model_usia.onnx
  ```

---

## 🚀 Cara Menjalankan

### Mode Operasi

- **Menggunakan Kamera**:
  ```bash
  # Menggunakan kamera default (indeks 0)
  python3 app.py --camera-index 0
  ```
- **Menggunakan File Video**:
  ```bash
  python3 app.py --video sample.mp4
  ```
- **Mode Sintetis (Tanpa Kamera/Video)**:
  Mode ini mensimulasikan deteksi wajah, berguna untuk development.
  ```bash
  python3 app.py --synthetic
  ```

### Opsi Tambahan

- **Mengaktifkan Estimasi Usia**:
  Gunakan argumen `--age-model` dan berikan path ke file `.onnx`.
  ```bash
  python3 app.py --camera-index 0 --age-model /path/to/your/age_model.onnx
  ```
- **Mengaktifkan Dasbor Web**:
  Tambahkan flag `--web`. Dasbor akan tersedia di `http://localhost:8000`.
  ```bash
  python3 app.py --camera-index 0 --web
  ```
- **Mode Headless (Tanpa Jendela Video)**:
  Tambahkan flag `--no-window` untuk berjalan di background.
  ```bash
  python3 app.py --synthetic --no-window
  ```

### Utilitas

- **Melihat Daftar Kamera Tersedia**:
  ```bash
  python3 app.py --list-cams
  ```
- **Menjalankan Unit Tests**:
  Untuk memverifikasi fungsionalitas inti seperti IoU dan metrik.
  ```bash
  python3 app.py --run-tests
  ```

---

## 🚀 Skalabilitas & Deployment: Penggunaan di Banyak Lokasi

Untuk menggunakan aplikasi ini di beberapa lokasi fisik (misal: cabang toko, titik iklan berbeda), arsitektur yang direkomendasikan adalah model **Klien-Server Terpusat**.

**Alur Konseptual:**

```
[Lokasi A] -- app.py mengirim data -->
[Lokasi B] -- app.py mengirim data --> [Server API Pusat] -> [Database] -> [Dasbor Analitik Terpusat]
[Lokasi C] -- app.py mengirim data -->
```

### 1. Pengaturan Klien (di setiap lokasi)
- **Perangkat Keras**: Sebuah komputer mini (spt. Intel NUC, Raspberry Pi) dengan kamera terhubung.
- **Perangkat Lunak**: Jalankan `app.py` di komputer tersebut dalam mode headless yang hanya fokus pada pengumpulan dan pengiriman data. Server web lokal tidak perlu diaktifkan.
  ```bash
  # Contoh perintah untuk dijalankan di PC klien
  python3 app.py --camera-index 0 --no-window
  ```
- **Modifikasi**: Ubah `app.py` untuk mengirim data snapshot (JSON) ke API pusat melalui HTTP POST. Setiap klien harus memiliki `location_id` yang unik.

### 2. Pengaturan Server Pusat
- **Backend API**: Buat sebuah API (misal: menggunakan Flask/FastAPI) untuk menerima data dari semua klien.
- **Database**: API akan menyimpan data yang masuk ke database terpusat (misal: PostgreSQL, MongoDB), lengkap dengan `location_id` dan `timestamp`.
- **Dasbor**: Buat aplikasi web yang membaca dari database pusat untuk menampilkan analitik gabungan dari semua lokasi.

Arsitektur ini memungkinkan pemantauan dan analisis data dari semua titik secara terpusat dan real-time.

---

## ⚙️ Konfigurasi Lanjutan

Anda dapat mengubah parameter utama melalui argumen CLI saat menjalankan `app.py`:

- `--window-sec`: Interval agregasi data dalam detik (default: 60).
- `--dwell`: Durasi minimum (detik) wajah harus terdeteksi untuk dihitung sebagai "view" (default: 2.0).
- `--web-port`: Mengubah port untuk dasbor web (default: 8000).

Untuk keluar dari jendela video, tekan tombol `q`.