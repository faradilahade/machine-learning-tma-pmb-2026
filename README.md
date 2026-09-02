# 🌊 Prediksi TMA per Periode — Musim Kering Agustus–Desember 2026

Sistem machine learning (LSTM) untuk memprediksi Tinggi Muka Air (TMA) bendungan pada **skala periode 10/15-harian sesuai format RTOW tiap bendungan**, dengan sandingan terhadap **BON A / BON B / RTOW** dan analisis **neraca air** (kebutuhan vs ketersediaan).

**Skala periode** (kolom `format` pada `fromat-rtow.csv`):
- `15 Harian` — 2 periode/bulan: tgl 1–15 dan 16–akhir → label `01-01, 01-02, 02-01, …, 12-02`
- `10 Harian` — 3 periode/bulan: tgl 1–10, 11–20, 21–akhir → label `01-01, 01-02, 01-03, 02-01, …, 12-03`
Label periode ini menjadi sumbu-x semua grafik (format grafik RTOW standar).

**Rekapitulasi status BON per periode — 3 kategori utama:**
`Di atas BON A` · `Di antara BON A dan BON B` · `Di bawah BON B` (+ `Tanpa Data BON`).

**Fitur sifat musim** — dari `sifat_musim.csv` (per bendungan per tahun air Nov–Okt: Basah/Normal/Kering, 2022-2023 s.d. 2025-2026). Diskor Kering=-1 / Normal=0 / Basah=+1 dan menjadi fitur masukan LSTM sehingga model belajar karakter tahun-tahun sebelumnya dan menyesuaikan prediksi dengan sifat tahun berjalan; tahun sebelum 2022-2023 dianggap Normal; horizon setelah tahun air terakhir memakai nilai tahun air terakhir yang diketahui.

Pusat Monitoring Bendungan · Subdit OP Bendungan dan Danau · Dit. Bina OP · Ditjen SDA

## Alur Pipeline

```
1. EKSTRAKSI    PostgreSQL SINBAD (rawdata.sinbad_bendungan_tma, _outflow,
                _inflow, _storagecurve) → parquet harian
                · single-query semua bendungan + groupby (pola extract_tma v2)
                · pre-check freshness (deteksi ingestion stall)
                · outflow total = Σ (turbin + abaku + aindustri + irigasi
                  + limpas + pemeliharaan)

2. QC           · deteksi anomali: spike harian > 3 m thd median rolling 7 hari,
                  nilai di luar rentang fisik storage curve
                · gap pendek (≤ 7 hari) → interpolasi linear
                · gap panjang / sisa kosong → AMBIL DATA SEBELUMNYA (ffill)
                · bendungan yang SELURUH nilainya ditolak QC dikeluarkan dari
                  output (ditandai kolom lolos_qc/alasan di rekap_qc.xlsx)
                · agregasi PERIODE 10/15-harian (format RTOW per bendungan):
                  TMA periode = AVERAGE harian → tma_periode.parquet

3. NERACA AIR   V_inflow(t) = V(t) + Q_outflow(t)·60·60·24 − V(t−1)
                · V dari interpolasi TMA-QC ke storage curve (m³)
                · outflow anomali (> qc.batas_outflow_m3s) → data sebelumnya
                · volume inflow per periode = Σ harian (juta m³)
                  → gabungan_periode.parquet (siap model)

4. LSTM         DUA model global, satu per format periode:
                  15 Harian: input 24 periode → output 10 (Agu–Des)
                  10 Harian: input 36 periode → output 15 (Agu–Des)
                Encoder LSTM(64) → RepeatVector → Decoder LSTM(64)
                → TimeDistributed(Dense(1))
                · fitur: tma_norm, inflow_norm, sin/cos posisi periode,
                  SKOR SIFAT MUSIM tahun air (-1/0/+1)
                · normalisasi min–max PER BENDUNGAN
                · EarlyStopping + ReduceLROnPlateau, split temporal 80/20

5. PREDIKSI     BERGULIR BULANAN: cutoff = periode lengkap terakhir pada
                data (periode ekor yang belum selesai dibuang), input 1 tahun
                periode sebelum cutoff → output 10/15 periode berikutnya
                (bulan berjalan + ±4 bulan ke depan, dipangkas s.d. Desember
                tahun berjalan) → status per periode: Di atas BON A /
                Di antara BON A-B / Di bawah BON B (+ selisih ke RTOW)

6. AKURASI      arsip prediksi tiap run (run_id = bulan cutoff, mis. 2026-08)
                → riwayat_prediksi.parquet; periode arsip yang realisasinya
                sudah masuk disandingkan otomatis: error (m) + kesesuaian
                kategori BON per bendungan per periode
                → akurasi_realisasi.parquet/.xlsx (menu 🎯 Akurasi Model)

7. EVALUASI     backtest 2 periode uji (Agu–Des 2025, Mar–Jul 2026):
                LSTM vs Persistensi vs Naif musiman vs Klimatologi
                vs Tren linier → MAE/RMSE/bias/MAPE-rentang

8. VISUALISASI  Streamlit + Plotly, sumbu-x label periode '01-01'…'12-03':
                🟢 hijau  = TMA realisasi (per periode)
                🔴 merah  = TMA prediksi Agu–Des (putus-putus)
                🟡 amber  = BON A   ·   🟠 coral = BON B
                🔵 biru titik-titik = RTOW  ·  abu = historis per tahun
```

## Dashboard (`streamlit run app.py`)

**Filter (panel kiri, berlaku di semua halaman)**

- **Pulau / wilayah** dan **Balai (BBWS/BWS)** — dari kolom `nama_pulau` dan
  `nama_balai` pada `data_bon/daftar_bendungan.csv`, sehingga langsung terlihat
  balai mana yang bendungannya di bawah BON B.
- **Status terburuk prediksi** — Kritis (di bawah BON B) / Waspada (di bawah
  BON A) / Normal / Tanpa Data BON.

**Menu**

| Menu | Isi |
|---|---|
| 🏜️ Siaga Kekeringan | Dashboard pengambilan keputusan: KPI 3 kategori BON, donat distribusi status, batang bertumpuk status per bulan (periode terburuk) & per balai, defisit vs surplus air per bulan, kalender panas bendungan kritis, 15 defisit air terbesar, **daftar prioritas berskor**, **akurasi & perbandingan 5 metode**, **kesimpulan otomatis seluruh data**, dan **unduh laporan PDF** |
| 📈 Detail Bendungan | Grafik sandingan interaktif per periode (realisasi, prediksi, BON A/B, RTOW, historis per tahun), metrik TMA, **Hasil Analisis** (narasi status BON + sifat musim + akurasi backtest bendungan tsb), **neraca air kebutuhan vs ketersediaan (surplus/defisit)**, unduh JPG + Excel |
| 🚨 Pemantauan Bulanan | Bulan pemantauan (default bulan prediksi pertama = bulan berjalan): nilai dari **periode terburuk** dalam bulan, selisih ke BON A/B & RTOW, rekap per balai, grafik peringkat, unduh Excel |
| 🎯 Akurasi Model | Prediksi run terdahulu vs realisasi per bendungan: KPI (MAE, % status BON sesuai), tren akurasi antar-run, tabel per bendungan (Sesuai / Sebagian / Belum sesuai), 20 MAE terbesar, detail prediksi tiap run vs realisasi per bendungan, unduh Excel — riwayat terakumulasi tiap run untuk evaluasi kekurangan model |
| 🗂️ Rekap & Di bawah BON B | Rekapitulasi **3 kategori**: Di atas BON A / Di antara BON A–B / Di bawah BON B — tab per kategori + rincian periode yang menembus BON B + rekap per balai |
| 🧩 Data Belum Cocok | Bendungan yang datanya belum terhubung antar sumber, per kategori masalah + tindak lanjut, unduh Excel |
| ⬇️ Unduh Laporan | Excel lengkap, ZIP grafik JPG, dan **laporan PDF** — untuk **semua bendungan** maupun **hanya yang di bawah BON B** |

**Laporan PDF** (`laporan_siaga_kekeringan_*.pdf`, A4 lanskap) — kesimpulan
otomatis, rekap status per bulan & per balai, tabel bendungan di bawah BON B,
dan grafik sandingan bendungan paling rawan. Tersedia di menu Siaga Kekeringan
dan Unduh Laporan; mengikuti filter aktif.

**Skor prioritas (menu Siaga Kekeringan)** = 3×bulan di bawah BON B + 1×bulan
di bawah BON A + 2×bulan defisit air (Agu–Des) + 1×bulan di bawah RTOW —
semakin tinggi semakin butuh perhatian.

**Neraca air** — dari sheet `ketersediaan_air` dan `kebutuhan_air` (m³/bulan):
defisit bila kebutuhan > ketersediaan pada bulan tersebut, selain itu surplus;
ketersediaan bernilai negatif (salah entri) dipotong ke 0.

**Akurasi & metode pembanding** (`src/evaluasi.py`, hasil di
`output/prediksi/evaluasi_metode.*`) — backtest dua periode uji (Agu–Des 2025 dan
Mar–Jul 2026) pada skala periode: model dimundurkan ke cutoff, memprediksi
horizon Agu–Des hanya dari data sebelum cutoff, dibandingkan dengan realisasi.
Metode: **LSTM produksi** (dengan fitur sifat musim), **Persistensi**,
**Naif musiman** (periode sama tahun lalu), **Klimatologi** (rerata per label
periode), **Tren linier** (ekstrapolasi 12 periode terakhir). Metrik: MAE,
RMSE, bias, MAPE terhadap rentang TMA. Hasil run terakhir: **LSTM terbaik di
kedua periode uji** — MAE 1,07 m (vs 1,69 m Persistensi) di Agu–Des 2025 dan
0,88 m (vs 1,12 m Naif musiman) di Mar–Jul 2026, dengan bias hampir nol.
Catatan: periode uji masih dalam rentang data pelatihan model produksi, jadi
skor LSTM cenderung optimis.

**Kategori "Data Belum Cocok"**

1. **1a. Tidak ada di database** — kode/nama ada di daftar BON tetapi tidak
   ditemukan pada data TMA hasil ekstraksi SINBAD (umumnya pemetaan
   `BDxxx → id` tidak cocok → tambahkan kolom `id_db`, atau bendungan belum
   mengirim data).
   **1b. Data ditolak QC** — data TMA ada, tetapi seluruh nilainya di luar
   rentang elevasi storage curve bendungan tersebut (indikasi id/satuan tidak
   cocok). Sumber: kolom `lolos_qc` pada `rekap_qc.xlsx`.
2. **Tidak ada hasil prediksi** — histori ada tetapi < 12 bulan (panjang sekuens input).
3. **Tidak ada di daftar BON** — kode muncul di data TMA/prediksi tetapi belum
   terdaftar di `daftar_bendungan.csv`.
4. **Tanpa nilai BON A/B** — tetap diprediksi, status tidak dapat dinilai.
5. **BON terisi otomatis** — sebagian bulan BON kosong lalu diinterpolasi linear.

## Ekspor Laporan

Semua ekspor dibuat di memori (tanpa menulis ke disk) oleh `src/laporan.py`.

- **Excel** (`laporan_prediksi_tma_2026_*.xlsx`) — lembar: `Penjelasan`
  (metodologi, arti status BON/RTOW/neraca, definisi kolom, batasan),
  `Ringkasan Bendungan` (termasuk narasi otomatis per bendungan),
  `Pemantauan <Bulan>` (dengan selisih ke RTOW), `Prediksi Bulanan`
  (realisasi + prediksi + BON + RTOW), `Di bawah BON B`,
  `Di bawah BON B (Agu-Des)`, `Rekap per Balai`, `Rekap Neraca Air`
  (dengan narasi analisis), `Neraca Air Bulanan`, `Data Belum Cocok`.
  Tersedia dua cakupan: **semua bendungan** dan **hanya di bawah BON B**.
- **Grafik JPG** — per bendungan (grafik sandingan realisasi/prediksi + BON A/B
  + RTOW, bulan kritis ditandai lingkaran merah) dan grafik peringkat kerawanan
  bulanan.
- **ZIP grafik** — satu JPG per bendungan, dikelompokkan folder `KRITIS/`,
  `WASPADA/`, `NORMAL/`, `TANPA-BON/`. Merender ±210 grafik memerlukan ±45 detik.

## Struktur Folder

```
prediksi_tma_bulanan/
├── config.yaml                  # kredensial DB (LOKAL, di-.gitignore)
├── config.cloud.yaml            # config tanpa kredensial (dipakai di cloud)
├── run_pipeline.py              # orkestrasi end-to-end
├── app.py                       # dashboard Streamlit
├── requirements.txt             # dependensi dashboard (dibaca Streamlit Cloud)
├── requirements-pipeline.txt    # + dependensi pipeline (TensorFlow, psycopg2)
├── data_bon/                    # ← folder referensi (sumber daftar bendungan)
│   ├── data_bon_a_bon_b_rtow_ketersediaan_kebutuhan_air.xlsx
│   │                            #   sheet: bon a, bon b, rtow (mdpl, Jan–Des),
│   │                            #   ketersediaan_air, kebutuhan_air (m³/bulan);
│   │                            #   kunci = kolom id (id database SINBAD)
│   ├── fromat-rtow.csv          #   id;nama;format → 10 Harian / 15 Harian
│   ├── sifat_musim.csv          #   id;nama;2022-2023;…;2025-2026 (Basah/Normal/Kering)
│   └── daftar_bendungan.csv     #   id;kode;nama;nama_balai;nama_pulau
├── src/
│   ├── utils.py                 # loader config, BON/RTOW/neraca, daftar bendungan
│   ├── laporan.py               # tabel rekap/pemantauan/neraca + ekspor JPG & Excel
│   ├── evaluasi.py              # backtest akurasi LSTM vs 4 metode pembanding
│   ├── buat_template_bon.py     # generator template BON (sekali pakai)
│   ├── buat_daftar_dari_bon.py  # regenerasi daftar dari sheet BON_A
│   ├── extract_data.py          # SQL SINBAD (mode database)
│   ├── generate_dummy_data.py   # data sintetis (mode dummy / demo)
│   ├── qc_interpolasi.py        # QC + interpolasi + rerata bulanan
│   ├── hitung_inflow.py         # neraca air → volume inflow bulanan
│   ├── train_lstm.py            # training model
│   └── predict_2026.py          # prediksi + grafik sandingan statis
├── data/                        # raw & processed (parquet)
└── output/                      # models, prediksi (xlsx/parquet), grafik (png)
```

## Cara Pakai

```bash
# Dashboard saja (data hasil pipeline sudah ada di repo):
pip install -r requirements.txt
streamlit run app.py

# Pipeline lengkap (ekstraksi DB + training LSTM) — hanya lokal:
pip install -r requirements-pipeline.txt

# 1. Pastikan file referensi terisi:
#    data_bon/data_bon_a_bon_b_rtow_ketersediaan_kebutuhan_air.xlsx
#    (sheet: bon a, bon b, rtow, ketersediaan_air, kebutuhan_air; kunci = id DB)
#    dan data_bon/daftar_bendungan.csv (id;kode;nama;balai;pulau)

# 2. Di jaringan internal PU, ubah config.yaml:  mode: "database"
#    (di luar jaringan, mode: "dummy" untuk demo)

# 3. Jalankan pipeline
python run_pipeline.py

# 4. Buka dashboard
streamlit run app.py
```

### Rutinitas bulanan (prediksi bergulir otomatis)

Cukup jalankan `python run_pipeline.py` **setiap awal bulan** (setelah data
bulan sebelumnya masuk ke SINBAD) — semua bergulir otomatis:

1. Cutoff terdeteksi sendiri dari data (periode lengkap terakhir), prediksi
   dimulai dari **bulan berjalan** + bulan-bulan berikutnya.
2. Prediksi run sebelumnya **diarsipkan** lalu **dibandingkan dengan
   realisasi yang baru masuk** — error dan kesesuaian status BON per
   bendungan per periode terakumulasi di menu 🎯 Akurasi Model.
3. Commit & push → Streamlit Cloud redeploy otomatis.

Agar benar-benar tanpa sentuhan, jadwalkan lewat Windows Task Scheduler
(mis. tanggal 2 tiap bulan, di jaringan internal PU):
`python f:\...\run_pipeline.py` diikuti `git add -A && git commit && git push`.

Catatan horizon lintas tahun: periode prediksi yang jatuh setelah Desember
tahun berjalan DIPANGKAS (dashboard dan label periode 'MM-PP' berbasis satu
tahun kalender). Run awal Januari otomatis memprediksi Jan–Mei tahun baru.

## Deploy ke Streamlit Community Cloud

Repo ini **siap deploy** ke [share.streamlit.io](https://share.streamlit.io) —
dashboard hanya membaca hasil pipeline (parquet/xlsx) yang ikut di-commit,
tanpa koneksi database dan tanpa TensorFlow.

- `requirements.txt` — dependensi ringan khusus dashboard (yang dibaca
  Streamlit Cloud). Dependensi pipeline (TensorFlow, psycopg2) dipisah ke
  `requirements-pipeline.txt` agar deploy cepat dan hemat memori.
- `config.yaml` (kredensial DB) **di-.gitignore**; di cloud,
  `src/utils.load_config()` otomatis memakai `config.cloud.yaml`
  (salinan tanpa kredensial).
- Main file: `app.py` · Python 3.12.

**Langkah**: push ke GitHub → login share.streamlit.io dengan akun GitHub →
*Create app* → pilih repo & branch `main`, main file `app.py` → *Deploy*.

**Memperbarui data di cloud**: jalankan `python run_pipeline.py` di jaringan
internal, lalu commit & push hasil `output/` dan `data/processed/` —
Streamlit Cloud melakukan redeploy otomatis pada setiap push.

## Catatan Penting

- **Daftar bendungan yang dimodelkan = HANYA yang tercantum di daftar/file BON** — sesuai keputusan fokus pada bendungan yang punya data BON.
- Skema kolom sudah **terverifikasi dari DBeaver (2026-07)**: identitas bendungan = kolom `id` (int4) di semua tabel rawdata; kolom waktu = `"timestamp"`; storage curve memakai kolom `tma` (elevasi), `volume`, dan `periode` (dipilih versi dengan `sync_date` terbaru).
- `daftar_bendungan.csv` memakai delimiter `;` (diatur di `config → bon.csv_sep`)
  dengan kolom `id;kode_bendungan;nama_bendungan;nama_balai;nama_pulau`.
  Kolom `nama_balai` dan `nama_pulau` menjadi sumber filter wilayah dan
  rekap per balai di dashboard — bila kosong, ditampilkan `(TIDAK DIKETAHUI)`.
- **Pemetaan id memakai kolom `id` pada `daftar_bendungan.csv`** (id asli di
  database SINBAD) — dipakai langsung untuk penarikan TMA/inflow/outflow/storage
  curve dan sebagai kunci relasi semua sheet file referensi BON/RTOW/neraca.
  Fallback strip_prefix hanya bila kolom id tidak ada.
- Kondisi data saat ini (run 31 Juli 2026): 212 bendungan terdaftar, seluruhnya
  punya nilai BON A/B & RTOW; **210 masuk pemodelan** — 2 dikeluarkan karena
  seluruh nilai TMA-nya ditolak QC (BD046 WR JENGGOT/GROJOKAN, BD200 RAKAWATU;
  lihat menu 🧩 Data Belum Cocok).
- **Satuan kolom `volume` pada `rawdata.sinbad_bendungan_storagecurve` adalah m³,
  bukan juta m³.** Diverifikasi empiris: neraca air `dV + Qout·86400` dibanding
  `inflow_tercatat·86400` memberi rasio median **1,01** pada 135 bendungan bila
  kurva diperlakukan sebagai m³ (dan ~10⁶ bila dianggap juta m³). Konversi ke
  juta m³ dilakukan sekali saja, pada agregasi bulanan.
- `tanggal` pada parquet `data/raw/` berasal dari `"timestamp"::date` sehingga
  psycopg2 mengembalikan `datetime.date` (dtype `object`). Semua tahap melewati
  `utils.normalisasi_tanggal()` agar dtype seragam `datetime64` — tanpa itu merge
  antar-tahap gagal dengan *"trying to merge on datetime64[ns] and object columns"*.
- Outflow SINBAD memuat salah entri ekstrem (mis. 3,2 juta m³/detik). Nilai di
  atas `qc.batas_outflow_m3s` (default 10.000 m³/detik) dianggap anomali dan
  diisi data sebelumnya — 6 baris pada data saat ini. Tanpa ini, satu hari bogus
  menambah miliaran m³ ke inflow bulanan dan merusak normalisasi min–max fitur.
- Sel BON yang kosong sebagian diisi interpolasi antar-bulan (ditandai kolom `bon_terisi_otomatis`); bendungan tanpa nilai BON sama sekali tetap diprediksi dengan status `Tanpa Data BON`. Pada file referensi saat ini seluruh 212 bendungan sudah lengkap.
- Nilai BON A/B/RTOW dan neraca air bersumber **bulanan** — pada skala periode, nilai bulan yang sama berlaku untuk seluruh periode dalam bulan tersebut (tampak sebagai garis berundak pada grafik).
- Bendungan berformat `Tidak Melayani` atau tidak tercantum di `fromat-rtow.csv` memakai default **15 Harian**.
- Agregasi lintas-bendungan (heatmap, batang per bulan, skor prioritas) menilai bulan dari **periode terburuknya** agar bendungan 10-harian dan 15-harian sebanding.
- Pre-check freshness akan **menghentikan pipeline** bila MAX(waktu) TMA lebih tua dari 30 hari sebelum akhir periode (indikasi ingestion stall, pelajaran dari kasus extract_tma).
- Kekurangan data: bendungan dengan histori < 12 bulan dilewati saat prediksi (butuh 1 sekuens input penuh).
- `output/prediksi/prediksi_tma_2026.xlsx` memuat realisasi + prediksi + BON A/B + kolom `status_bon` per bulan — siap dipakai untuk rekap Nota Dinas.
- Keamanan: `config.yaml` berisi kredensial database — jangan commit ke repository publik; tambahkan ke `.gitignore` bila diversion-control.
