"""Orkestrasi pipeline end-to-end.

Urutan:
  0. Template BON (hanya jika belum ada)
  1. Ekstraksi data  : mode "database" -> PostgreSQL SINBAD
                       mode "dummy"    -> data sintetis (demo tanpa DB)
  2. QC + interpolasi + agregasi bulanan (average)
  3. Neraca air -> volume inflow bulanan
  4. Training LSTM
  5. Prediksi Agustus-Desember 2026 + grafik sandingan
  6. Evaluasi akurasi: backtest LSTM vs metode pembanding

Jalankan:  python run_pipeline.py
Lalu    :  streamlit run app.py
"""
import os
import sys
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from src.utils import load_config, path_root


def jalankan(modul):
    print(f"\n{'=' * 60}\n>> {modul}\n{'=' * 60}")
    r = subprocess.run([sys.executable, os.path.join(ROOT, modul)], cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"GAGAL di tahap: {modul}")


def main():
    cfg = load_config()

    if not os.path.exists(path_root(cfg["bon"]["file_bon"])):
        jalankan("src/buat_template_bon.py")

    if cfg["mode"] == "database":
        jalankan("src/extract_data.py")
    else:
        print("\n[MODE DUMMY] Menggunakan data sintetis. "
              "Ubah mode: 'database' di config.yaml saat di jaringan internal PU.")
        jalankan("src/generate_dummy_data.py")

    jalankan("src/qc_interpolasi.py")
    jalankan("src/hitung_inflow.py")
    jalankan("src/train_lstm.py")
    jalankan("src/predict_2026.py")
    jalankan("src/evaluasi.py")

    print("\n" + "=" * 60)
    print("PIPELINE SELESAI (OK)")  # tanpa simbol unicode: konsol Windows cp1252
    print("Grafik statis : output/grafik/")
    print("Hasil prediksi: output/prediksi/prediksi_tma_2026.xlsx")
    print("Dashboard     : streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
