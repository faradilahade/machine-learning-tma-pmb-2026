"""Arsip prediksi antar-run + akurasi prediksi terdahulu vs realisasi.

Alur bulanan otomatis (bagian dari run_pipeline, setelah predict_2026):

1. ARSIP — setiap run pipeline diarsipkan ke riwayat_prediksi.parquet dengan
   run_id = BULAN CUTOFF data realisasi ('YYYY-MM', bulan lengkap terakhir
   yang menjadi input model). predict_2026 mengarsipkan file prediksi LAMA
   sebelum menimpanya, lalu modul ini mengarsipkan run BARU — run ulang pada
   cutoff yang sama menggantikan arsip run tersebut (dedup per run_id).

2. EVALUASI — semua baris arsip yang target periodenya kini SUDAH punya
   realisasi (akhir periode <= tanggal data harian terakhir) disandingkan:
     error_m         = tma_prediksi - tma_realisasi
     status_sesuai   = kategori BON prediksi == kategori BON realisasi
   Hasil per periode per bendungan per run:
     output/prediksi/akurasi_realisasi.parquet  (dibaca menu 🎯 Akurasi Model)
     output/prediksi/akurasi_realisasi.xlsx     (Detail + rekap per bendungan
                                                 per run + rekap per run)

Dengan demikian tiap awal bulan cukup menjalankan `python run_pipeline.py`:
prediksi bergulir ke bulan berjalan, dan akurasi run-run sebelumnya otomatis
terakumulasi untuk dievaluasi di dashboard.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (load_config, path_root, baca_daftar_bendungan, baca_bon,
                       akhir_periode)

ATAS_A = "Di atas BON A"
ANTARA = "Di antara BON A dan BON B"
BAWAH_B = "Di bawah BON B"
TANPA_BON = "Tanpa Data BON"

FILE_RIWAYAT = os.path.join("output", "prediksi", "riwayat_prediksi.parquet")
FILE_AKURASI = os.path.join("output", "prediksi", "akurasi_realisasi.parquet")
FILE_AKURASI_XLSX = os.path.join("output", "prediksi", "akurasi_realisasi.xlsx")

# klasifikasi kesesuaian per bendungan per run (dari % periode dengan
# kategori BON prediksi == realisasi)
SESUAI, SEBAGIAN, BELUM = "Sesuai", "Sebagian sesuai", "Belum sesuai"


def _run_id_dari(pred_df: pd.DataFrame) -> str:
    """run_id = bulan realisasi terakhir yang dipakai run tsb ('YYYY-MM')."""
    real = pred_df[pred_df["jenis"] == "realisasi"]
    if not real.empty:
        t = real["tanggal"].max()
    else:  # fallback: sebulan sebelum periode prediksi pertama
        t = pred_df["tanggal"].min() - pd.offsets.MonthBegin(1)
    return f"{t.year:04d}-{t.month:02d}"


def arsipkan(pred_df: pd.DataFrame) -> str:
    """Simpan baris PREDIKSI sebuah run ke riwayat (dedup per run_id)."""
    rid = _run_id_dari(pred_df)
    pred = pred_df[pred_df["jenis"] == "prediksi"][
        ["kode_bendungan", "nama_bendungan", "format_periode",
         "tanggal", "periode", "tma"]].rename(columns={"tma": "tma_prediksi"})
    if pred.empty:
        print(f"LEWATI arsip run {rid}: tidak ada baris prediksi")
        return rid
    pred.insert(0, "run_id", rid)
    fp = path_root(FILE_RIWAYAT)
    if os.path.exists(fp):
        lama = pd.read_parquet(fp)
        lama = lama[lama["run_id"] != rid]
        pred = pd.concat([lama, pred], ignore_index=True)
    pred = pred.sort_values(["run_id", "kode_bendungan", "tanggal"])
    pred.to_parquet(fp, index=False)
    print(f"OK  arsip run {rid} -> riwayat_prediksi.parquet "
          f"({pred['run_id'].nunique()} run, {len(pred):,} baris)")
    return rid


def arsipkan_file(fp: str) -> str | None:
    """Arsipkan file prediksi lama (dipanggil predict_2026 sebelum menimpa)."""
    if not os.path.exists(fp):
        return None
    return arsipkan(pd.read_parquet(fp))


def _status_bon(tma: pd.Series, bon_a: pd.Series, bon_b: pd.Series) -> np.ndarray:
    return np.select(
        [bon_a.isna() & bon_b.isna(), tma < bon_b, tma < bon_a],
        [TANPA_BON, BAWAH_B, ANTARA], ATAS_A)


def rekap_per_bendungan(detail: pd.DataFrame) -> pd.DataFrame:
    """Rekap akurasi per bendungan per run: MAE, bias, % status BON sesuai."""
    if detail.empty:
        return pd.DataFrame()
    rk = (detail.groupby(["run_id", "kode_bendungan", "nama_bendungan",
                          "format_periode"])
          .agg(n_periode=("periode", "size"),
               mae_m=("abs_error_m", "mean"),
               bias_m=("error_m", "mean"),
               n_status_sesuai=("status_sesuai", "sum"))
          .reset_index())
    rk[["mae_m", "bias_m"]] = rk[["mae_m", "bias_m"]].round(3)
    rk["n_status_sesuai"] = rk["n_status_sesuai"].astype(int)
    rk["pct_status_sesuai"] = (rk["n_status_sesuai"] / rk["n_periode"]
                               * 100).round(1)
    rk["kesesuaian"] = np.select(
        [rk["pct_status_sesuai"] >= 100, rk["pct_status_sesuai"] >= 50],
        [SESUAI, SEBAGIAN], BELUM)
    return rk.sort_values(["run_id", "pct_status_sesuai", "mae_m"],
                          ascending=[True, True, False]).reset_index(drop=True)


def rekap_per_run(detail: pd.DataFrame) -> pd.DataFrame:
    """Rekap per run: MAE keseluruhan & % status sesuai (tren antar-run)."""
    if detail.empty:
        return pd.DataFrame()
    rr = (detail.groupby("run_id")
          .agg(n_bendungan=("kode_bendungan", "nunique"),
               n_periode=("periode", "size"),
               mae_m=("abs_error_m", "mean"),
               bias_m=("error_m", "mean"),
               pct_status_sesuai=("status_sesuai",
                                  lambda s: round(s.mean() * 100, 1)))
          .reset_index())
    rr[["mae_m", "bias_m"]] = rr[["mae_m", "bias_m"]].round(3)
    return rr


def evaluasi_riwayat() -> pd.DataFrame:
    """Sandingkan seluruh arsip prediksi dengan realisasi yang sudah ada."""
    cfg = load_config()
    fp_r = path_root(FILE_RIWAYAT)
    if not os.path.exists(fp_r):
        print("LEWATI evaluasi: riwayat_prediksi.parquet belum ada")
        return pd.DataFrame()
    riw = pd.read_parquet(fp_r)

    real = pd.read_parquet(path_root("data", "processed",
                                     "tma_periode.parquet"))
    real = real[["kode_bendungan", "tanggal", "tma"]].rename(
        columns={"tma": "tma_realisasi"})
    det = riw.merge(real, on=["kode_bendungan", "tanggal"], how="inner")
    det = det.dropna(subset=["tma_realisasi"])

    # hanya periode yang sudah LENGKAP (akhirnya <= tanggal data terakhir)
    harian = pd.read_parquet(path_root("data", "processed",
                                       "tma_harian_qc.parquet"))
    tgl_maks = harian["tanggal"].max()
    if not det.empty:
        akhir = [akhir_periode(t.year, t.month, int(p[3:]), f)
                 for t, p, f in zip(det["tanggal"], det["periode"],
                                    det["format_periode"])]
        det = det[pd.Series(akhir, index=det.index) <= tgl_maks]
    if det.empty:
        print("Evaluasi: belum ada periode prediksi yang sudah terealisasi "
              "(wajar pada run pertama — akan terisi run bulan berikutnya)")
        return pd.DataFrame()

    daftar = baca_daftar_bendungan(cfg)
    bon = baca_bon(cfg, daftar)
    det["bulan"] = det["tanggal"].dt.month
    det = det.merge(bon.drop(columns="nama_bendungan"),
                    on=["kode_bendungan", "bulan"], how="left")
    det["error_m"] = (det["tma_prediksi"] - det["tma_realisasi"]).round(3)
    det["abs_error_m"] = det["error_m"].abs().round(3)
    det["status_prediksi"] = _status_bon(det["tma_prediksi"],
                                         det["bon_a"], det["bon_b"])
    det["status_realisasi"] = _status_bon(det["tma_realisasi"],
                                          det["bon_a"], det["bon_b"])
    det["status_sesuai"] = det["status_prediksi"] == det["status_realisasi"]
    det = det.sort_values(["run_id", "kode_bendungan", "tanggal"]
                          ).reset_index(drop=True)

    det.to_parquet(path_root(FILE_AKURASI), index=False)
    rk, rr = rekap_per_bendungan(det), rekap_per_run(det)
    with pd.ExcelWriter(path_root(FILE_AKURASI_XLSX)) as xl:
        det.to_excel(xl, sheet_name="Detail per Periode", index=False)
        rk.to_excel(xl, sheet_name="Rekap per Bendungan", index=False)
        rr.to_excel(xl, sheet_name="Rekap per Run", index=False)
    print(f"OK  akurasi_realisasi.parquet ({len(det):,} baris periode "
          f"terevaluasi, {det['kode_bendungan'].nunique()} bendungan, "
          f"{det['run_id'].nunique()} run)")
    for _, b in rr.iterrows():
        print(f"    run {b['run_id']}: MAE {b['mae_m']:.2f} m · "
              f"status BON sesuai {b['pct_status_sesuai']:.1f}% "
              f"({b['n_periode']:,} periode, {b['n_bendungan']} bendungan)")
    return det


def main():
    cfg = load_config()
    fp_pred = path_root(cfg["output"]["dir_prediksi"],
                        "prediksi_tma_2026.parquet")
    if os.path.exists(fp_pred):
        arsipkan_file(fp_pred)          # arsipkan run yang baru saja dibuat
    evaluasi_riwayat()


if __name__ == "__main__":
    main()
