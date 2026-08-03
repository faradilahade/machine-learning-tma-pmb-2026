"""QC data TMA harian + agregasi bulanan (average).

Langkah QC per bendungan:
  1. Reindex ke kalender harian penuh (gap eksplisit jadi NaN).
  2. Deteksi anomali:
     - lonjakan harian > batas_perubahan_harian_m (spike fisik tak wajar)
     - nilai di luar rentang storage curve +/- margin (di luar rentang fisik)
     -> nilai anomali di-NaN-kan.
  3. Interpolasi linear untuk gap pendek (<= max_gap_interpolasi hari).
  4. Sisa gap (kosong/anomali panjang) -> forward fill: AMBIL DATA SEBELUMNYA.
  5. Agregasi bulanan: TMA bulanan = AVERAGE TMA harian hasil QC.

Output:
  data/processed/tma_harian_qc.parquet
  data/processed/tma_bulanan.parquet
  data/processed/rekap_qc.xlsx  (rekap jumlah data diperbaiki per bendungan)
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (load_config, path_root, normalisasi_tanggal,
                       baca_daftar_bendungan, baca_format_rtow,
                       agregasi_periode)


def qc_satu_bendungan(df: pd.DataFrame, curve: pd.DataFrame, cfg: dict):
    q = cfg["qc"]
    s = (df.set_index("tanggal")["tma"]
           .reindex(pd.date_range(cfg["periode"]["awal"],
                                  cfg["periode"]["akhir"], freq="D")))
    n_kosong_awal = int(s.isna().sum())

    # -- deteksi anomali rentang fisik (berdasarkan storage curve)
    n_range = 0
    if curve is not None and len(curve) > 1:
        lo = curve["elevasi"].min() - 2.0
        hi = curve["elevasi"].max() + 2.0
        mask = (s < lo) | (s > hi)
        n_range = int(mask.sum())
        s[mask] = np.nan

    # -- deteksi spike: selisih terhadap median rolling 7 hari
    med = s.rolling(7, center=True, min_periods=3).median()
    mask_spike = (s - med).abs() > q["batas_perubahan_harian_m"]
    n_spike = int(mask_spike.sum())
    s[mask_spike] = np.nan

    # -- interpolasi gap pendek
    s_interp = s.interpolate(method=q["metode_interpolasi"],
                             limit=q["max_gap_interpolasi"],
                             limit_area="inside")
    n_interp = int(s.isna().sum() - s_interp.isna().sum())

    # -- sisa gap: ambil data sebelumnya (forward fill), lalu bfill utk awal deret
    s_final = s_interp.ffill().bfill()
    n_ffill = int(s_interp.isna().sum())

    rekap = {"data_kosong": n_kosong_awal, "anomali_rentang": n_range,
             "anomali_spike": n_spike, "diinterpolasi": n_interp,
             "diisi_data_sebelumnya": n_ffill}
    hasil = s_final.rename("tma").rename_axis("tanggal").reset_index()
    return hasil, rekap


def main():
    cfg = load_config()
    raw = path_root("data", "raw")
    # tanggal dari parquet raw bisa berupa datetime.date (object); diseragamkan
    # agar reindex ke kalender harian di qc_satu_bendungan cocok tepat
    tma = normalisasi_tanggal(pd.read_parquet(
        os.path.join(raw, "tma_harian.parquet")))
    curve = pd.read_parquet(os.path.join(raw, "storagecurve_harian.parquet"))

    hasil, rekap_rows, ditolak = [], [], []
    for kode, g in tma.groupby("kode_bendungan"):
        c = curve[curve["kode_bendungan"] == kode]
        h, rk = qc_satu_bendungan(g, c, cfg)
        rk["kode_bendungan"] = kode

        # Bila SELURUH nilai ditolak QC (mis. TMA di luar rentang storage curve
        # karena pemetaan id/satuan tidak cocok), ffill/bfill tidak punya nilai
        # acuan sehingga deret tetap NaN seluruhnya. Bendungan seperti ini
        # dikeluarkan dari output -- kalau diloloskan, NaN akan masuk ke fitur
        # dan membuat loss training model global menjadi NaN.
        if h["tma"].isna().all():
            rk["lolos_qc"] = False
            rk["alasan"] = ("seluruh nilai TMA ditolak QC (di luar rentang "
                            "storage curve atau data kosong)")
            ditolak.append(kode)
            rekap_rows.append(rk)
            print(f"LEWATI {kode}: seluruh nilai TMA ditolak QC "
                  f"(kosong={rk['data_kosong']}, di luar rentang="
                  f"{rk['anomali_rentang']}, spike={rk['anomali_spike']})")
            continue

        rk["lolos_qc"] = True
        rk["alasan"] = ""
        h["kode_bendungan"] = kode
        hasil.append(h)
        rekap_rows.append(rk)
        print(f"QC {kode}: {rk}")

    if not hasil:
        raise RuntimeError(
            "Tidak ada bendungan yang lolos QC. Cek rentang storage curve vs "
            "nilai TMA (indikasi pemetaan id database atau satuan tidak cocok).")
    if ditolak:
        print(f"\n[QC] {len(ditolak)} bendungan dikeluarkan karena seluruh "
              f"nilainya ditolak: {ditolak}")

    df_qc = pd.concat(hasil, ignore_index=True)

    # agregasi bulanan: AVERAGE (dipertahankan utk kompatibilitas/QA)
    df_qc["bulan_periode"] = df_qc["tanggal"].dt.to_period("M")
    bulanan = (df_qc.groupby(["kode_bendungan", "bulan_periode"])["tma"]
                    .mean().round(3).reset_index())
    bulanan["tanggal"] = bulanan["bulan_periode"].dt.to_timestamp()
    bulanan = bulanan[["kode_bendungan", "tanggal", "tma"]]

    # agregasi PERIODE (10/15 harian sesuai format RTOW per bendungan)
    daftar = baca_daftar_bendungan(cfg)
    fmt_map = baca_format_rtow(cfg, daftar)
    per_frames = []
    for kode, g in df_qc.groupby("kode_bendungan"):
        fmt = fmt_map.get(kode, "15 Harian")
        p = agregasi_periode(g, fmt, "tma", "mean")
        p["tma"] = p["tma"].round(3)
        p["kode_bendungan"] = kode
        p["format_periode"] = fmt
        per_frames.append(p)
    periode = pd.concat(per_frames, ignore_index=True)[
        ["kode_bendungan", "tanggal", "periode", "format_periode", "tma"]]

    out = path_root("data", "processed")
    os.makedirs(out, exist_ok=True)
    df_qc.drop(columns="bulan_periode").to_parquet(
        os.path.join(out, "tma_harian_qc.parquet"), index=False)
    bulanan.to_parquet(os.path.join(out, "tma_bulanan.parquet"), index=False)
    periode.to_parquet(os.path.join(out, "tma_periode.parquet"), index=False)
    print(f"OK  tma_periode.parquet ({len(periode):,} baris periode, "
          f"{periode['kode_bendungan'].nunique()} bendungan)")

    rekap = pd.DataFrame(rekap_rows)[
        ["kode_bendungan", "data_kosong", "anomali_rentang", "anomali_spike",
         "diinterpolasi", "diisi_data_sebelumnya", "lolos_qc", "alasan"]]
    rekap.to_excel(os.path.join(out, "rekap_qc.xlsx"), index=False)
    print(f"\nOK  tma_bulanan.parquet ({len(bulanan):,} baris bulanan, "
          f"{bulanan['kode_bendungan'].nunique()} bendungan)")
    print(f"OK  rekap_qc.xlsx")


if __name__ == "__main__":
    main()
