"""Perhitungan volume inflow dengan neraca air (harian), lalu agregasi bulanan.

Rumus (per hari, sesuai spesifikasi):
    V_inflow(t) = V(t) + Q_outflow(t) * 60 * 60 * 24 - V(t-1)

dengan:
    V(t)        = volume tampungan hari t (m3), hasil interpolasi
                  TMA hasil QC ke storage curve (elevasi -> volume)
    Q_outflow(t)= outflow total (m3/detik) = jumlah komponen turbin, abaku,
                  aindustri, irigasi, limpas, pemeliharaan
                  (sudah dijumlahkan di tahap ekstraksi)

Volume inflow bulanan = SUM(V_inflow harian) dalam sebulan (juta m3).
Nilai negatif kecil (error pengukuran) dipotong ke 0.

Catatan satuan: kolom volume pada rawdata.sinbad_bendungan_storagecurve
bersatuan m3 (BUKAN juta m3). Diverifikasi empiris terhadap inflow tercatat
SINBAD: neraca air dV + Qout*86400 dibanding inflow_tercatat*86400 memberi
rasio median 1,01 pada 135 bendungan bila volume kurva diperlakukan sebagai m3.
Konversi ke juta m3 hanya dilakukan sekali, saat agregasi bulanan.

Output: data/processed/inflow_bulanan.parquet
        data/processed/gabungan_bulanan.parquet  (TMA + inflow, siap model)
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (load_config, path_root, normalisasi_tanggal,
                       baca_daftar_bendungan, baca_format_rtow,
                       agregasi_periode)

DETIK_PER_HARI = 60 * 60 * 24


def tma_ke_volume(tma: pd.Series, curve: pd.DataFrame) -> np.ndarray:
    """Interpolasi linear elevasi -> volume tampungan (m3) dari storage curve.

    Di luar rentang kurva, np.interp menahan nilai pada volume ujung (clamp),
    sehingga TMA di luar rentang fisik tidak menghasilkan volume ekstrem.
    """
    c = curve.sort_values("elevasi")
    return np.interp(tma.values, c["elevasi"].values, c["volume"].values)


def main():
    cfg = load_config()
    batas_out = cfg["qc"].get("batas_outflow_m3s", 10000)
    proc, raw = path_root("data", "processed"), path_root("data", "raw")

    # dtype tanggal diseragamkan: parquet raw menyimpan datetime.date (object),
    # hasil QC sudah datetime64 -> tanpa ini merge di bawah gagal
    tma = normalisasi_tanggal(pd.read_parquet(
        os.path.join(proc, "tma_harian_qc.parquet")))
    outflow = normalisasi_tanggal(pd.read_parquet(
        os.path.join(raw, "outflow_harian.parquet")))
    curve = pd.read_parquet(os.path.join(raw, "storagecurve_harian.parquet"))

    hasil, n_anomali, tanpa_outflow = [], 0, []
    for kode, g in tma.groupby("kode_bendungan"):
        g = g.sort_values("tanggal").copy()
        c = curve[curve["kode_bendungan"] == kode]
        if len(c) < 2:
            print(f"LEWATI {kode}: storage curve tidak tersedia")
            continue

        # volume tampungan harian (m3) -- kurva SINBAD sudah bersatuan m3
        g["volume_m3"] = tma_ke_volume(g["tma"], c)

        # gabung outflow; kosong/anomali -> ambil data sebelumnya (ffill)
        o = outflow[outflow["kode_bendungan"] == kode][["tanggal", "outflow"]]
        if o.empty:
            tanpa_outflow.append(kode)
            g["outflow"] = 0.0
        else:
            g = g.merge(o.drop_duplicates("tanggal"), on="tanggal", how="left")
            # anomali outflow: negatif, atau di luar batas fisik (salah entri
            # SINBAD, mis. 3,2 juta m3/detik) -> di-NaN-kan lalu di-ffill.
            # Satu hari bogus bisa menambah miliaran m3 ke inflow bulanan dan
            # merusak normalisasi min-max fitur bendungan tsb.
            anomali = (g["outflow"] < 0) | (g["outflow"] > batas_out)
            n_anomali += int(anomali.sum())
            g.loc[anomali, "outflow"] = np.nan
            g["outflow"] = g["outflow"].ffill().bfill().fillna(0.0)

        # neraca air harian: V_in = V(t) + Qout*86400 - V(t-1)
        g["v_inflow_m3"] = (g["volume_m3"]
                            + g["outflow"] * DETIK_PER_HARI
                            - g["volume_m3"].shift(1))
        g.loc[g["v_inflow_m3"] < 0, "v_inflow_m3"] = 0.0   # potong negatif
        hasil.append(g)

    if not hasil:
        raise RuntimeError(
            "Tidak ada bendungan yang bisa dihitung: storage curve kosong untuk "
            "semua bendungan. Cek data/raw/storagecurve_harian.parquet.")
    if n_anomali:
        print(f"  [outflow] {n_anomali} nilai anomali (negatif atau "
              f"> {batas_out:,} m3/detik) diisi data sebelumnya")
    if tanpa_outflow:
        print(f"  [outflow] tanpa data outflow, dianggap 0: "
              f"{len(tanpa_outflow)} bendungan (contoh: {tanpa_outflow[:5]})")

    df = pd.concat(hasil, ignore_index=True)

    # agregasi bulanan: SUM volume inflow (dalam juta m3)
    df["bulan_periode"] = df["tanggal"].dt.to_period("M")
    inflow_bln = (df.groupby(["kode_bendungan", "bulan_periode"])["v_inflow_m3"]
                    .sum().div(1e6).round(3)
                    .rename("inflow_juta_m3").reset_index())
    inflow_bln["tanggal"] = inflow_bln["bulan_periode"].dt.to_timestamp()
    inflow_bln = inflow_bln[["kode_bendungan", "tanggal", "inflow_juta_m3"]]
    inflow_bln.to_parquet(os.path.join(proc, "inflow_bulanan.parquet"), index=False)

    # gabungan siap model: TMA bulanan + inflow bulanan
    tma_bln = normalisasi_tanggal(pd.read_parquet(
        os.path.join(proc, "tma_bulanan.parquet")))
    gab = tma_bln.merge(inflow_bln, on=["kode_bendungan", "tanggal"], how="left")
    gab["inflow_juta_m3"] = (gab.groupby("kode_bendungan")["inflow_juta_m3"]
                                .transform(lambda s: s.ffill().bfill()))
    # bendungan tanpa inflow sama sekali (mis. storage curve dilewati) tetap NaN
    # setelah ffill/bfill -> diisi 0 agar tidak menghasilkan NaN saat training
    kosong = gab["inflow_juta_m3"].isna()
    if kosong.any():
        kode_kosong = sorted(gab.loc[kosong, "kode_bendungan"].unique())
        print(f"  [inflow] tanpa nilai inflow, diisi 0: {len(kode_kosong)} "
              f"bendungan (contoh: {kode_kosong[:5]})")
        gab["inflow_juta_m3"] = gab["inflow_juta_m3"].fillna(0.0)

    # jaring pengaman: TMA bulanan NaN (mis. seluruh nilai ditolak QC) tidak
    # boleh lolos ke fitur model -- NaN membuat loss training menjadi NaN
    tma_nan = gab["tma"].isna()
    if tma_nan.any():
        kode_nan = sorted(gab.loc[tma_nan, "kode_bendungan"].unique())
        print(f"  [tma] {int(tma_nan.sum())} baris TMA NaN dibuang dari "
              f"gabungan_bulanan: {len(kode_nan)} bendungan (contoh: {kode_nan[:5]})")
        gab = gab[~tma_nan].reset_index(drop=True)
    gab.to_parquet(os.path.join(proc, "gabungan_bulanan.parquet"), index=False)
    print(f"OK  inflow_bulanan.parquet ({len(inflow_bln):,} baris)")
    print(f"OK  gabungan_bulanan.parquet ({len(gab):,} baris, "
          f"{gab['kode_bendungan'].nunique()} bendungan)")

    # ---------------- skala PERIODE (10/15 harian sesuai format RTOW) -------
    daftar = baca_daftar_bendungan(cfg)
    fmt_map = baca_format_rtow(cfg, daftar)
    frames = []
    for kode, g in df.groupby("kode_bendungan"):
        fmt = fmt_map.get(kode, "15 Harian")
        p = agregasi_periode(g, fmt, "v_inflow_m3", "sum")
        p["inflow_juta_m3"] = (p["v_inflow_m3"] / 1e6).round(3)
        p["kode_bendungan"] = kode
        frames.append(p[["kode_bendungan", "tanggal", "periode",
                         "inflow_juta_m3"]])
    inflow_per = pd.concat(frames, ignore_index=True)

    tma_per = normalisasi_tanggal(pd.read_parquet(
        os.path.join(proc, "tma_periode.parquet")))
    gab_per = tma_per.merge(inflow_per,
                            on=["kode_bendungan", "tanggal", "periode"],
                            how="left")
    gab_per["inflow_juta_m3"] = (gab_per.groupby("kode_bendungan")
                                 ["inflow_juta_m3"]
                                 .transform(lambda s: s.ffill().bfill())
                                 .fillna(0.0))
    tma_nan = gab_per["tma"].isna()
    if tma_nan.any():
        gab_per = gab_per[~tma_nan].reset_index(drop=True)
    gab_per.to_parquet(os.path.join(proc, "gabungan_periode.parquet"),
                       index=False)
    print(f"OK  gabungan_periode.parquet ({len(gab_per):,} baris, "
          f"{gab_per['kode_bendungan'].nunique()} bendungan)")


if __name__ == "__main__":
    main()
