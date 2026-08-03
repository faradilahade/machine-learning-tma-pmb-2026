"""Prediksi TMA Agustus-Desember 2026 per PERIODE (10/15-harian per bendungan).

Input model : 1 tahun periode terakhir (Agu 2025 - Jul 2026)
Output      : periode Agustus-Desember 2026 (10 atau 15 langkah sesuai format)

Status per periode terhadap BON (kategori utama rekapitulasi):
  "Di atas BON A"            : TMA >= BON A
  "Di antara BON A dan BON B": BON B <= TMA < BON A
  "Di bawah BON B"           : TMA < BON B
  "Tanpa Data BON"           : BON A/B tidak tersedia

Hasil:
  output/prediksi/prediksi_tma_2026.parquet  (skala periode, label 'MM-PP')
  output/prediksi/prediksi_tma_2026.xlsx
  output/grafik/sandingan_<kode>.png         (QA statis 20 bendungan pertama)
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (load_config, path_root, baca_daftar_bendungan, baca_bon,
                       baca_rtow, baca_sifat_musim, skor_musim_untuk,
                       daftar_periode, awal_periode, PERIODE_PER_BULAN,
                       FORMAT_15, FORMAT_10)
from src.train_lstm import FITUR, NAMA_FILE, OUTPUT_PERIODE

WARNA = {"navy": "#0F2A47", "biru": "#1C5D8C", "teal": "#2A9D8F",
         "amber": "#E9C46A", "coral": "#E76F51",
         "hijau": "#2E7D32", "merah": "#C62828"}

ATAS_A = "Di atas BON A"
ANTARA = "Di antara BON A dan BON B"
BAWAH_B = "Di bawah BON B"
TANPA_BON = "Tanpa Data BON"


def fitur_bendungan(g: pd.DataFrame, s: dict, peta_musim: dict, kode: str):
    """Bangun fitur PERSIS seperti training (lihat train_lstm.siapkan_fitur)."""
    fmt = s["format"]
    n_thn = 12 * PERIODE_PER_BULAN[fmt]
    rng_t = (s["tma_max"] - s["tma_min"]) or 1.0
    rng_i = (s["inflow_max"] - s["inflow_min"]) or 1.0
    g = g.sort_values("tanggal").copy()
    g["tma_norm"] = (g["tma"] - s["tma_min"]) / rng_t
    g["inflow_norm"] = (g["inflow_juta_m3"] - s["inflow_min"]) / rng_i
    bln = g["periode"].str[:2].astype(int)
    idx = g["periode"].str[3:].astype(int)
    pos = (bln - 1) * PERIODE_PER_BULAN[fmt] + (idx - 1)
    g["sin_pos"] = np.sin(2 * np.pi * pos / n_thn)
    g["cos_pos"] = np.cos(2 * np.pi * pos / n_thn)
    g["skor_musim"] = [skor_musim_untuk(t, peta_musim, kode)
                       for t in g["tanggal"]]
    return g


def prediksi_per_bendungan(model, g, s, peta_musim, kode, n_in, n_out):
    g = fitur_bendungan(g, s, peta_musim, kode)
    X = g[FITUR].tail(n_in).values.astype("float32")[None, ...]
    if X.shape[1] < n_in:
        raise ValueError(f"{kode}: histori < {n_in} periode")
    rng_t = (s["tma_max"] - s["tma_min"]) or 1.0
    y = model.predict(X, verbose=0)[0, :n_out, 0]
    return y * rng_t + s["tma_min"]


def plot_sandingan(kode, nama, fmt, g26, out_png):
    """QA statis: sumbu-x label periode 'MM-PP' Jan-Des."""
    urut = daftar_periode(fmt, 1, 12)
    d = g26.set_index("periode").reindex(urut)
    x = np.arange(len(urut))
    fig, ax = plt.subplots(figsize=(12.5, 5.5), dpi=120)
    for kol, warna, gaya, label in [("rtow", WARNA["biru"], ":", "RTOW"),
                                    ("bon_a", WARNA["amber"], "-.", "BON A"),
                                    ("bon_b", WARNA["coral"], "-.", "BON B")]:
        if kol in d.columns and d[kol].notna().any():
            ax.plot(x, d[kol].values, color=warna, ls=gaya, lw=2, label=label)
    real = d[d["jenis"] == "realisasi"]
    pred = d[d["jenis"] == "prediksi"]
    ax.plot([urut.index(p) for p in real.index], real["tma"], color=WARNA["hijau"],
            lw=2.5, marker="o", ms=4, label="TMA Realisasi 2026")
    ax.plot([urut.index(p) for p in pred.index], pred["tma"], color=WARNA["merah"],
            lw=2.5, ls="--", marker="s", ms=4, label="TMA Prediksi (Agu-Des)")
    ax.set_xticks(x)
    ax.set_xticklabels(urut, rotation=90, fontsize=7)
    ax.set_ylabel("TMA (mdpl)")
    ax.set_title(f"Sandingan TMA per Periode {fmt} 2026 — {nama} ({kode})",
                 fontsize=12, fontweight="bold", color=WARNA["navy"])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def main():
    from tensorflow.keras.models import load_model
    cfg = load_config()
    daftar = baca_daftar_bendungan(cfg)
    bon = baca_bon(cfg, daftar)          # bulanan
    rtow = baca_rtow(cfg, daftar)        # bulanan
    musim = baca_sifat_musim(cfg, daftar)
    peta_musim = {k: dict(zip(g["tahun_air"], g["skor_musim"]))
                  for k, g in musim.groupby("kode_bendungan")}

    df = pd.read_parquet(path_root("data", "processed",
                                   "gabungan_periode.parquet"))
    dir_model = path_root(cfg["output"]["dir_model"])
    with open(os.path.join(dir_model, "skala_periode.json")) as f:
        meta = json.load(f)
    skala = meta["skala"]
    model = {fmt: load_model(os.path.join(
        dir_model, f"lstm_tma_periode_{NAMA_FILE[fmt]}.keras"))
        for fmt in (FORMAT_15, FORMAT_10)}

    dir_pred = path_root(cfg["output"]["dir_prediksi"])
    dir_graf = path_root(cfg["output"]["dir_grafik"])
    os.makedirs(dir_pred, exist_ok=True)
    os.makedirs(dir_graf, exist_ok=True)

    m = cfg["model"]
    rows, n_png = [], 0
    for _, r in daftar.iterrows():
        kode, nama = r["kode_bendungan"], r["nama_bendungan"]
        if kode not in skala:
            print(f"LEWATI {kode} ({nama}): tidak ada di data historis")
            continue
        s = skala[kode]
        fmt = s["format"]
        n_in = 12 * PERIODE_PER_BULAN[fmt] * m.get("input_tahun", 1)
        n_out = OUTPUT_PERIODE[fmt]
        g = df[df["kode_bendungan"] == kode]
        try:
            pred = prediksi_per_bendungan(model[fmt], g, s, peta_musim,
                                          kode, n_in, n_out)
        except ValueError as e:
            print(f"LEWATI {kode}: {e}")
            continue

        g26 = g[(g["tanggal"] >= "2026-01-01")
                & (g["tanggal"] <= "2026-07-31")].sort_values("tanggal")
        for t, p, v in zip(g26["tanggal"], g26["periode"], g26["tma"]):
            rows.append([kode, nama, fmt, t, p, round(float(v), 3), "realisasi"])
        lbl_pred = daftar_periode(fmt, 8, 12)
        for lbl, v in zip(lbl_pred, pred):
            b, i = int(lbl[:2]), int(lbl[3:])
            rows.append([kode, nama, fmt, awal_periode(2026, b, i, fmt),
                         lbl, round(float(v), 3), "prediksi"])
        print(f"OK  {kode} {nama} [{fmt}]: "
              f"{np.round(pred[:4], 2).tolist()} ... ({n_out} periode)")

    hasil = pd.DataFrame(rows, columns=["kode_bendungan", "nama_bendungan",
                                        "format_periode", "tanggal", "periode",
                                        "tma", "jenis"])
    hasil["bulan"] = hasil["tanggal"].dt.month
    hasil = hasil.merge(bon.drop(columns="nama_bendungan"),
                        on=["kode_bendungan", "bulan"], how="left")
    hasil = hasil.merge(rtow.drop(columns="nama_bendungan"),
                        on=["kode_bendungan", "bulan"], how="left")
    hasil["status_bon"] = np.select(
        [hasil["bon_a"].isna() & hasil["bon_b"].isna(),
         hasil["tma"] < hasil["bon_b"],
         hasil["tma"] < hasil["bon_a"]],
        [TANPA_BON, BAWAH_B, ANTARA], ATAS_A)
    hasil["selisih_rtow"] = (hasil["tma"] - hasil["rtow"]).round(3)

    for kode, g26 in hasil.groupby("kode_bendungan"):
        if n_png >= 20:
            break
        plot_sandingan(kode, g26["nama_bendungan"].iloc[0],
                       g26["format_periode"].iloc[0], g26,
                       os.path.join(dir_graf, f"sandingan_{kode}.png"))
        n_png += 1

    hasil.to_parquet(os.path.join(dir_pred, "prediksi_tma_2026.parquet"),
                     index=False)
    hasil.to_excel(os.path.join(dir_pred, "prediksi_tma_2026.xlsx"), index=False)
    print(f"\nOK  prediksi_tma_2026.parquet ({len(hasil):,} baris, "
          f"{hasil['kode_bendungan'].nunique()} bendungan, skala periode)")


if __name__ == "__main__":
    main()
