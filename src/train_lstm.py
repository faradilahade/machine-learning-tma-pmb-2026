"""Pelatihan model LSTM skala PERIODE (10/15-harian sesuai format RTOW).

Dua model global (pooled) terpisah — satu per format periode:
  '15 Harian' : 24 periode/tahun -> input 24, output 10 (Agu-Des)
  '10 Harian' : 36 periode/tahun -> input 36, output 15 (Agu-Des)

Fitur per timestep:
  1. tma_norm       : TMA per periode, min-max per bendungan
  2. inflow_norm    : volume inflow per periode, min-max per bendungan
  3. sin_pos/cos_pos: posisi periode dalam tahun kalender (musiman siklis)
  4. skor_musim     : sifat musim tahun air (sifat_musim.csv)
                      Kering=-1, Normal=0, Basah=+1 — model belajar bahwa
                      tahun Basah/Kering menggeser pola resesi TMA, sehingga
                      prediksi 2026 mengikuti karakter tahun 2025-2026.
                      (tahun sebelum 2022-2023 dianggap Normal)

Output:
  output/models/lstm_tma_periode_15H.keras
  output/models/lstm_tma_periode_10H.keras
  output/models/skala_periode.json   (skala per bendungan + format + meta)
  output/models/riwayat_periode_<fmt>.csv
"""
import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (load_config, path_root, baca_daftar_bendungan,
                       baca_sifat_musim, skor_musim_untuk,
                       PERIODE_PER_BULAN, FORMAT_15, FORMAT_10)

FITUR = ["tma_norm", "inflow_norm", "sin_pos", "cos_pos", "skor_musim"]
NAMA_FILE = {FORMAT_15: "15H", FORMAT_10: "10H"}
# horizon Agustus-Desember = 5 bulan x periode/bulan
OUTPUT_PERIODE = {FORMAT_15: 10, FORMAT_10: 15}


def siapkan_fitur(df: pd.DataFrame, peta_musim: dict):
    """Normalisasi per bendungan + fitur posisi tahunan + sifat musim."""
    skala, frames = {}, []
    for kode, g in df.groupby("kode_bendungan"):
        g = g.sort_values("tanggal").copy()
        fmt = g["format_periode"].iloc[0]
        n_thn = 12 * PERIODE_PER_BULAN[fmt]
        t_min, t_max = g["tma"].min(), g["tma"].max()
        i_min, i_max = g["inflow_juta_m3"].min(), g["inflow_juta_m3"].max()
        rng_t = (t_max - t_min) or 1.0
        rng_i = (i_max - i_min) or 1.0
        g["tma_norm"] = (g["tma"] - t_min) / rng_t
        g["inflow_norm"] = (g["inflow_juta_m3"] - i_min) / rng_i
        bln = g["periode"].str[:2].astype(int)
        idx = g["periode"].str[3:].astype(int)
        pos = (bln - 1) * PERIODE_PER_BULAN[fmt] + (idx - 1)
        g["sin_pos"] = np.sin(2 * np.pi * pos / n_thn)
        g["cos_pos"] = np.cos(2 * np.pi * pos / n_thn)
        g["skor_musim"] = [skor_musim_untuk(t, peta_musim, kode)
                           for t in g["tanggal"]]
        skala[kode] = {"tma_min": float(t_min), "tma_max": float(t_max),
                       "inflow_min": float(i_min), "inflow_max": float(i_max),
                       "format": fmt}
        frames.append(g)
    return pd.concat(frames, ignore_index=True), skala


def buat_sekuens(df: pd.DataFrame, n_in: int, n_out: int):
    X, y = [], []
    for _, g in df.groupby("kode_bendungan"):
        g = g.sort_values("tanggal")
        arr = g[FITUR].values.astype("float32")
        target = g["tma_norm"].values.astype("float32")
        for i in range(len(g) - n_in - n_out + 1):
            X.append(arr[i:i + n_in])
            y.append(target[i + n_in:i + n_in + n_out])
    return np.array(X), np.array(y)[..., None]


def bangun_model(n_in, n_out, n_fitur, m):
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers, regularizers
    tf.keras.utils.set_random_seed(m["seed"])
    model = models.Sequential([
        layers.Input(shape=(n_in, n_fitur)),
        layers.LSTM(m["hidden_units"], kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(m["dropout"]),
        layers.RepeatVector(n_out),
        layers.LSTM(m["hidden_units"], return_sequences=True),
        layers.Dropout(m["dropout"]),
        layers.TimeDistributed(layers.Dense(1)),
    ])
    model.compile(optimizer=optimizers.Adam(m["learning_rate"]), loss="mse",
                  metrics=["mae"])
    return model


def latih_grup(df_grup, fmt, m, out_dir):
    from tensorflow.keras import callbacks
    n_in = 12 * PERIODE_PER_BULAN[fmt] * m.get("input_tahun", 1)
    n_out = OUTPUT_PERIODE[fmt]
    X, y = buat_sekuens(df_grup, n_in, n_out)
    print(f"[{fmt}] {df_grup['kode_bendungan'].nunique()} bendungan, "
          f"sekuens X={X.shape} y={y.shape}")
    n_tr = int(len(X) * 0.8)
    model = bangun_model(n_in, n_out, X.shape[2], m)
    fp = os.path.join(out_dir, f"lstm_tma_periode_{NAMA_FILE[fmt]}.keras")
    hist = model.fit(
        X[:n_tr], y[:n_tr], validation_data=(X[n_tr:], y[n_tr:]),
        epochs=m["epochs"], batch_size=m["batch_size"], verbose=2,
        callbacks=[
            callbacks.EarlyStopping(patience=m["patience"],
                                    restore_best_weights=True),
            callbacks.ReduceLROnPlateau(patience=max(m["patience"] // 2, 3),
                                        factor=0.5, min_lr=1e-5),
            callbacks.ModelCheckpoint(fp, save_best_only=True),
        ])
    model.save(fp)
    pd.DataFrame(hist.history).to_csv(
        os.path.join(out_dir, f"riwayat_periode_{NAMA_FILE[fmt]}.csv"),
        index=False)
    va = float(np.min(hist.history["val_mae"]))
    print(f"[{fmt}] OK {fp}  val MAE (norm 0-1): {va:.4f}")
    return va


def main():
    cfg = load_config()
    m = cfg["model"]
    daftar = baca_daftar_bendungan(cfg)
    musim = baca_sifat_musim(cfg, daftar)
    peta_musim = {k: dict(zip(g["tahun_air"], g["skor_musim"]))
                  for k, g in musim.groupby("kode_bendungan")}

    df = pd.read_parquet(path_root("data", "processed",
                                   "gabungan_periode.parquet"))
    df, skala = siapkan_fitur(df, peta_musim)

    out_dir = path_root(cfg["output"]["dir_model"])
    os.makedirs(out_dir, exist_ok=True)

    hasil = {}
    for fmt in (FORMAT_15, FORMAT_10):
        grup = df[df["format_periode"] == fmt]
        if grup.empty:
            print(f"[{fmt}] tidak ada bendungan — dilewati")
            continue
        hasil[fmt] = latih_grup(grup, fmt, m, out_dir)

    with open(os.path.join(out_dir, "skala_periode.json"), "w") as f:
        json.dump({"skala": skala, "fitur": FITUR,
                   "output_periode": OUTPUT_PERIODE,
                   "val_mae": hasil}, f, indent=2)
    print("\nOK  skala_periode.json")


if __name__ == "__main__":
    main()
