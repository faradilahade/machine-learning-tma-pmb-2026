"""Evaluasi akurasi prediksi TMA: backtest LSTM vs metode pembanding.

Prinsip: mundur ke titik waktu (cutoff) yang 5 bulan berikutnya sudah punya
realisasi, prediksi dengan beberapa metode dari data SEBELUM cutoff saja,
lalu bandingkan dengan realisasi.

Periode uji:
  - Agu-Des 2025 : kalender sama dengan horizon prediksi 2026 (musim kering)
  - Mar-Jul 2026 : periode terbaru yang realisasinya tersedia

Metode yang dibandingkan:
  1. LSTM            : model encoder-decoder yang dipakai produksi
  2. Persistensi     : TMA bulan terakhir dianggap konstan (naive baseline)
  3. Naif musiman    : TMA bulan yang sama tahun sebelumnya
  4. Klimatologi     : rata-rata TMA bulan kalender yang sama seluruh histori
                       sebelum cutoff
  5. Tren linier     : regresi linier 6 bulan terakhir, diekstrapolasi

Metrik: MAE, RMSE, bias (rata-rata error), MAPE terhadap RENTANG TMA
bendungan (bukan nilai absolut, karena TMA bersatuan elevasi -- persen
terhadap mdpl tidak bermakna).

Catatan jujur: sekuens periode uji ikut berada di rentang data training model
produksi (split validasi 80/20 tidak memisahkan berdasarkan waktu kalender),
sehingga skor LSTM di sini cenderung optimis. Baseline tetap adil karena
tidak "belajar" apa pun.

Output: output/prediksi/evaluasi_metode.parquet  (baris = kode x periode x
        metode x bulan, kolom y_true/y_pred/error)
        output/prediksi/evaluasi_metode.xlsx     (ringkasan per metode)
"""
import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (load_config, path_root, baca_daftar_bendungan,
                       baca_sifat_musim, PERIODE_PER_BULAN,
                       FORMAT_15, FORMAT_10)
from src.train_lstm import FITUR, NAMA_FILE, OUTPUT_PERIODE

METODE_LSTM = "LSTM (produksi)"
METODE_INFO = {
    METODE_LSTM: "Model LSTM encoder-decoder produksi (1 tahun periode → "
                 "periode Agu-Des), termasuk fitur sifat musim",
    "Persistensi": "TMA periode terakhir sebelum cutoff dianggap konstan",
    "Naif musiman": "TMA periode kalender yang sama pada tahun sebelumnya",
    "Klimatologi": "Rata-rata TMA periode kalender yang sama, seluruh histori "
                   "sebelum cutoff",
    "Tren linier": "Regresi linier 12 periode terakhir sebelum cutoff, "
                   "diekstrapolasi sepanjang horizon",
}


def prediksi_lstm(model, g, s, peta_musim, kode, n_in, n_out):
    """Prediksi n_out PERIODE dari n_in periode terakhir (sudah dipotong cutoff)."""
    from src.predict_2026 import fitur_bendungan
    g = fitur_bendungan(g, s, peta_musim, kode)
    X = g[FITUR].tail(n_in).values.astype("float32")[None, ...]
    if X.shape[1] < n_in:
        return None
    rng_t = (s["tma_max"] - s["tma_min"]) or 1.0
    y = model.predict(X, verbose=0)[0, :n_out, 0]
    return y * rng_t + s["tma_min"]


def prediksi_baseline(g, lbl_uji, n_per_tahun):
    """Empat baseline dari data sebelum cutoff (skala periode)."""
    hasil = {}
    n_out = len(lbl_uji)

    hasil["Persistensi"] = np.repeat(g["tma"].iloc[-1], n_out)

    # naif musiman: periode label sama tahun sebelumnya
    # (= n_per_tahun langkah sebelum titik uji)
    per_label_tahun_lalu = g.tail(n_per_tahun).set_index("periode")["tma"]
    hasil["Naif musiman"] = np.array(
        [per_label_tahun_lalu.get(lbl, np.nan) for lbl in lbl_uji])

    klim = g.groupby("periode")["tma"].mean()
    hasil["Klimatologi"] = np.array([klim.get(lbl, np.nan) for lbl in lbl_uji])

    akhir = g.tail(12)     # ~1/2 tahun (15H) atau 4 bulan (10H)
    if len(akhir) >= 4:
        x = np.arange(len(akhir), dtype=float)
        b, a = np.polyfit(x, akhir["tma"].values, 1)
        x_dep = np.arange(len(akhir), len(akhir) + n_out, dtype=float)
        hasil["Tren linier"] = a + b * x_dep
    else:
        hasil["Tren linier"] = np.full(n_out, np.nan)
    return hasil


def jalankan_backtest(df, model, skala, peta_musim, cutoffs, input_tahun=1):
    baris = []
    for label, cutoff in cutoffs.items():
        cutoff = pd.Timestamp(cutoff)
        for kode, g in df.groupby("kode_bendungan"):
            if kode not in skala:
                continue
            s = skala[kode]
            fmt = s["format"]
            n_per_tahun = 12 * PERIODE_PER_BULAN[fmt]
            n_in = n_per_tahun * input_tahun
            n_out = OUTPUT_PERIODE[fmt]
            g = g.sort_values("tanggal")
            latih = g[g["tanggal"] <= cutoff]
            uji = g[g["tanggal"] > cutoff].head(n_out)
            if len(latih) < n_in or len(uji) < n_out or uji["tma"].isna().any():
                continue
            lbl_uji = uji["periode"].tolist()

            pred = {}
            y = prediksi_lstm(model[fmt], latih, s, peta_musim, kode,
                              n_in, n_out)
            if y is not None:
                pred[METODE_LSTM] = y
            pred.update(prediksi_baseline(latih, lbl_uji, n_per_tahun))

            rng = g["tma"].max() - g["tma"].min()
            for t, lbl, yt, ys in zip(uji["tanggal"], lbl_uji, uji["tma"],
                                      zip(*[pred[k] for k in pred])):
                for metode, yp in zip(pred, ys):
                    baris.append({
                        "kode_bendungan": kode, "periode_uji": label,
                        "metode": metode, "tanggal": t, "periode": lbl,
                        "y_true": round(float(yt), 3),
                        "y_pred": round(float(yp), 3) if np.isfinite(yp) else np.nan,
                        "error": round(float(yp - yt), 3) if np.isfinite(yp) else np.nan,
                        "rentang_tma": round(float(rng), 3),
                    })
    return pd.DataFrame(baris)


def ringkas(detail: pd.DataFrame) -> pd.DataFrame:
    """Agregasi metrik per periode x metode."""
    d = detail.dropna(subset=["error"]).copy()
    d["abs_err"] = d["error"].abs()
    d["sq_err"] = d["error"] ** 2
    d["nrmae"] = d["abs_err"] / d["rentang_tma"].replace(0, np.nan)
    r = (d.groupby(["periode_uji", "metode"])
           .agg(n_prediksi=("error", "size"),
                n_bendungan=("kode_bendungan", "nunique"),
                mae_m=("abs_err", "mean"),
                rmse_m=("sq_err", lambda s: float(np.sqrt(s.mean()))),
                bias_m=("error", "mean"),
                mape_rentang_pct=("nrmae", lambda s: 100 * s.mean()))
           .round(3).reset_index())
    return r.sort_values(["periode_uji", "mae_m"]).reset_index(drop=True)


def main():
    from tensorflow.keras.models import load_model
    cfg = load_config()
    m = cfg["model"]
    daftar = baca_daftar_bendungan(cfg)
    musim = baca_sifat_musim(cfg, daftar)
    peta_musim = {k: dict(zip(g["tahun_air"], g["skor_musim"]))
                  for k, g in musim.groupby("kode_bendungan")}

    df = pd.read_parquet(path_root("data", "processed",
                                   "gabungan_periode.parquet"))
    dir_model = path_root(cfg["output"]["dir_model"])
    with open(os.path.join(dir_model, "skala_periode.json")) as f:
        skala = json.load(f)["skala"]
    model = {fmt: load_model(os.path.join(
        dir_model, f"lstm_tma_periode_{NAMA_FILE[fmt]}.keras"))
        for fmt in (FORMAT_15, FORMAT_10)}

    # cutoff = akhir Juli 2025 dan akhir Februari 2026 (skala periode)
    cutoffs = {"Agu-Des 2025": "2025-07-31", "Mar-Jul 2026": "2026-02-28"}
    detail = jalankan_backtest(df, model, skala, peta_musim, cutoffs,
                               m.get("input_tahun", 1))
    r = ringkas(detail)
    print(r.to_string(index=False))

    dir_pred = path_root(cfg["output"]["dir_prediksi"])
    os.makedirs(dir_pred, exist_ok=True)
    detail.to_parquet(os.path.join(dir_pred, "evaluasi_metode.parquet"),
                      index=False)
    with pd.ExcelWriter(os.path.join(dir_pred, "evaluasi_metode.xlsx")) as w:
        r.to_excel(w, sheet_name="Ringkasan", index=False)
        pd.DataFrame([{"metode": k, "deskripsi": v}
                      for k, v in METODE_INFO.items()]
                     ).to_excel(w, sheet_name="Metode", index=False)
    print(f"\nOK  evaluasi_metode.parquet ({len(detail):,} baris)")


if __name__ == "__main__":
    main()
