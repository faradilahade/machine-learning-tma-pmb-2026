"""Generator data dummy harian (mode demo tanpa akses DB).

Struktur output PERSIS sama dengan extract_data.py. Karakteristik tiap
bendungan ditambatkan ke data BON real-nya (rerata & amplitudo BON A)
agar level elevasi masuk akal; bendungan tanpa BON memakai profil default
deterministik dari kode-nya.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_config, path_root, baca_daftar_bendungan, baca_bon


def karakter_bendungan(kode, bon):
    b = bon[(bon["kode_bendungan"] == kode) & bon["bon_a"].notna()]
    if len(b):
        rerata = float(b["bon_a"].mean()) + 0.8          # sedikit di atas BON A
        amp = max(float(b["bon_a"].max() - b["bon_a"].min()) / 2, 0.5)
    else:  # tanpa BON: profil deterministik dari kode
        seed = int("".join(filter(str.isdigit, kode)) or 0)
        rerata = 100.0 + (seed % 150)
        amp = 1.5 + (seed % 5) * 0.6
    kapasitas = 20 + amp * 120                            # juta m3 (kasar)
    return rerata, amp, kapasitas


def deret_tma(rng, tanggal, rerata, amp):
    doy = tanggal.dayofyear.values
    tahun = tanggal.year.values
    musim = amp * np.cos((doy - 60) * 2 * np.pi / 365.25)
    kering = np.zeros(len(tanggal))
    kering[tahun == 2023] = -0.4 * amp
    m26 = tahun == 2026
    kering[m26] = -0.5 * amp * np.clip((doy[m26] - 90) / 180, 0, 1)
    acak = np.cumsum(rng.normal(0, 0.02, len(tanggal)))
    acak -= np.linspace(acak[0], acak[-1], len(acak))
    return rerata + musim + kering + acak + rng.normal(0, 0.03, len(tanggal))


def main():
    cfg = load_config()
    daftar = baca_daftar_bendungan(cfg)
    bon = baca_bon(cfg)
    tanggal = pd.date_range(cfg["periode"]["awal"], cfg["periode"]["akhir"], freq="D")
    n = len(tanggal)

    tma_all, out_all, in_all, curve_all = [], [], [], []
    for _, r in daftar.iterrows():
        kode = r["kode_bendungan"]
        rng = np.random.default_rng(abs(hash(kode)) % (2**32))
        rerata, amp, kap = karakter_bendungan(kode, bon)
        tma = deret_tma(rng, tanggal, rerata, amp)

        # suntik anomali & gap utk menguji QC
        tma_kotor = tma.copy()
        i_spike = rng.choice(n, 12, replace=False)
        tma_kotor[i_spike] += rng.choice([-1, 1], 12) * rng.uniform(4, 10, 12)
        tma_kotor[rng.choice(n, 35, replace=False)] = np.nan
        g0 = rng.integers(200, n - 300)
        tma_kotor[g0:g0 + 12] = np.nan
        tma_all.append(pd.DataFrame(
            {"kode_bendungan": kode, "tanggal": tanggal, "tma": tma_kotor}))

        doy = tanggal.dayofyear.values
        outflow = np.clip(20 + 15 * np.sin((doy - 150) * 2 * np.pi / 365.25)
                          + rng.normal(0, 3, n), 2, None) * (kap / 1000)
        out_all.append(pd.DataFrame(
            {"kode_bendungan": kode, "tanggal": tanggal, "outflow": outflow}))
        in_all.append(pd.DataFrame(
            {"kode_bendungan": kode, "tanggal": tanggal,
             "inflow_tercatat": np.clip(outflow + rng.normal(0, 5, n), 0, None)}))

        elev = np.linspace(rerata - amp - 6, rerata + amp + 4, 40)
        vol = kap * ((elev - elev[0]) / (elev[-1] - elev[0])) ** 1.8
        curve_all.append(pd.DataFrame(
            {"kode_bendungan": kode, "elevasi": elev, "volume": vol}))

    out_dir = path_root("data", "raw")
    os.makedirs(out_dir, exist_ok=True)
    for nama, frames in [("tma", tma_all), ("outflow", out_all),
                         ("inflow", in_all), ("storagecurve", curve_all)]:
        df = pd.concat(frames, ignore_index=True)
        fp = os.path.join(out_dir, f"{nama}_harian.parquet")
        df.to_parquet(fp, index=False)
        print(f"OK  {fp}  ({len(df):,} baris)")


if __name__ == "__main__":
    main()
