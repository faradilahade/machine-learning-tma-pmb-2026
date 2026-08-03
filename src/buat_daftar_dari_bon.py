"""Regenerasi daftar_bendungan.csv dari sheet BON_A di data_bon_a_bon_b.xlsx.

Jalankan setiap kali file BON diperbarui, agar daftar bendungan yang
dimodelkan selalu konsisten dengan data BON (single source of truth).
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_config, path_root


def main():
    cfg = load_config()
    df = pd.read_excel(path_root(cfg["bon"]["file_bon"]), sheet_name="BON_A",
                       dtype={"kode_bendungan": str})
    daftar = df[["kode_bendungan", "nama_bendungan"]].dropna().drop_duplicates()
    daftar["nama_bendungan"] = daftar["nama_bendungan"].str.strip()
    out = path_root(cfg["bon"]["file_daftar"])
    daftar.to_csv(out, index=False)
    print(f"OK  {out}  ({len(daftar)} bendungan)")


if __name__ == "__main__":
    main()
