"""Ekstrak referensi Gongseng (ketersediaan, kebutuhan, elevasi kekeringan) dari workbook RTOW ke JSON kecil
agar dashboard app_gongseng.py cepat dimuat (tanpa membaca Excel besar). Jalankan ulang bila workbook berubah."""
import json
import pandas as pd

X = "data_bon/new_data_bon_a_bon_b_rtow_ketersediaan_kebutuhan_air.xlsx"
N = "GONGSENG"


def baris(sheet):
    d = pd.read_excel(X, sheet_name=sheet)
    r = d[d.nama_bendungan == N].iloc[0]
    return {c: (None if pd.isna(r[c]) else round(float(r[c]), 4)) for c in d.columns if len(str(c)) == 5 and "-" in str(c)}


ek = pd.read_excel(X, sheet_name="elevasi_kekeringan")
ek = ek[ek.nama_bendungan == N].iloc[0]
out = {"ketersediaan": baris("ketersediaan_air_10"), "kebutuhan": baris("kebutuhan_air_10"),
       "kekeringan": float(ek.elevasi_kekeringan_rtow), "dasar": float(ek.elevasi_dasar_waduk_rtow),
       "intake": float(ek.elevasi_dasar_intake_sinbad),
       "bon_a": baris("bon_a_10"), "bon_b": baris("bon_b_10"), "rtow": baris("rtow_10")}
json.dump(out, open("data_gongseng/ref_gongseng.json", "w"), indent=1)
