"""Turunan tabel laporan + ekspor grafik JPG dan Excel untuk dashboard prediksi TMA.

Dipakai oleh app.py (Streamlit). Semua fungsi murni pandas/matplotlib supaya
bisa dipanggil juga dari skrip batch tanpa Streamlit.
"""
import io
import os
import sys
import zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WARNA = {"navy": "#0F2A47", "biru": "#1C5D8C", "teal": "#2A9D8F",
         "amber": "#E9C46A", "coral": "#E76F51",
         "hijau": "#2E7D32", "merah": "#C62828", "abu": "#9E9E9E"}

BULAN_ID = {1: "Januari", 2: "Februari", 3: "Maret", 4: "April", 5: "Mei",
            6: "Juni", 7: "Juli", 8: "Agustus", 9: "September", 10: "Oktober",
            11: "November", 12: "Desember"}

# Tiga kategori utama rekapitulasi status BON (per periode):
KRITIS = "Di bawah BON B"                 # TMA < BON B
WASPADA = "Di antara BON A dan BON B"     # BON B <= TMA < BON A
NORMAL = "Di atas BON A"                  # TMA >= BON A
TANPA_BON = "Tanpa Data BON"
# alias eksplisit
BAWAH_B, ANTARA_AB, ATAS_A = KRITIS, WASPADA, NORMAL

BOBOT_STATUS = {KRITIS: 3, WASPADA: 2, NORMAL: 1, TANPA_BON: 0}
TAK_DIKETAHUI = "(TIDAK DIKETAHUI)"


# ------------------------------------------------------------------ wilayah
def gabung_wilayah(df: pd.DataFrame, daftar: pd.DataFrame) -> pd.DataFrame:
    """Tempelkan nama_balai & nama_pulau dari daftar_bendungan.csv ke df."""
    kolom = ["kode_bendungan"]
    for c in ("nama_balai", "nama_pulau"):
        if c in daftar.columns:
            kolom.append(c)
    out = df.merge(daftar[kolom].drop_duplicates("kode_bendungan"),
                   on="kode_bendungan", how="left")
    for c in ("nama_balai", "nama_pulau"):
        if c not in out.columns:
            out[c] = TAK_DIKETAHUI
        out[c] = out[c].fillna(TAK_DIKETAHUI).astype(str).str.strip().str.upper()
    return out


def dibawah_bon(s: pd.Series) -> pd.Series:
    """True bila status termasuk pelanggaran BON (bukan Normal / Tanpa Data)."""
    return ~s.isin([NORMAL, TANPA_BON])


# ------------------------------------------------------------------ narasi
def _sebut(baris) -> str:
    """Sebutan waktu satu baris: 'periode 08-01 (Agustus)' bila ada label."""
    t = baris["tanggal"]
    if "periode" in baris and pd.notna(baris.get("periode")):
        return f"periode {baris['periode']} ({BULAN_ID[t.month]} {t.year})"
    return f"{BULAN_ID[t.month]} {t.year}"


def narasi_bendungan(g: pd.DataFrame) -> str:
    """Penjelasan otomatis satu bendungan (dipakai di app & sheet Excel)."""
    g = g.sort_values("tanggal")
    real = g[g["jenis"] == "realisasi"]
    pred = g[g["jenis"] == "prediksi"]
    nama = g["nama_bendungan"].iloc[0]
    satuan = "periode" if "periode" in g.columns else "bulan"
    if pred.empty:
        return (f"Bendungan {nama} belum memiliki hasil prediksi — "
                f"kemungkinan histori TMA kurang dari 1 tahun atau data "
                f"tidak ditemukan di database.")

    bagian = []
    if not real.empty:
        bagian.append(f"TMA realisasi terakhir ({_sebut(real.iloc[-1])}) "
                      f"{real['tma'].iloc[-1]:.2f} mdpl")

    p_awal, p_akhir = pred.iloc[0], pred.iloc[-1]
    tren = p_akhir["tma"] - p_awal["tma"]
    arah = "naik" if tren > 0.05 else ("turun" if tren < -0.05 else "relatif datar")
    bagian.append(
        f"prediksi {_sebut(p_awal)} {p_awal['tma']:.2f} mdpl → "
        f"{_sebut(p_akhir)} {p_akhir['tma']:.2f} mdpl ({arah} {abs(tren):.2f} m)")
    p_min = pred.loc[pred["tma"].idxmin()]
    bagian.append(f"TMA minimum prediksi {p_min['tma']:.2f} mdpl pada {_sebut(p_min)}")

    if "rtow" in pred.columns and pred["rtow"].notna().any():
        di_bawah_rtow = pred[pred["tma"] < pred["rtow"]]
        if len(di_bawah_rtow):
            bagian.append(
                f"prediksi di bawah elevasi rencana RTOW pada "
                f"{len(di_bawah_rtow)} dari {len(pred)} {satuan} (mulai "
                f"{_sebut(di_bawah_rtow.iloc[0])})")
        else:
            bagian.append("prediksi berada di atas/ sesuai rencana RTOW "
                          "sepanjang horizon")

    kritis = pred[pred["status_bon"] == BAWAH_B]
    antara = pred[pred["status_bon"] == ANTARA_AB]
    tanpa = pred[pred["status_bon"] == TANPA_BON]

    if len(tanpa) == len(pred):
        akhir = ("Status BON tidak dapat dinilai karena bendungan ini belum "
                 "memiliki nilai BON A/BON B pada file referensi.")
    elif not kritis.empty:
        b = kritis.iloc[0]
        akhir = (f"DI BAWAH BON B: prediksi menembus BON B mulai {_sebut(b)} "
                 f"(TMA {b['tma']:.2f} mdpl vs BON B {b['bon_b']:.2f} mdpl, "
                 f"defisit {b['bon_b'] - b['tma']:.2f} m) selama {len(kritis)} "
                 f"{satuan}. Perlu penyesuaian pola operasi/alokasi air.")
    elif not antara.empty:
        b = antara.iloc[0]
        akhir = (f"DI ANTARA BON A DAN BON B: prediksi turun di bawah BON A "
                 f"mulai {_sebut(b)} (TMA {b['tma']:.2f} mdpl vs BON A "
                 f"{b['bon_a']:.2f} mdpl, selisih {b['bon_a'] - b['tma']:.2f} m) "
                 f"selama {len(antara)} {satuan}, namun masih di atas BON B.")
    else:
        akhir = (f"DI ATAS BON A: seluruh {satuan} prediksi berada di atas "
                 f"BON A — pasokan air pada musim kering diperkirakan aman.")
    return "; ".join(bagian) + ". " + akhir


def status_terburuk(s: pd.Series) -> str:
    urut = s.map(BOBOT_STATUS).fillna(-1)
    if urut.empty:
        return TANPA_BON
    return s.iloc[int(np.argmax(urut.values))]


# ------------------------------------------------------------------ tabel
def rekap_bendungan(df: pd.DataFrame) -> pd.DataFrame:
    """Satu baris per bendungan: kondisi terakhir, ringkasan prediksi, narasi."""
    baris = []
    for (kode, nama), g in df.groupby(["kode_bendungan", "nama_bendungan"],
                                      sort=False):
        g = g.sort_values("tanggal")
        real = g[g["jenis"] == "realisasi"]
        pred = g[g["jenis"] == "prediksi"]
        kritis = pred[pred["status_bon"] == BAWAH_B]
        antara = pred[pred["status_bon"] == ANTARA_AB]
        atas = pred[pred["status_bon"] == ATAS_A]
        pelanggaran = pred[dibawah_bon(pred["status_bon"])]
        p_min = pred.loc[pred["tma"].idxmin()] if not pred.empty else None
        agu = pred[pred["tanggal"].dt.month == 8]
        baris.append({
            "kode_bendungan": kode,
            "nama_bendungan": nama,
            "nama_balai": g["nama_balai"].iloc[0] if "nama_balai" in g else TAK_DIKETAHUI,
            "nama_pulau": g["nama_pulau"].iloc[0] if "nama_pulau" in g else TAK_DIKETAHUI,
            "format_periode": (g["format_periode"].iloc[0]
                               if "format_periode" in g.columns else "-"),
            "tma_realisasi_terakhir": round(real["tma"].iloc[-1], 2) if not real.empty else np.nan,
            "tma_prediksi_agustus": round(agu["tma"].iloc[0], 2) if not agu.empty else np.nan,
            "tma_min_prediksi": round(pred["tma"].min(), 2) if not pred.empty else np.nan,
            "waktu_tma_min": _sebut(p_min) if p_min is not None else "",
            "jumlah_periode_prediksi": len(pred),
            "periode_diatas_bon_a": len(atas),
            "periode_antara_a_b": len(antara),
            "periode_dibawah_bon_b": len(kritis),
            "pertama_dibawah_bon": (_sebut(pelanggaran.iloc[0])
                                    if not pelanggaran.empty else "-"),
            "status_terburuk": status_terburuk(pred["status_bon"]) if not pred.empty else TANPA_BON,
            "bon_terisi_otomatis": bool(g.get("bon_terisi_otomatis", pd.Series([False])).any()),
            "penjelasan": narasi_bendungan(g),
        })
    out = pd.DataFrame(baris)
    if out.empty:
        return out
    out["_bobot"] = out["status_terburuk"].map(BOBOT_STATUS)
    return (out.sort_values(["_bobot", "periode_dibawah_bon_b", "nama_bendungan"],
                            ascending=[False, False, True])
               .drop(columns="_bobot").reset_index(drop=True))


def tabel_pemantauan_bulan(df: pd.DataFrame, bulan: int = 8) -> pd.DataFrame:
    """Tabel pemantauan satu bulan prediksi (default Agustus) untuk semua bendungan.

    Kolom selisih bertanda positif = masih di atas ambang BON.
    """
    pred = df[(df["jenis"] == "prediksi") & (df["tanggal"].dt.month == bulan)].copy()
    if pred.empty:
        return pred
    # skala periode: ambil PERIODE TERBURUK (TMA terendah) per bendungan
    # pada bulan tinjauan sebagai representasi pemantauan bulan itu
    pred = (pred.sort_values("tma")
                .drop_duplicates("kode_bendungan", keep="first"))
    real_akhir = (df[df["jenis"] == "realisasi"].sort_values("tanggal")
                    .groupby("kode_bendungan")
                    .agg(tma_realisasi_terakhir=("tma", "last"),
                         bulan_realisasi_terakhir=("tanggal", "last")))
    out = pred.merge(real_akhir, on="kode_bendungan", how="left")
    out["bulan_realisasi_terakhir"] = (out["bulan_realisasi_terakhir"]
                                       .dt.month.map(BULAN_ID).fillna("-"))
    out["perubahan_dari_realisasi_m"] = (out["tma"] - out["tma_realisasi_terakhir"]).round(2)
    out["selisih_ke_bon_a_m"] = (out["tma"] - out["bon_a"]).round(2)
    out["selisih_ke_bon_b_m"] = (out["tma"] - out["bon_b"]).round(2)
    if "rtow" in out.columns:
        out["selisih_ke_rtow_m"] = (out["tma"] - out["rtow"]).round(2)
    out = out.rename(columns={"tma": "tma_prediksi",
                              "periode": "periode_terburuk"})
    out["tma_prediksi"] = out["tma_prediksi"].round(2)
    kolom = ["kode_bendungan", "nama_bendungan", "nama_balai", "nama_pulau",
             "periode_terburuk", "format_periode",
             "bulan_realisasi_terakhir", "tma_realisasi_terakhir", "tma_prediksi",
             "perubahan_dari_realisasi_m", "bon_a", "bon_b", "rtow",
             "selisih_ke_bon_a_m", "selisih_ke_bon_b_m", "selisih_ke_rtow_m",
             "status_bon", "bon_terisi_otomatis"]
    kolom = [c for c in kolom if c in out.columns]
    out = out[kolom]
    out["_bobot"] = out["status_bon"].map(BOBOT_STATUS)
    return (out.sort_values(["_bobot", "selisih_ke_bon_b_m", "nama_bendungan"],
                            ascending=[False, True, True])
               .drop(columns="_bobot").reset_index(drop=True))


def rekap_per_balai(df: pd.DataFrame, bulan: int | None = None) -> pd.DataFrame:
    """Jumlah bendungan per status, dikelompokkan per pulau & balai."""
    pred = df[df["jenis"] == "prediksi"]
    if bulan:
        pred = pred[pred["tanggal"].dt.month == bulan]
    if pred.empty:
        return pd.DataFrame()
    per_bdg = (pred.groupby(["nama_pulau", "nama_balai",
                             "kode_bendungan", "nama_bendungan"])["status_bon"]
                   .apply(status_terburuk).reset_index())
    rekap = (per_bdg.assign(n=1)
             .pivot_table(index=["nama_pulau", "nama_balai"], columns="status_bon",
                          values="n", aggfunc="sum", fill_value=0)
             .reset_index())
    for s in (KRITIS, WASPADA, NORMAL, TANPA_BON):
        if s not in rekap.columns:
            rekap[s] = 0
    rekap["jumlah_bendungan"] = rekap[[KRITIS, WASPADA, NORMAL, TANPA_BON]].sum(axis=1)
    rekap = rekap[["nama_pulau", "nama_balai", "jumlah_bendungan",
                   KRITIS, WASPADA, NORMAL, TANPA_BON]]
    return rekap.sort_values([KRITIS, WASPADA, "jumlah_bendungan"],
                            ascending=False).reset_index(drop=True)


def daftar_belum_cocok(daftar: pd.DataFrame, df: pd.DataFrame,
                       hist: pd.DataFrame | None = None,
                       rekap_qc: pd.DataFrame | None = None) -> pd.DataFrame:
    """Bendungan yang datanya belum cocok / tidak lengkap antar sumber.

    Kategori:
      1. TIDAK ADA DI DATABASE   - ada di daftar/BON, tapi tak ada histori TMA
         (dipecah: data memang tidak ada vs seluruh nilainya ditolak QC bila
          rekap_qc.xlsx tersedia)
      2. TIDAK ADA PREDIKSI      - ada histori, tapi hasil prediksi kosong
      3. TIDAK ADA DI DAFTAR BON - kode muncul di data TMA/prediksi, tapi tak
                                   terdaftar di daftar_bendungan.csv
      4. TANPA NILAI BON A/B     - diprediksi, tapi BON belum diisi -> status
                                   tidak dapat dinilai
      5. BON TERISI OTOMATIS     - sebagian bulan BON kosong lalu diinterpolasi
    """
    kolom_wil = [c for c in ("nama_balai", "nama_pulau") if c in daftar.columns]
    ref = (daftar[["kode_bendungan", "nama_bendungan"] + kolom_wil]
           .drop_duplicates("kode_bendungan").copy())
    for c in kolom_wil:
        ref[c] = ref[c].fillna(TAK_DIKETAHUI).astype(str).str.strip().str.upper()
    for c in ("nama_balai", "nama_pulau"):
        if c not in ref.columns:
            ref[c] = TAK_DIKETAHUI

    kode_hist = set(hist["kode_bendungan"].unique()) if hist is not None and not hist.empty else set()
    pred = df[df["jenis"] == "prediksi"]
    kode_pred = set(pred["kode_bendungan"].unique())
    kode_ref = set(ref["kode_bendungan"])

    baris = []

    def tambah(kode, kategori, keterangan, tindakan):
        r = ref[ref["kode_bendungan"] == kode]
        if not r.empty:
            nama = r["nama_bendungan"].iloc[0]
            balai, pulau = r["nama_balai"].iloc[0], r["nama_pulau"].iloc[0]
        else:
            g = df[df["kode_bendungan"] == kode]
            nama = g["nama_bendungan"].iloc[0] if not g.empty else TAK_DIKETAHUI
            balai = pulau = TAK_DIKETAHUI
        baris.append({"kode_bendungan": kode, "nama_bendungan": nama,
                      "nama_balai": balai, "nama_pulau": pulau,
                      "kategori_masalah": kategori, "keterangan": keterangan,
                      "tindak_lanjut": tindakan})

    # bendungan yang seluruh nilainya ditolak QC (dari rekap_qc.xlsx)
    ditolak_qc = {}
    if rekap_qc is not None and not rekap_qc.empty and "lolos_qc" in rekap_qc.columns:
        gagal = rekap_qc[~rekap_qc["lolos_qc"].fillna(True).astype(bool)]
        ditolak_qc = dict(zip(gagal["kode_bendungan"],
                              gagal.get("alasan", pd.Series(dtype=str))
                              .reindex(gagal.index).fillna("")))

    # 1. terdaftar tapi tidak ada histori TMA di database
    if kode_hist:
        for kode in sorted(kode_ref - kode_hist):
            if kode in ditolak_qc:
                tambah(kode, "1b. Data ditolak QC",
                       f"Data TMA ada di database tetapi {ditolak_qc[kode] or 'seluruh nilainya ditolak QC'} "
                       f"— nilai TMA berada di luar rentang elevasi storage "
                       f"curve bendungan ini, indikasi id database atau satuan "
                       f"tidak cocok.",
                       "Verifikasi pemetaan kode BDxxx → id database (query "
                       "SELECT DISTINCT id FROM rawdata.sinbad_bendungan_tma) "
                       "dan bandingkan rentang TMA dengan storage curve-nya.")
            else:
                tambah(kode, "1a. Tidak ada di database",
                       "Kode/nama bendungan tidak ditemukan pada data TMA hasil "
                       "ekstraksi SINBAD (rawdata.sinbad_bendungan_tma).",
                       "Cek pemetaan kode BDxxx → id database (kolom id_db pada "
                       "daftar_bendungan.csv) atau pastikan bendungan sudah "
                       "terintegrasi/mengirim data ke SINBAD.")

    # 2. ada histori, tetapi tidak menghasilkan prediksi
    for kode in sorted((kode_hist & kode_ref) - kode_pred):
        n = int((hist["kode_bendungan"] == kode).sum()) if kode_hist else 0
        tambah(kode, "2. Tidak ada hasil prediksi",
               f"Ada data historis ({n} bulan) tetapi tidak ada baris prediksi — "
               f"kemungkinan histori < 12 bulan (panjang sekuens input model).",
               "Tambah panjang histori data TMA atau turunkan parameter "
               "model.input_bulan pada config.yaml.")

    # 3. muncul di data, tetapi tidak terdaftar
    for kode in sorted((kode_pred | kode_hist) - kode_ref):
        tambah(kode, "3. Tidak ada di daftar BON",
               "Kode muncul di data TMA/hasil prediksi tetapi tidak terdaftar "
               "pada daftar_bendungan.csv / file BON.",
               "Tambahkan bendungan ke data_bon_a_bon_b.xlsx lalu jalankan "
               "src/buat_daftar_dari_bon.py.")

    # 4. tanpa nilai BON sama sekali
    tanpa = (pred.groupby("kode_bendungan")["status_bon"]
                 .apply(lambda s: (s == TANPA_BON).all()))
    for kode in sorted(tanpa[tanpa].index):
        tambah(kode, "4. Tanpa nilai BON A/B",
               "Diprediksi normal, tetapi nilai BON A/BON B belum diisi pada "
               "file referensi sehingga status tidak dapat dinilai.",
               "Isi baris bendungan ini pada sheet BON_A dan BON_B di "
               "data_bon/data_bon_a_bon_b.xlsx.")

    # 5. BON hasil interpolasi otomatis
    if "bon_terisi_otomatis" in df.columns:
        otomatis = df.groupby("kode_bendungan")["bon_terisi_otomatis"].any()
        for kode in sorted(otomatis[otomatis].index):
            if kode in tanpa[tanpa].index:
                continue
            tambah(kode, "5. BON terisi otomatis (interpolasi)",
                   "Sebagian bulan BON kosong dan diisi interpolasi linear "
                   "antar-bulan — status BON bulan tersebut bersifat perkiraan.",
                   "Lengkapi nilai BON bulanan resmi agar penilaian status akurat.")

    out = pd.DataFrame(baris)
    if out.empty:
        return out
    return out.sort_values(["kategori_masalah", "nama_pulau", "nama_balai",
                            "nama_bendungan"]).reset_index(drop=True)


# ------------------------------------------------------------------ neraca air
def tabel_neraca(neraca: pd.DataFrame) -> pd.DataFrame:
    """Format tampil neraca air per bulan (juta m3) + status surplus/defisit."""
    if neraca is None or neraca.empty:
        return pd.DataFrame()
    t = neraca.copy()
    t["nama_bulan"] = t["bulan"].map(BULAN_ID)
    for c in ("ketersediaan_m3", "kebutuhan_m3", "neraca_m3"):
        t[c.replace("_m3", "_juta_m3")] = (t[c] / 1e6).round(3)
    return t[["kode_bendungan", "nama_bendungan", "bulan", "nama_bulan",
              "ketersediaan_juta_m3", "kebutuhan_juta_m3", "neraca_juta_m3",
              "status_neraca"]]


def rekap_neraca(neraca: pd.DataFrame) -> pd.DataFrame:
    """Satu baris per bendungan: ringkasan defisit/surplus setahun + narasi."""
    if neraca is None or neraca.empty:
        return pd.DataFrame()
    baris = []
    for (kode, nama), n in neraca.groupby(["kode_bendungan", "nama_bendungan"],
                                          sort=False):
        n = n.sort_values("bulan")
        defisit = n[n["status_neraca"] == "Defisit"]
        kering = n[n["bulan"] >= 8]           # musim kering = horizon prediksi
        defisit_kering = kering[kering["status_neraca"] == "Defisit"]
        terparah = (defisit.loc[defisit["neraca_m3"].idxmin()]
                    if not defisit.empty else None)
        baris.append({
            "kode_bendungan": kode, "nama_bendungan": nama,
            "nama_balai": n["nama_balai"].iloc[0] if "nama_balai" in n else TAK_DIKETAHUI,
            "nama_pulau": n["nama_pulau"].iloc[0] if "nama_pulau" in n else TAK_DIKETAHUI,
            "bulan_defisit_setahun": len(defisit),
            "bulan_defisit_agu_des": len(defisit_kering),
            "defisit_terbesar_juta_m3": (round(-terparah["neraca_m3"] / 1e6, 3)
                                         if terparah is not None else 0.0),
            "bulan_defisit_terbesar": (BULAN_ID[int(terparah["bulan"])]
                                       if terparah is not None else "-"),
            "total_defisit_juta_m3": round(-defisit["neraca_m3"].sum() / 1e6, 3),
            "total_kebutuhan_juta_m3": round(n["kebutuhan_m3"].sum() / 1e6, 3),
            "total_ketersediaan_juta_m3": round(n["ketersediaan_m3"].sum() / 1e6, 3),
            "penjelasan_neraca": narasi_neraca(n),
        })
    out = pd.DataFrame(baris)
    return (out.sort_values(["bulan_defisit_agu_des", "defisit_terbesar_juta_m3"],
                            ascending=False).reset_index(drop=True))


def narasi_neraca(n: pd.DataFrame) -> str:
    """Analisis otomatis kebutuhan vs ketersediaan air satu bendungan."""
    n = n.sort_values("bulan")
    nama = n["nama_bendungan"].iloc[0]
    ada = n[n["status_neraca"] != "Tanpa Data"]
    if ada.empty:
        return (f"Bendungan {nama} belum memiliki data ketersediaan/kebutuhan "
                f"air pada file referensi.")
    defisit = ada[ada["status_neraca"] == "Defisit"]
    if defisit.empty:
        tot = ada["neraca_m3"].sum() / 1e6
        return (f"Ketersediaan air Bendungan {nama} mencukupi kebutuhan pada "
                f"seluruh {len(ada)} bulan berdata (surplus kumulatif "
                f"{tot:,.2f} juta m³).")
    bln = [BULAN_ID[int(b)] for b in defisit["bulan"]]
    terparah = defisit.loc[defisit["neraca_m3"].idxmin()]
    tot_def = -defisit["neraca_m3"].sum() / 1e6
    pakai = (f"kebutuhan {terparah['kebutuhan_m3'] / 1e6:,.2f} vs ketersediaan "
             f"{terparah['ketersediaan_m3'] / 1e6:,.2f} juta m³")
    return (f"Bendungan {nama} DEFISIT air pada {len(defisit)} dari {len(ada)} "
            f"bulan ({', '.join(bln)}); terbesar {-terparah['neraca_m3'] / 1e6:,.2f} "
            f"juta m³ pada {BULAN_ID[int(terparah['bulan'])]} ({pakai}); total "
            f"defisit setahun {tot_def:,.2f} juta m³. Perlu prioritas alokasi "
            f"dan penyesuaian pola tanam/pemberian air pada bulan tersebut.")


# ------------------------------------------------------------------ grafik
def fig_bendungan(g: pd.DataFrame, hist: pd.DataFrame | None = None,
                  tampil_hist: bool = False, dpi: int = 120):
    """Grafik sandingan (matplotlib) satu bendungan — sumbu-x label PERIODE
    'MM-PP' Januari-Desember sesuai format 10/15-harian bendungan tsb."""
    from src.utils import daftar_periode
    g = g.sort_values("tanggal")
    nama = g["nama_bendungan"].iloc[0]
    kode = g["kode_bendungan"].iloc[0]
    balai = g["nama_balai"].iloc[0] if "nama_balai" in g.columns else ""
    fmt = (g["format_periode"].iloc[0]
           if "format_periode" in g.columns else "15 Harian")
    urut = daftar_periode(fmt, 1, 12)
    pos = {p: i for i, p in enumerate(urut)}
    d = g.drop_duplicates("periode").set_index("periode").reindex(urut)
    real = g[g["jenis"] == "realisasi"]
    pred = g[g["jenis"] == "prediksi"]
    x = np.arange(len(urut))

    fig, ax = plt.subplots(figsize=(12.5, 6.0), dpi=dpi)

    if tampil_hist and hist is not None and not hist.empty:
        h = hist[(hist["kode_bendungan"] == kode)
                 & (hist["tanggal"] < "2026-01-01")].copy()
        if not h.empty and "periode" in h.columns:
            pertama = True
            for thn, ht in h.groupby(h["tanggal"].dt.year):
                s = ht.set_index("periode")["tma"].reindex(urut)
                ax.plot(x, s.values, color="lightgrey", lw=1.0, zorder=1,
                        label="Historis per tahun (2018–2025)" if pertama else None)
                pertama = False

    for kol, warna, gaya, label in [("rtow", WARNA["biru"], ":", "RTOW"),
                                    ("bon_a", WARNA["amber"], "-.", "BON A"),
                                    ("bon_b", WARNA["coral"], "-.", "BON B")]:
        if kol in d.columns and d[kol].notna().any():
            ax.plot(x, d[kol].values, color=warna, lw=2, ls=gaya,
                    label=label, zorder=2)

    if not real.empty:
        ax.plot([pos[p] for p in real["periode"]], real["tma"],
                color=WARNA["hijau"], lw=2.5, marker="o", ms=4,
                label="TMA Realisasi 2026", zorder=4)
    if not pred.empty:
        xp = ([pos[real["periode"].iloc[-1]]] if not real.empty else []) + \
             [pos[p] for p in pred["periode"]]
        yp = ([real["tma"].iloc[-1]] if not real.empty else []) + list(pred["tma"])
        ax.plot(xp, yp, color=WARNA["merah"], lw=2.5, ls="--", marker="s", ms=4,
                label="TMA Prediksi LSTM (Agu–Des 2026)", zorder=4)
        for p, v, s in zip(pred["periode"], pred["tma"], pred["status_bon"]):
            if s == BAWAH_B:
                ax.scatter([pos[p]], [v], s=150, facecolors="none",
                           edgecolors=WARNA["merah"], lw=2, zorder=5)

    y0, y1 = ax.get_ylim()
    if "bon_b" in d.columns and d["bon_b"].notna().any():
        ax.fill_between(x, d["bon_b"].values.astype(float), y0,
                        color=WARNA["coral"], alpha=0.07, zorder=0,
                        label="Zona di bawah BON B")
        ax.set_ylim(y0, y1)

    if not real.empty and not pred.empty:
        batas = pos[real["periode"].iloc[-1]] + 0.5
        ax.axvline(batas, color="grey", ls=":", lw=1.2)
        ax.annotate("realisasi | prediksi  ", xy=(batas, y1),
                    xytext=(-4, -6), textcoords="offset points",
                    fontsize=8, color="grey", va="top", ha="right")

    judul = (f"Sandingan TMA Realisasi vs Prediksi 2026 (periode {fmt}) — "
             f"Bendungan {nama} ({kode})")
    ax.set_title(judul, fontsize=12.5, fontweight="bold", color=WARNA["navy"],
                 pad=20 if balai else 10)
    if balai:
        sub = balai + (f" · {g['nama_pulau'].iloc[0]}" if "nama_pulau" in g.columns else "")
        ax.text(0.5, 1.012, sub, transform=ax.transAxes, ha="center",
                fontsize=9, color=WARNA["biru"])
    ax.set_ylabel("TMA (mdpl)")
    ax.set_xticks(x)
    ax.set_xticklabels(urut, rotation=90, fontsize=6.5 if len(urut) > 24 else 7.5)
    ax.set_xlim(-0.5, len(urut) - 0.5)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8.5)

    status = status_terburuk(pred["status_bon"]) if not pred.empty else TANPA_BON
    warna_status = {KRITIS: WARNA["merah"], WASPADA: WARNA["amber"],
                    NORMAL: WARNA["hijau"], TANPA_BON: WARNA["abu"]}[status]
    fig.text(0.01, 0.015, f"Status prediksi: {status}", fontsize=9,
             color=warna_status, fontweight="bold")
    fig.text(0.99, 0.015,
             "Model LSTM per periode (1 tahun → Agu–Des) + sifat musim · "
             "Subdit OP Bendungan dan Danau, Dit. Bina OP",
             fontsize=7.5, color="grey", ha="right")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def status_per_bulan(pred: pd.DataFrame) -> pd.DataFrame:
    """Status TERBURUK per bendungan per bulan dari baris per periode.

    Return: [kode_bendungan, nama_bendungan, bulan, status_bon] — dipakai
    agregasi lintas-bendungan (format 10/15-harian jadi sebanding).
    """
    if pred.empty:
        return pd.DataFrame(columns=["kode_bendungan", "nama_bendungan",
                                     "bulan", "status_bon"])
    d = pred.copy()
    d["bulan"] = d["tanggal"].dt.month
    return (d.groupby(["kode_bendungan", "nama_bendungan", "bulan"])
             ["status_bon"].apply(status_terburuk).reset_index())


def fig_peringkat_bulan(tab: pd.DataFrame, bulan: int = 8, n: int = 25, dpi: int = 120):
    """Grafik batang peringkat selisih TMA prediksi terhadap BON B."""
    d = tab.dropna(subset=["selisih_ke_bon_b_m"]).copy()
    if d.empty:
        return None
    d = d.sort_values("selisih_ke_bon_b_m").head(n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(11.5, max(4.5, 0.32 * len(d) + 1.6)), dpi=dpi)
    warna = [WARNA["merah"] if v < 0 else (WARNA["amber"] if v < 1 else WARNA["teal"])
             for v in d["selisih_ke_bon_b_m"]]
    label = d["nama_bendungan"] + " (" + d["kode_bendungan"] + ")"
    ax.barh(label, d["selisih_ke_bon_b_m"], color=warna)
    ax.axvline(0, color=WARNA["navy"], lw=1.2)
    for y, v in zip(range(len(d)), d["selisih_ke_bon_b_m"]):
        ax.text(v + (0.06 if v >= 0 else -0.06), y, f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=8)
    ax.set_xlabel("Selisih TMA prediksi terhadap BON B (m) — negatif = di bawah BON B")
    ax.set_title(f"{n} Bendungan Paling Rawan — Prediksi {BULAN_ID[bulan]} 2026",
                 fontsize=13, fontweight="bold", color=WARNA["navy"])
    ax.grid(axis="x", alpha=0.3)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ kesimpulan
def kesimpulan_nasional(df: pd.DataFrame, neraca: pd.DataFrame | None = None,
                        evaluasi: pd.DataFrame | None = None) -> list:
    """Kesimpulan otomatis seluruh data (list paragraf) — dipakai dashboard & PDF."""
    pred = df[df["jenis"] == "prediksi"]
    if pred.empty:
        return ["Belum ada hasil prediksi."]
    per_bdg = (pred.groupby(["kode_bendungan", "nama_bendungan"])["status_bon"]
                   .apply(status_terburuk))
    n = len(per_bdg)
    n_bawah = int((per_bdg == BAWAH_B).sum())
    n_antara = int((per_bdg == ANTARA_AB).sum())
    n_atas = int((per_bdg == ATAS_A).sum())
    par = []

    sbm = status_per_bulan(pred)
    puncak = (sbm[sbm["status_bon"] == BAWAH_B].groupby("bulan").size()
              if not sbm.empty else pd.Series(dtype=int))
    bln_puncak = (f"puncaknya {BULAN_ID[int(puncak.idxmax())]} "
                  f"({int(puncak.max())} bendungan)" if len(puncak) else "-")
    par.append(
        f"Dari {n} bendungan yang diprediksi (Agustus–Desember 2026, skala "
        f"periode 10/15-harian): {n_atas} bendungan ({n_atas / n:.0%}) tetap "
        f"DI ATAS BON A, {n_antara} ({n_antara / n:.0%}) turun ke ANTARA BON A "
        f"DAN BON B, dan {n_bawah} ({n_bawah / n:.0%}) diprediksi DI BAWAH "
        f"BON B minimal satu periode — {bln_puncak}.")

    kritis = pred[pred["status_bon"] == BAWAH_B]
    if not kritis.empty:
        terparah = (kritis.assign(defisit=kritis["bon_b"] - kritis["tma"])
                    .sort_values("defisit", ascending=False)
                    .drop_duplicates("kode_bendungan").head(3))
        daftar3 = "; ".join(
            f"{r['nama_bendungan']} (defisit {r['defisit']:.2f} m thd BON B "
            f"pada {_sebut(r)})" for _, r in terparah.iterrows())
        balai_terdampak = (kritis.groupby("nama_balai")["kode_bendungan"]
                           .nunique().sort_values(ascending=False)
                           if "nama_balai" in kritis.columns else pd.Series(dtype=int))
        par.append(
            f"Bendungan dengan pelanggaran BON B terdalam: {daftar3}. "
            + (f"Balai paling terdampak: {balai_terdampak.index[0]} "
               f"({int(balai_terdampak.iloc[0])} bendungan di bawah BON B)."
               if len(balai_terdampak) else ""))

    if "rtow" in pred.columns:
        n_rtow = int((pred.assign(b=pred["tma"] < pred["rtow"])
                      .groupby("kode_bendungan")["b"].any()).sum())
        par.append(
            f"Terhadap rencana operasi (RTOW), {n_rtow} dari {n} bendungan "
            f"diprediksi berada di bawah elevasi rencana pada satu periode "
            f"atau lebih — indikasi realisasi operasi lebih rendah dari "
            f"rencana tahunan.")

    if neraca is not None and not neraca.empty:
        nk = neraca[neraca["bulan"] >= 8]
        d_bdg = (nk.assign(d=nk["status_neraca"] == "Defisit")
                 .groupby("kode_bendungan")["d"].any())
        n_def = int(d_bdg.sum())
        tot_def = -nk.loc[nk["neraca_m3"] < 0, "neraca_m3"].sum() / 1e6
        terbesar = (nk[nk["neraca_m3"] < 0]
                    .sort_values("neraca_m3").drop_duplicates("kode_bendungan"))
        contoh = (f" Defisit bulanan terbesar: "
                  f"{terbesar.iloc[0]['nama_bendungan']} "
                  f"({-terbesar.iloc[0]['neraca_m3'] / 1e6:,.1f} juta m³ pada "
                  f"{BULAN_ID[int(terbesar.iloc[0]['bulan'])]})."
                  if not terbesar.empty else "")
        par.append(
            f"Kecukupan air musim kering (Agu–Des): {n_def} bendungan mengalami "
            f"DEFISIT (kebutuhan > ketersediaan) dengan total defisit "
            f"{tot_def:,.1f} juta m³.{contoh} Bendungan surplus tetap perlu "
            f"menjaga pola operasi agar tampungan aman hingga akhir tahun.")

    if evaluasi is not None and not evaluasi.empty:
        d = evaluasi.dropna(subset=["error"]).copy()
        d["abs"] = d["error"].abs()
        mae = d.groupby("metode")["abs"].mean().sort_values()
        if len(mae):
            par.append(
                f"Keandalan prediksi: model LSTM (dengan fitur sifat musim "
                f"Basah/Normal/Kering) memberi MAE {mae.iloc[0]:.2f} m pada "
                f"backtest — terbaik dari {len(mae)} metode yang diuji "
                f"(pembanding terbaik: {mae.index[1]} {mae.iloc[1]:.2f} m). "
                f"Gunakan prediksi sebagai indikasi dini, dikonfirmasi "
                f"pemantauan harian.")

    par.append(
        "Rekomendasi: prioritaskan bendungan pada Daftar Prioritas Perhatian "
        "(gabungan status BON, RTOW, dan defisit air); siapkan penyesuaian "
        "alokasi/pola tanam pada bulan-bulan defisit; perbarui pipeline tiap "
        "awal bulan agar prediksi bergerak maju.")
    return par


# ------------------------------------------------------------------ PDF
def pdf_laporan(df: pd.DataFrame, neraca: pd.DataFrame | None = None,
                evaluasi: pd.DataFrame | None = None,
                filter_teks: str = "Semua pulau & balai",
                waktu: str = "", n_grafik: int = 8) -> bytes:
    """Laporan PDF ringkas: kesimpulan, rekap status, prioritas, grafik
    sandingan bendungan paling rawan. Dibangun murni matplotlib (PdfPages)."""
    from matplotlib.backends.backend_pdf import PdfPages

    pred = df[df["jenis"] == "prediksi"]
    rek = rekap_bendungan(df)
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # --- halaman 1: judul + kesimpulan
        fig = plt.figure(figsize=(11.69, 8.27), dpi=130)   # A4 landscape
        fig.text(0.06, 0.90, "LAPORAN SIAGA KEKERINGAN — PREDIKSI TMA 2026",
                 fontsize=19, fontweight="bold", color=WARNA["navy"])
        fig.text(0.06, 0.855, "Pusat Monitoring Bendungan · Subdit OP Bendungan "
                              "dan Danau · Dit. Bina OP · Ditjen SDA",
                 fontsize=10, color=WARNA["biru"])
        fig.text(0.06, 0.825, f"Cakupan: {filter_teks}   ·   Dibuat: {waktu}",
                 fontsize=9, color="grey")
        fig.text(0.06, 0.77, "KESIMPULAN", fontsize=13, fontweight="bold",
                 color=WARNA["navy"])
        y = 0.735
        import textwrap
        for i, p in enumerate(kesimpulan_nasional(df, neraca, evaluasi), 1):
            for j, baris in enumerate(textwrap.wrap(f"{i}. {p}", width=125)):
                fig.text(0.06, y, baris, fontsize=9.5,
                         color="#222A35")
                y -= 0.028
            y -= 0.012
        pdf.savefig(fig)
        plt.close(fig)

        # --- halaman 2: rekap status per bulan + per balai
        sbm = status_per_bulan(pred)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.69, 8.27), dpi=130,
                                       gridspec_kw={"width_ratios": [1, 1.3]})
        if not sbm.empty:
            piv = (sbm.groupby(["bulan", "status_bon"]).size()
                   .unstack(fill_value=0)
                   .reindex(columns=[BAWAH_B, ANTARA_AB, ATAS_A, TANPA_BON],
                            fill_value=0))
            bawah = np.zeros(len(piv))
            for kol, warna in [(BAWAH_B, WARNA["merah"]),
                               (ANTARA_AB, WARNA["amber"]),
                               (ATAS_A, WARNA["hijau"]),
                               (TANPA_BON, WARNA["abu"])]:
                if piv[kol].sum() == 0:
                    continue
                ax1.bar([BULAN_ID[b][:3] for b in piv.index], piv[kol],
                        bottom=bawah, color=warna, label=kol)
                for xi, (v, b0) in enumerate(zip(piv[kol], bawah)):
                    if v:
                        ax1.text(xi, b0 + v / 2, str(int(v)), ha="center",
                                 va="center", fontsize=8, color="white")
                bawah += piv[kol].values
            ax1.set_title("Jumlah bendungan per status per bulan",
                          fontsize=11, fontweight="bold", color=WARNA["navy"])
            ax1.legend(fontsize=7, loc="lower left")
            ax1.grid(axis="y", alpha=0.3)

        balai = rekap_per_balai(df)
        if not balai.empty:
            b15 = balai.head(15).iloc[::-1]
            kiri = np.zeros(len(b15))
            for kol, warna in [(BAWAH_B, WARNA["merah"]),
                               (ANTARA_AB, WARNA["amber"]),
                               (ATAS_A, WARNA["hijau"]),
                               (TANPA_BON, WARNA["abu"])]:
                if kol not in b15.columns:
                    continue
                ax2.barh(b15["nama_balai"], b15[kol], left=kiri,
                         color=warna, label=kol)
                kiri += b15[kol].values
            ax2.set_title("15 balai dengan kondisi terberat",
                          fontsize=11, fontweight="bold", color=WARNA["navy"])
            ax2.tick_params(axis="y", labelsize=7)
            ax2.legend(fontsize=7, loc="lower right")
            ax2.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # --- halaman 3: tabel prioritas (di bawah BON B)
        if not rek.empty:
            prio = rek[rek["periode_dibawah_bon_b"] > 0].head(28)
            fig, ax = plt.subplots(figsize=(11.69, 8.27), dpi=130)
            ax.axis("off")
            ax.set_title("Bendungan dengan Prediksi DI BAWAH BON B (Agu–Des 2026)",
                         fontsize=13, fontweight="bold", color=WARNA["navy"],
                         loc="left")
            if prio.empty:
                ax.text(0, 0.9, "Tidak ada bendungan di bawah BON B.", fontsize=11)
            else:
                kolom = ["kode_bendungan", "nama_bendungan", "nama_balai",
                         "tma_min_prediksi", "periode_dibawah_bon_b",
                         "periode_antara_a_b", "pertama_dibawah_bon"]
                judul = ["Kode", "Nama", "Balai", "TMA Min", "Periode <BON B",
                         "Periode A–B", "Pertama < BON"]
                tabel = ax.table(cellText=prio[kolom].astype(str).values,
                                 colLabels=judul, loc="upper left",
                                 cellLoc="left", colLoc="left")
                tabel.auto_set_font_size(False)
                tabel.set_fontsize(7)
                tabel.scale(1, 1.25)
                for j in range(len(judul)):
                    sel = tabel[0, j]
                    sel.set_facecolor(WARNA["navy"])
                    sel.set_text_props(color="white", fontweight="bold")
            pdf.savefig(fig)
            plt.close(fig)

        # --- halaman grafik: bendungan paling rawan
        rawan = (rek[rek["periode_dibawah_bon_b"] > 0]
                 .head(n_grafik)["kode_bendungan"].tolist() if not rek.empty else [])
        for kode in rawan:
            g = df[df["kode_bendungan"] == kode]
            fig = fig_bendungan(g)
            pdf.savefig(fig)
            plt.close(fig)
    return buf.getvalue()


def fig_ke_jpg(fig, dpi: int = 120) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="jpg", dpi=dpi, facecolor="white",
                pil_kwargs={"quality": 92})
    plt.close(fig)
    return buf.getvalue()


def zip_grafik(df: pd.DataFrame, kodes, hist=None, tampil_hist=False,
               progress=None) -> bytes:
    """ZIP berisi grafik JPG untuk setiap kode bendungan pada `kodes`."""
    buf = io.BytesIO()
    kodes = list(dict.fromkeys(kodes))
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for i, kode in enumerate(kodes, 1):
            g = df[df["kode_bendungan"] == kode]
            if g.empty:
                continue
            nama = str(g["nama_bendungan"].iloc[0]).replace("/", "-")
            status = status_terburuk(g[g["jenis"] == "prediksi"]["status_bon"])
            tag = {KRITIS: "KRITIS", WASPADA: "WASPADA",
                   NORMAL: "NORMAL", TANPA_BON: "TANPA-BON"}[status]
            jpg = fig_ke_jpg(fig_bendungan(g, hist, tampil_hist))
            z.writestr(f"{tag}/{kode}_{nama}.jpg", jpg)
            if progress:
                progress(i, len(kodes), f"{kode} {nama}")
    return buf.getvalue()


# ------------------------------------------------------------------ Excel
_JUDUL_KOLOM = {
    "kode_bendungan": "Kode", "nama_bendungan": "Nama Bendungan",
    "nama_balai": "Balai", "nama_pulau": "Pulau", "tanggal": "Bulan",
    "tma": "TMA (mdpl)", "jenis": "Jenis Data", "bulan": "Bulan (angka)",
    "bon_a": "BON A (mdpl)", "bon_b": "BON B (mdpl)", "rtow": "RTOW (mdpl)",
    "selisih_rtow": "Selisih ke RTOW (m)", "selisih_ke_rtow_m": "Selisih ke RTOW (m)",
    "status_bon": "Status BON", "bon_terisi_otomatis": "BON Interpolasi",
    "nama_bulan": "Nama Bulan",
    "ketersediaan_juta_m3": "Ketersediaan Air (juta m³)",
    "kebutuhan_juta_m3": "Kebutuhan Air (juta m³)",
    "neraca_juta_m3": "Neraca Air (juta m³)",
    "status_neraca": "Status Neraca",
    "bulan_defisit_setahun": "Bulan Defisit (setahun)",
    "bulan_defisit_agu_des": "Bulan Defisit (Agu–Des)",
    "defisit_terbesar_juta_m3": "Defisit Terbesar (juta m³)",
    "bulan_defisit_terbesar": "Bulan Defisit Terbesar",
    "total_defisit_juta_m3": "Total Defisit Setahun (juta m³)",
    "total_kebutuhan_juta_m3": "Total Kebutuhan (juta m³)",
    "total_ketersediaan_juta_m3": "Total Ketersediaan (juta m³)",
    "penjelasan_neraca": "Penjelasan Neraca Air",
    "tma_prediksi": "TMA Prediksi (mdpl)",
    "tma_realisasi_terakhir": "TMA Realisasi Terakhir (mdpl)",
    "bulan_realisasi_terakhir": "Bulan Realisasi Terakhir",
    "perubahan_dari_realisasi_m": "Perubahan dari Realisasi (m)",
    "selisih_ke_bon_a_m": "Selisih ke BON A (m)",
    "selisih_ke_bon_b_m": "Selisih ke BON B (m)",
    "tma_prediksi_agustus": "TMA Prediksi Awal Agustus (mdpl)",
    "tma_min_prediksi": "TMA Minimum Prediksi (mdpl)",
    "waktu_tma_min": "Waktu TMA Minimum",
    "jumlah_periode_prediksi": "Jumlah Periode Prediksi",
    "periode_diatas_bon_a": "Periode Di atas BON A",
    "periode_antara_a_b": "Periode Di antara BON A-B",
    "periode_dibawah_bon_b": "Periode Di bawah BON B",
    "pertama_dibawah_bon": "Pertama di bawah BON",
    "periode": "Periode", "periode_terburuk": "Periode Terburuk",
    "format_periode": "Format Periode",
    "status_terburuk": "Status Terburuk", "penjelasan": "Penjelasan",
    "kategori_masalah": "Kategori Masalah", "keterangan": "Keterangan",
    "tindak_lanjut": "Tindak Lanjut", "jumlah_bendungan": "Jumlah Bendungan",
}


def _kol_excel(j: int) -> str:
    """0 -> A, 25 -> Z, 26 -> AA."""
    s = ""
    j += 1
    while j:
        j, sisa = divmod(j - 1, 26)
        s = chr(65 + sisa) + s
    return s


def _tulis(writer, nama_sheet: str, df: pd.DataFrame, catatan: str = ""):
    wb = writer.book
    if df is None or df.empty:
        ws = wb.add_worksheet(nama_sheet[:31])
        ws.write(0, 0, catatan or "Tidak ada data untuk lembar ini.",
                 wb.add_format({"italic": True, "font_color": "#666666"}))
        return
    d = df.copy()
    if "tanggal" in d.columns and pd.api.types.is_datetime64_any_dtype(d["tanggal"]):
        d["tanggal"] = d["tanggal"].dt.strftime("%Y-%m") + " (" + \
                       d["tanggal"].dt.month.map(BULAN_ID) + ")"
    d = d.rename(columns={k: v for k, v in _JUDUL_KOLOM.items() if k in d.columns})
    baris_awal = 2 if catatan else 0
    d.to_excel(writer, sheet_name=nama_sheet[:31], index=False,
               startrow=baris_awal + 1, header=False)
    ws = writer.sheets[nama_sheet[:31]]

    f_judul = wb.add_format({"bold": True, "font_size": 11,
                             "font_color": WARNA["navy"]})
    f_head = wb.add_format({"bold": True, "bg_color": WARNA["navy"],
                            "font_color": "white", "border": 1,
                            "text_wrap": True, "valign": "vcenter",
                            "align": "center"})
    if catatan:
        ws.write(0, 0, catatan, f_judul)
    for j, kol in enumerate(d.columns):
        ws.write(baris_awal, j, kol, f_head)
        panjang = d[kol].astype(str).str.len().quantile(0.92)
        lebar = max(len(str(kol)) + 2,
                    (int(panjang) + 2) if pd.notna(panjang) else 12)
        ws.set_column(j, j, min(max(lebar, 9), 60))
    ws.freeze_panes(baris_awal + 1, 2)
    ws.autofilter(baris_awal, 0, baris_awal + len(d), len(d.columns) - 1)
    ws.set_row(baris_awal, 32)

    for nm in ("Status BON", "Status Terburuk", "Kategori Masalah",
               "Status Neraca"):
        if nm in d.columns:
            j = list(d.columns).index(nm)
            kol = _kol_excel(j)
            rng = f"{kol}{baris_awal + 2}:{kol}{baris_awal + 1 + len(d)}"
            for teks, bg, fg in [(KRITIS, "#F8CBCB", "#9C0006"),
                                 (WASPADA, "#FFEB9C", "#9C6500"),
                                 (NORMAL, "#C6EFCE", "#006100"),
                                 (TANPA_BON, "#E4E4E4", "#5A5A5A"),
                                 ("Defisit", "#F8CBCB", "#9C0006"),
                                 ("Surplus", "#C6EFCE", "#006100"),
                                 ("Tanpa Data", "#E4E4E4", "#5A5A5A")]:
                ws.conditional_format(rng, {
                    "type": "text", "criteria": "containing", "value": teks,
                    "format": wb.add_format({"bg_color": bg, "font_color": fg})})


def _lembar_penjelasan(writer, konteks: dict):
    wb = writer.book
    ws = wb.add_worksheet("Penjelasan")
    ws.set_column(0, 0, 26)
    ws.set_column(1, 1, 110)
    f_h1 = wb.add_format({"bold": True, "font_size": 15, "font_color": WARNA["navy"]})
    f_h2 = wb.add_format({"bold": True, "font_size": 11, "bg_color": "#DCE6F1",
                          "font_color": WARNA["navy"], "border": 1})
    f_k = wb.add_format({"bold": True, "valign": "top", "border": 1,
                         "bg_color": "#F5F7FA"})
    f_v = wb.add_format({"text_wrap": True, "valign": "top", "border": 1})

    ws.write(0, 0, "LAPORAN PREDIKSI TMA BULANAN 2026 — PUSAT MONITORING BENDUNGAN", f_h1)
    ws.write(1, 0, "Subdit OP Bendungan dan Danau · Dit. Bina OP · Ditjen SDA")
    r = 3

    def bagian(judul, isi):
        nonlocal r
        ws.write(r, 0, judul, f_h2)
        ws.write(r, 1, "", f_h2)
        r += 1
        for k, v in isi:
            ws.write(r, 0, k, f_k)
            ws.write(r, 1, v, f_v)
            r += 1
        r += 1

    bagian("A. IDENTITAS LAPORAN", [
        ("Cakupan data", konteks.get("cakupan", "-")),
        ("Filter wilayah", konteks.get("filter", "Semua pulau & balai")),
        ("Jumlah bendungan", str(konteks.get("n_bendungan", "-"))),
        ("Periode realisasi", konteks.get("periode_real", "-")),
        ("Periode prediksi", konteks.get("periode_pred", "-")),
        ("Waktu pembuatan", konteks.get("waktu", "-")),
    ])
    bagian("B. METODOLOGI", [
        ("Skala periode", "Mengikuti format RTOW per bendungan: 15 Harian "
                          "(2 periode/bulan, label MM-01..MM-02) atau 10 Harian "
                          "(3 periode/bulan, MM-01..MM-03)."),
        ("Model", "LSTM encoder–decoder per periode (2 model global, satu per "
                  "format): Encoder LSTM(64) → RepeatVector → Decoder LSTM(64) "
                  "→ TimeDistributed(Dense(1)). Input 1 tahun periode terakhir, "
                  "output periode Agustus–Desember 2026."),
        ("Fitur masukan", "TMA rerata per periode hasil QC-interpolasi, volume "
                          "inflow per periode hasil neraca air, posisi musiman "
                          "sin/cos, dan SIFAT MUSIM tahun air (sifat_musim.csv: "
                          "Kering=-1/Normal=0/Basah=+1) sehingga model belajar "
                          "dari karakter tahun-tahun sebelumnya."),
        ("Normalisasi", "Min–max per bendungan; model global (pooled) dilatih dari "
                        "seluruh bendungan agar bendungan berdata pendek tetap "
                        "memperoleh pola musiman bersama."),
        ("Sumber data", "PostgreSQL SINBAD skema rawdata: sinbad_bendungan_tma, "
                        "_inflow, _outflow, _storagecurve."),
        ("QC data", "Spike harian > 3 m terhadap median rolling 7 hari dibuang; "
                    "gap ≤ 7 hari diinterpolasi linear; gap panjang diisi data "
                    "sebelumnya (ffill); TMA bulanan = rerata TMA harian ter-QC."),
        ("Referensi BON/RTOW", "data_bon/data_bon_a_bon_b_rtow_ketersediaan_"
                               "kebutuhan_air.xlsx — sheet 'bon a', 'bon b', "
                               "'rtow' (elevasi bulanan, mdpl) serta "
                               "'ketersediaan_air' dan 'kebutuhan_air' (m³/bulan); "
                               "kunci relasi = kolom id database SINBAD."),
        ("RTOW", "Rencana Tahunan Operasi Waduk — elevasi rencana operasi bulan "
                 "berjalan; sandingan tambahan di grafik (garis biru titik-titik)."),
        ("Neraca air", "Defisit bila kebutuhan > ketersediaan pada bulan tsb; "
                       "selain itu surplus. Nilai dalam m³/bulan dari file "
                       "referensi, ditampilkan dalam juta m³."),
    ])
    bagian("C. ARTI STATUS BON", [
        (NORMAL, "TMA prediksi berada di atas BON A — pasokan air aman sesuai "
                 "pola operasi normal."),
        (WASPADA, "TMA prediksi turun di bawah BON A tetapi masih di atas BON B — "
                  "mulai perlu pengetatan alokasi air dan pemantauan lebih rapat."),
        (KRITIS, "TMA prediksi menembus BON B — batas bawah operasi normal "
                 "terlampaui; perlu penyesuaian pola operasi, prioritas air baku, "
                 "dan koordinasi dengan pengguna air."),
        (TANPA_BON, "Nilai BON A/BON B belum tersedia pada file referensi sehingga "
                    "status tidak dapat dinilai (bukan berarti aman)."),
    ])
    bagian("D. PENJELASAN KOLOM UTAMA", [
        ("TMA (mdpl)", "Tinggi Muka Air, meter di atas permukaan laut; nilai "
                       "bulanan = rerata harian."),
        ("Jenis Data", "'realisasi' = data terukur; 'prediksi' = keluaran model LSTM."),
        ("Selisih ke BON A/B (m)", "TMA prediksi dikurangi ambang BON. Positif = "
                                   "masih di atas ambang; negatif = sudah di bawah "
                                   "ambang (defisit)."),
        ("Perubahan dari Realisasi (m)", "TMA prediksi bulan tinjauan dikurangi TMA "
                                         "realisasi terakhir; negatif = penurunan."),
        ("BON Interpolasi", "TRUE bila sebagian nilai BON bulanan kosong lalu diisi "
                            "interpolasi linear antar-bulan — status bulan tersebut "
                            "bersifat perkiraan."),
        ("Penjelasan", "Narasi otomatis kondisi bendungan: tren prediksi, TMA "
                       "minimum, dan bulan pertama pelanggaran BON."),
    ])
    bagian("E. ISI LEMBAR", [
        ("Ringkasan Bendungan", "Satu baris per bendungan: kondisi terakhir, "
                                "ringkasan prediksi, status terburuk, dan narasi."),
        (f"Pemantauan {konteks.get('bulan_fokus', 'Agustus')}",
         "Tabel pemantauan bulan fokus: TMA prediksi vs BON A/B beserta selisihnya."),
        ("Prediksi Bulanan", "Data lengkap format panjang (realisasi + prediksi) "
                             "per bendungan per bulan."),
        ("Di bawah BON B", "Hanya baris prediksi berstatus kritis — bahan utama "
                           "Nota Dinas kewaspadaan."),
        ("Rekap per Balai", "Jumlah bendungan per status, dikelompokkan per pulau "
                            "dan balai."),
        ("Rekap/Neraca Air", "Ketersediaan vs kebutuhan air per bendungan per "
                             "bulan (juta m³) beserta status surplus/defisit dan "
                             "narasi analisisnya."),
        ("Data Belum Cocok", "Bendungan yang datanya belum terhubung/lengkap "
                             "antar sumber beserta tindak lanjutnya."),
    ])
    bagian("F. CATATAN & BATASAN", [
        ("Ketidakpastian", "Prediksi statistik dari pola historis; kejadian ekstrem "
                           "(hujan anomali, operasi darurat, perubahan alokasi) "
                           "tidak terwakili. Gunakan sebagai indikasi dini, "
                           "dikonfirmasi dengan pemantauan harian."),
        ("Kualitas data masukan", "Bendungan dengan banyak gap data (lihat "
                                  "rekap_qc.xlsx) berpotensi memiliki galat lebih "
                                  "besar."),
        ("Pembaruan", "Jalankan python run_pipeline.py setiap awal bulan agar "
                      "realisasi dan prediksi bergerak maju."),
    ])
    ws.freeze_panes(3, 0)


def excel_laporan(df: pd.DataFrame, daftar: pd.DataFrame,
                  hist: pd.DataFrame | None = None, *,
                  hanya_kritis: bool = False, bulan_fokus: int = 8,
                  filter_teks: str = "Semua pulau & balai",
                  waktu: str = "",
                  rekap_qc: pd.DataFrame | None = None,
                  neraca: pd.DataFrame | None = None) -> bytes:
    """Workbook multi-sheet: penjelasan + ringkasan + tabel prediksi + masalah data."""
    d = df.copy()
    if hanya_kritis:
        kode_kritis = (d[(d["jenis"] == "prediksi") & (d["status_bon"] == KRITIS)]
                       ["kode_bendungan"].unique())
        d = d[d["kode_bendungan"].isin(kode_kritis)]

    real = d[d["jenis"] == "realisasi"]
    pred = d[d["jenis"] == "prediksi"]

    def rentang(x):
        if x.empty:
            return "-"
        a, b = x["tanggal"].min(), x["tanggal"].max()
        return f"{BULAN_ID[a.month]} {a.year} – {BULAN_ID[b.month]} {b.year}"

    konteks = {
        "cakupan": ("HANYA bendungan dengan prediksi di bawah BON B (kritis)"
                    if hanya_kritis else "Semua bendungan hasil pemodelan"),
        "filter": filter_teks,
        "n_bendungan": d["kode_bendungan"].nunique(),
        "periode_real": rentang(real),
        "periode_pred": rentang(pred),
        "waktu": waktu or "-",
        "bulan_fokus": BULAN_ID.get(bulan_fokus, "Agustus"),
    }

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter",
                        datetime_format="yyyy-mm") as writer:
        _lembar_penjelasan(writer, konteks)

        rekap = rekap_bendungan(d)
        _tulis(writer, "Ringkasan Bendungan", rekap,
               f"RINGKASAN PER BENDUNGAN — {konteks['cakupan']} · {filter_teks}")

        tab = tabel_pemantauan_bulan(d, bulan_fokus)
        _tulis(writer, f"Pemantauan {BULAN_ID[bulan_fokus]}", tab,
               f"PEMANTAUAN BENDUNGAN {BULAN_ID[bulan_fokus].upper()} 2026 — "
               f"TMA prediksi vs BON A/B")

        kolom_panjang = [c for c in ["kode_bendungan", "nama_bendungan", "nama_balai",
                                     "nama_pulau", "periode", "format_periode",
                                     "tanggal", "jenis", "tma", "bon_a",
                                     "bon_b", "rtow", "selisih_rtow", "status_bon",
                                     "bon_terisi_otomatis"]
                         if c in d.columns]
        _tulis(writer, "Prediksi per Periode",
               d[kolom_panjang].sort_values(["nama_bendungan", "tanggal"]),
               "DATA LENGKAP: REALISASI + PREDIKSI TMA PER PERIODE 10/15-HARIAN 2026")

        _tulis(writer, "Di bawah BON B",
               tab[tab["status_bon"] == KRITIS] if not tab.empty else tab,
               f"BENDUNGAN DI BAWAH BON B PADA {BULAN_ID[bulan_fokus].upper()} 2026")

        kritis_semua = (pred[pred["status_bon"] == KRITIS]
                        [[c for c in kolom_panjang if c in pred.columns]]
                        .sort_values(["nama_bendungan", "tanggal"]))
        _tulis(writer, "Di bawah BON B (Agu-Des)", kritis_semua,
               "SELURUH PERIODE PREDIKSI BERSTATUS DI BAWAH BON B "
               "(AGUSTUS–DESEMBER 2026)")

        _tulis(writer, "Rekap per Balai", rekap_per_balai(d),
               "REKAP JUMLAH BENDUNGAN PER STATUS MENURUT PULAU & BALAI")

        if neraca is not None and not neraca.empty:
            kode_d = set(d["kode_bendungan"].unique())
            n_d = neraca[neraca["kode_bendungan"].isin(kode_d)]
            _tulis(writer, "Rekap Neraca Air",
                   rekap_neraca(n_d).drop(columns=["nama_balai", "nama_pulau"],
                                          errors="ignore"),
                   "REKAP NERACA AIR PER BENDUNGAN — kebutuhan vs ketersediaan "
                   "(defisit bila kebutuhan > ketersediaan)")
            _tulis(writer, "Neraca Air Bulanan", tabel_neraca(n_d),
                   "NERACA AIR BULANAN (juta m³) — sumber sheet "
                   "ketersediaan_air & kebutuhan_air")

        _tulis(writer, "Data Belum Cocok",
               daftar_belum_cocok(daftar, df, hist, rekap_qc),
               "DATA BENDUNGAN YANG BELUM COCOK / BELUM LENGKAP ANTAR SUMBER "
               "(dihitung dari seluruh bendungan, tanpa filter)")
    return buf.getvalue()
