"""Kecukupan air berbasis RTOW PERIODIK (workbook baru) + storage curve SINBAD.

Sumber data:
1. data_bon/new_data_bon_a_bon_b_rtow_ketersediaan_kebutuhan_air.xlsx
   Sheet per format RTOW (sufiks _15 = 15-harian, _10 = 10-harian):
     bon_a_15/_10, bon_b_15/_10        : Bon A / Bon B (mdpl) per periode
     rtow_15/_10                       : elevasi rencana RTOW (mdpl) per periode
     ketersediaan_air_15/_10           : ketersediaan air (m3/s) per periode
     kebutuhan_air_15/_10              : kebutuhan air (m3/s) per periode
     elevasi_kekeringan                : elevasi_kekeringan_rtow &
                                         elevasi_dasar_waduk_rtow (nilai tunggal;
                                         fallback kolom *_sinbad bila kosong)
   Kunci relasi = kolom `id` (id bendungan di database SINBAD).
   NILAI KOSONG DIBIARKAN KOSONG — tanpa interpolasi/pengisian; tampilan
   cukup menampilkan placeholder "data belum tersedia".
2. data/raw/storagecurve_harian.parquet — pasangan elevasi–volume per bendungan
   dari rawdata.sinbad_bendungan_storagecurve (hasil ekstraksi pipeline lewat
   koneksi database yang sudah ada). Dipakai untuk konversi volume <-> elevasi
   (interpolasi kurva penuh, menggantikan interpolasi 2-titik LWL/NWL).

Seluruh logika di modul ini murni pandas/aturan (rule-based) — TIDAK menyentuh
model LSTM (arsitektur/training/inference tetap utuh).
"""
import math
import os
import re

import numpy as np
import pandas as pd

from src.utils import path_root, FORMAT_15, FORMAT_10, PERIODE_PER_BULAN
from src.laporan import KRITIS, WASPADA, NORMAL, TANPA_BON

FILE_RTOW_PERIODIK = os.path.join(
    "data_bon", "new_data_bon_a_bon_b_rtow_ketersediaan_kebutuhan_air.xlsx")
FILE_STORAGE_CURVE = os.path.join("data", "raw", "storagecurve_harian.parquet")

# Rumus 1: vol_m3 = nilai_m3s * jumlah_hari * 24 * 3600
HARI_PERIODE = {FORMAT_15: 15, FORMAT_10: 10}
DETIK_HARI = 24 * 3600

# "Mendekati batas elevasi kekeringan" = selisih TMA terhadap elevasi
# kekeringan di bawah ambang ini (meter). Parameter agar mudah dikalibrasi
# per bendungan (override lewat argumen buffer_m pada rekomendasi_skenario3).
BUFFER_KEKERINGAN_M = 0.5

# Kelas ketahanan air (days of supply, hari)
BATAS_KRITIS_HARI = 7
BATAS_SIAGA_HARI = 30

CUKUP, BELUM_CUKUP = "Cukup", "Belum cukup"


# ------------------------------------------------------------------ pembacaan
def _cari_sheet(xl: pd.ExcelFile, nama: str):
    """Nama sheet case-insensitive + strip spasi; None bila tidak ada."""
    peta = {str(s).strip().lower(): s for s in xl.sheet_names}
    return peta.get(nama.strip().lower())


def _lebar_ke_panjang(df: pd.DataFrame, nama_nilai: str) -> pd.DataFrame:
    """Sheet lebar (id, ..., '01-01'..'12-03') -> [id_db, periode, <nilai>].

    Toleran: kolom di-strip, id non-numerik dilewati, nilai non-numerik
    menjadi NaN (dibiarkan kosong — tidak diisi/diinterpolasi).
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    kol_per = [c for c in df.columns if re.fullmatch(r"\d{2}-\d{2}", c)]
    kol_id = next((c for c in df.columns if c.lower() == "id"), None)
    if kol_id is None or not kol_per:
        return pd.DataFrame(columns=["id_db", "periode", nama_nilai])
    df["id_db"] = pd.to_numeric(df[kol_id], errors="coerce")
    df = df.dropna(subset=["id_db"])
    df["id_db"] = df["id_db"].astype(int)
    df = df.drop_duplicates("id_db")
    m = df.melt(id_vars="id_db", value_vars=kol_per,
                var_name="periode", value_name=nama_nilai)
    m[nama_nilai] = pd.to_numeric(m[nama_nilai], errors="coerce")
    return m


def baca_rtow_periodik(fp: str = None) -> pd.DataFrame:
    """Gabungan seluruh sheet periodik workbook baru (format panjang).

    Return: [id_db, format, periode, bon_a, bon_b, rtow,
             ketersediaan_m3s, kebutuhan_m3s] — kosong bila file tidak ada.
    """
    fp = path_root(fp or FILE_RTOW_PERIODIK)
    if not os.path.exists(fp):
        return pd.DataFrame()
    xl = pd.ExcelFile(fp)
    dasar = {"bon_a": "bon_a", "bon_b": "bon_b", "rtow": "rtow",
             "ketersediaan_air": "ketersediaan_m3s",
             "kebutuhan_air": "kebutuhan_m3s"}
    hasil = []
    for suf, fmt in (("15", FORMAT_15), ("10", FORMAT_10)):
        gab = None
        for sheet_dasar, nilai in dasar.items():
            s = _cari_sheet(xl, f"{sheet_dasar}_{suf}")
            if s is None:
                continue
            m = _lebar_ke_panjang(xl.parse(s), nilai)
            gab = m if gab is None else gab.merge(m, on=["id_db", "periode"],
                                                  how="outer")
        if gab is not None and not gab.empty:
            gab["format"] = fmt
            hasil.append(gab)
    if not hasil:
        return pd.DataFrame()
    out = pd.concat(hasil, ignore_index=True)
    for c in ("bon_a", "bon_b", "rtow", "ketersediaan_m3s", "kebutuhan_m3s"):
        if c not in out.columns:
            out[c] = np.nan
    return out.sort_values(["id_db", "periode"]).reset_index(drop=True)


def baca_elevasi_kekeringan(fp: str = None) -> pd.DataFrame:
    """Sheet elevasi_kekeringan -> nilai tunggal per bendungan + fallback.

    elevasi_kekeringan : elevasi_kekeringan_rtow, kosong -> elevasi_kekeringan_sinbad
    elevasi_dasar      : elevasi_dasar_waduk_rtow, kosong -> elevasi_dasar_intake_sinbad
    Kolom sumber_* menandai asal nilai ('RTOW'/'SINBAD'/None).
    """
    fp = path_root(fp or FILE_RTOW_PERIODIK)
    if not os.path.exists(fp):
        return pd.DataFrame()
    xl = pd.ExcelFile(fp)
    s = _cari_sheet(xl, "elevasi_kekeringan")
    if s is None:
        return pd.DataFrame()
    df = xl.parse(s)
    df.columns = [str(c).strip().lower() for c in df.columns]
    kol_id = next((c for c in df.columns if c == "id"), None)
    if kol_id is None:
        return pd.DataFrame()
    df["id_db"] = pd.to_numeric(df[kol_id], errors="coerce")
    df = df.dropna(subset=["id_db"])
    df["id_db"] = df["id_db"].astype(int)

    def _angka(kolom):
        return (pd.to_numeric(df[kolom], errors="coerce")
                if kolom in df.columns else pd.Series(np.nan, index=df.index))

    kek_rtow = _angka("elevasi_kekeringan_rtow")
    kek_sin = _angka("elevasi_kekeringan_sinbad")
    das_rtow = _angka("elevasi_dasar_waduk_rtow")
    das_sin = _angka("elevasi_dasar_intake_sinbad")
    out = pd.DataFrame({
        "id_db": df["id_db"],
        "elevasi_kekeringan": kek_rtow.fillna(kek_sin),
        "sumber_kekeringan": np.select(
            [kek_rtow.notna(), kek_sin.notna()], ["RTOW", "SINBAD"], None),
        "elevasi_dasar": das_rtow.fillna(das_sin),
        "sumber_dasar": np.select(
            [das_rtow.notna(), das_sin.notna()], ["RTOW", "SINBAD"], None),
    })
    return out.drop_duplicates("id_db").reset_index(drop=True)


def baca_storage_curve(fp: str = None) -> pd.DataFrame:
    """Storage curve seluruh bendungan: [kode_bendungan, elevasi, volume(m3)]."""
    fp = path_root(fp or FILE_STORAGE_CURVE)
    if not os.path.exists(fp):
        return pd.DataFrame()
    sc = pd.read_parquet(fp)
    sc = sc.dropna(subset=["elevasi", "volume"])
    return sc.sort_values(["kode_bendungan", "elevasi"]).reset_index(drop=True)


# ------------------------------------------------------------------ konversi
def debit_ke_volume(nilai_m3s, fmt: str):
    """Rumus 1: vol_m3 = m3/s * jumlah_hari * 24 * 3600 (15 atau 10 hari)."""
    hari = HARI_PERIODE.get(fmt, 15)
    return nilai_m3s * hari * DETIK_HARI


def kurva_bendungan(sc: pd.DataFrame, kode: str) -> pd.DataFrame:
    """Kurva satu bendungan, terurut elevasi; kosong bila tidak tersedia."""
    if sc is None or sc.empty:
        return pd.DataFrame()
    k = sc[sc["kode_bendungan"] == kode]
    return k.sort_values("elevasi").reset_index(drop=True)


def volume_ke_elevasi(kurva: pd.DataFrame, vol_m3):
    """Rumus 2: interpolasi volume -> elevasi pada storage curve penuh."""
    if kurva is None or kurva.empty:
        return np.nan if np.isscalar(vol_m3) else np.full(len(vol_m3), np.nan)
    k = kurva.sort_values("volume")
    return np.interp(vol_m3, k["volume"].values, k["elevasi"].values)


def elevasi_ke_volume(kurva: pd.DataFrame, elevasi):
    """Interpolasi elevasi -> volume (m3) pada storage curve penuh."""
    if kurva is None or kurva.empty:
        return np.nan if np.isscalar(elevasi) else np.full(len(elevasi), np.nan)
    return np.interp(elevasi, kurva["elevasi"].values, kurva["volume"].values)


# ------------------------------------------------------------------ periode
def periode_aktif(fmt: str, tanggal=None, tersedia=None) -> str:
    """Label periode RTOW berjalan 'MM-PP' untuk tanggal berjalan.

    Bila `tersedia` (kumpulan label yang punya data) diberikan dan label
    aktif tidak ada di dalamnya -> pakai label terakhir yang tersedia.
    """
    t = pd.Timestamp(tanggal) if tanggal is not None else pd.Timestamp.now()
    n = PERIODE_PER_BULAN.get(fmt, 2)
    if n == 3:
        idx = 1 if t.day <= 10 else (2 if t.day <= 20 else 3)
    else:
        idx = 1 if t.day <= 15 else 2
    label = f"{t.month:02d}-{idx:02d}"
    if tersedia is not None:
        tersedia = sorted(set(tersedia))
        if label not in tersedia and tersedia:
            label = tersedia[-1]
    return label


# ------------------------------------------------------------------ keputusan
def status_tiga_zona(elevasi, bon_a, bon_b) -> str:
    """Keputusan 3 zona memakai Bon A/Bon B PER PERIODE (bukan interpolasi):
    elevasi > Bon A -> di atas; Bon B <= elevasi <= Bon A -> di antara;
    elevasi < Bon B -> di bawah. Data kurang -> Tanpa Data BON."""
    if pd.isna(elevasi) or pd.isna(bon_a) or pd.isna(bon_b):
        return TANPA_BON
    if elevasi > bon_a:
        return NORMAL
    if elevasi < bon_b:
        return KRITIS
    return WASPADA


def hitung_kecukupan(ketersediaan_m3s, kebutuhan_m3s, fmt: str):
    """Kecukupan air satu periode. None bila data belum tersedia.

    Return dict: ketersediaan_m3, kebutuhan_m3, deviasi_m3, deviasi_pct,
    status ('Cukup' bila deviasi >= 0, selain itu 'Belum cukup').
    """
    if pd.isna(ketersediaan_m3s) or pd.isna(kebutuhan_m3s):
        return None
    ket = float(debit_ke_volume(ketersediaan_m3s, fmt))
    keb = float(debit_ke_volume(kebutuhan_m3s, fmt))
    dev = ket - keb
    pct = (dev / keb * 100.0) if keb > 0 else np.nan   # jaga pembagi nol
    return {"ketersediaan_m3": ket, "kebutuhan_m3": keb,
            "deviasi_m3": dev, "deviasi_pct": pct,
            "status": CUKUP if dev >= 0 else BELUM_CUKUP}


def faktor_k(ketersediaan_m3s, kebutuhan_m3s):
    """Faktor alokasi K = ketersediaan / kebutuhan (dibatasi maksimum 1)."""
    if pd.isna(ketersediaan_m3s) or pd.isna(kebutuhan_m3s):
        return np.nan
    if kebutuhan_m3s <= 0:
        return 1.0
    return float(min(1.0, ketersediaan_m3s / kebutuhan_m3s))


def estimasi_ketahanan_hari(kurva, tma, elevasi_kekeringan,
                            ketersediaan_m3s, kebutuhan_m3s):
    """Days-of-Supply: volume tampungan di atas elevasi kekeringan dibagi
    laju pengurasan bersih periode berjalan (kebutuhan - ketersediaan).

    Return hari (float), math.inf bila tidak ada pengurasan bersih,
    NaN bila data tidak cukup."""
    if (kurva is None or kurva.empty or pd.isna(tma)
            or pd.isna(elevasi_kekeringan)
            or pd.isna(ketersediaan_m3s) or pd.isna(kebutuhan_m3s)):
        return np.nan
    vol = (float(elevasi_ke_volume(kurva, tma))
           - float(elevasi_ke_volume(kurva, elevasi_kekeringan)))
    vol = max(0.0, vol)
    neto_m3s = float(kebutuhan_m3s) - float(ketersediaan_m3s)
    if neto_m3s <= 0:
        return math.inf
    return vol / (neto_m3s * DETIK_HARI)


def kelas_ketahanan(hari) -> str:
    """Kelas ketahanan air: kritis (<7 hari), siaga (7-30), cukup (>=30)."""
    if hari is None or (isinstance(hari, float) and np.isnan(hari)):
        return "tidak diketahui"
    if hari < BATAS_KRITIS_HARI:
        return "kritis"
    if hari < BATAS_SIAGA_HARI:
        return "siaga"
    return "cukup"


def rekomendasi_skenario3(*, ada_permintaan_resmi: bool, k, tma, bon_a, bon_b,
                          elevasi_kekeringan, kecukupan, ketahanan_hari,
                          buffer_m: float = BUFFER_KEKERINGAN_M) -> dict:
    """Alur keputusan Skenario 3 (rule-based, di atas data RTOW/kecukupan air).

    Return dict:
      langkah  : list (ikon, teks) — tiap langkah alur + hasilnya
      badge    : label status utama
      warna    : 'hijau' / 'amber' / 'merah' / 'abu'
      k_rekomendasi : Faktor K yang direkomendasikan (0 saat curtailment)
      kelas    : kelas ketahanan air (bila dihitung)
    """
    langkah = []

    # 1. cek Faktor K terlebih dahulu
    if pd.isna(k):
        langkah.append(("1️⃣", "**Cek Faktor K** — belum dapat dihitung "
                               "(data ketersediaan/kebutuhan periode berjalan "
                               "belum tersedia)."))
    else:
        langkah.append(("1️⃣", f"**Cek Faktor K** — Faktor K periode berjalan "
                               f"= **{k:.2f}**."))

    dev = kecukupan["deviasi_m3"] if kecukupan else np.nan
    pct = kecukupan["deviasi_pct"] if kecukupan else np.nan

    # 2. ada permintaan pelayanan resmi (UPB/UPI, Kementan, pihak lain)
    if ada_permintaan_resmi:
        langkah.append(("2️⃣", "**Ada permintaan pelayanan resmi** (UPB/UPI, "
                               "Kementan, atau pihak lain — surat/permohonan "
                               "resmi)."))
        langkah.append(("3️⃣", "**Rekomendasi:** lakukan **evaluasi kebutuhan "
                               "air irigasi di lapangan melalui pemodelan "
                               "hidrologi**."))
        if kecukupan is None:
            return {"langkah": langkah + [
                        ("4️⃣", "Status ketersediaan air belum dapat dinilai — "
                               "data RTOW periode berjalan belum tersedia.")],
                    "badge": "Data belum tersedia", "warna": "abu",
                    "k_rekomendasi": k, "kelas": None}
        if dev >= 0:
            langkah.append(("4️⃣", f"**Status ketersediaan air: AMAN** — "
                                   f"ketersediaan {kecukupan['ketersediaan_m3']:,.0f} m³ "
                                   f"≥ kebutuhan {kecukupan['kebutuhan_m3']:,.0f} m³ "
                                   f"(deviasi {dev:+,.0f} m³ / {pct:+.1f}%)."))
            return {"langkah": langkah, "badge": "AMAN", "warna": "hijau",
                    "k_rekomendasi": k, "kelas": None}
        langkah.append(("4️⃣", f"**Status: PERLU PENYESUAIAN ALOKASI** — "
                               f"ketersediaan {kecukupan['ketersediaan_m3']:,.0f} m³ "
                               f"< kebutuhan {kecukupan['kebutuhan_m3']:,.0f} m³ "
                               f"(deviasi {dev:+,.0f} m³ / {pct:+.1f}%)."))
        return {"langkah": langkah, "badge": "Perlu penyesuaian alokasi",
                "warna": "amber", "k_rekomendasi": k, "kelas": None}

    # 3. tidak ada permintaan resmi -> mitigasi berbasis analisis RTOW
    langkah.append(("2️⃣", "**Tidak ada permintaan resmi** — lakukan "
                           "**mitigasi berbasis analisis RTOW**."))
    if pd.isna(tma) or pd.isna(bon_b):
        langkah.append(("3️⃣", "Perbandingan TMA terhadap Bon B belum dapat "
                               "dilakukan — data RTOW periode berjalan belum "
                               "tersedia."))
        return {"langkah": langkah, "badge": "Data belum tersedia",
                "warna": "abu", "k_rekomendasi": k, "kelas": None}

    dekat_kekeringan = (pd.notna(elevasi_kekeringan)
                        and (tma - elevasi_kekeringan) <= buffer_m)
    if tma < bon_b and dekat_kekeringan:
        kelas = kelas_ketahanan(ketahanan_hari)
        if ketahanan_hari is not None and not (
                isinstance(ketahanan_hari, float) and np.isnan(ketahanan_hari)):
            teks_hari = ("tanpa pengurasan bersih (ketersediaan ≥ kebutuhan)"
                         if math.isinf(ketahanan_hari)
                         else f"aman untuk ~{ketahanan_hari:.0f} hari ke depan")
        else:
            teks_hari = "belum dapat dihitung (data belum lengkap)"
        aksi_kelas = {
            "kritis": "segera koordinasikan pengurangan pelepasan dengan "
                      "pengelola dan siapkan pasokan alternatif",
            "siaga": "perketat alokasi, prioritaskan air baku, evaluasi ulang "
                     "tiap periode",
            "cukup": "pertahankan pembatasan dan pantau tiap periode",
        }.get(kelas, "lengkapi data untuk penilaian ketahanan air")
        langkah += [
            ("3️⃣", f"**TMA {tma:.2f} mdpl di bawah Bon B {bon_b:.2f} mdpl** "
                    f"dan **mendekati elevasi kekeringan "
                    f"{elevasi_kekeringan:.2f} mdpl** (buffer {buffer_m:.2f} m)."),
            ("4️⃣", "**Ubah faktor alokasi K menjadi 0** dan lakukan "
                    "**pengurangan debit pelepasan (outflow reduction)**."),
            ("5️⃣", f"**Estimasi ketahanan air:** {teks_hari} — kelas "
                    f"**{kelas.upper()}**; {aksi_kelas}."),
        ]
        return {"langkah": langkah, "badge": "K = 0 — outflow reduction",
                "warna": "merah", "k_rekomendasi": 0.0, "kelas": kelas}

    if tma < bon_b:
        langkah.append(("3️⃣", f"**TMA {tma:.2f} mdpl di bawah Bon B "
                               f"{bon_b:.2f} mdpl** namun belum mendekati "
                               f"elevasi kekeringan — perketat alokasi dan "
                               f"evaluasi tiap periode."))
        return {"langkah": langkah, "badge": "Pengetatan alokasi",
                "warna": "amber", "k_rekomendasi": k, "kelas": None}

    langkah.append(("3️⃣", f"**TMA {tma:.2f} mdpl masih di atas Bon B "
                           f"{bon_b:.2f} mdpl** — pemantauan lanjutan, belum "
                           f"perlu curtailment."))
    return {"langkah": langkah, "badge": "Pemantauan lanjutan",
            "warna": "hijau", "k_rekomendasi": k, "kelas": None}
