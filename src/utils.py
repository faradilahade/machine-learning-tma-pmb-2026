"""Utilitas umum: load config, baca BON A/B/RTOW/neraca, daftar bendungan,
skala periode (10/15-harian), dan sifat musim."""
import os
import re
import yaml
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BULAN_MAP = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "Mei": 5, "Jun": 6,
             "Jul": 7, "Agu": 8, "Sep": 9, "Okt": 10, "Nov": 11, "Des": 12}

# ------------------------------------------------------------------ periode
# Skala periode per bendungan mengikuti format RTOW-nya (fromat-rtow.csv):
#   "15 Harian" -> 2 periode/bulan : tgl 1-15, 16-akhir   -> label MM-01, MM-02
#   "10 Harian" -> 3 periode/bulan : 1-10, 11-20, 21-akhir -> MM-01..MM-03
FORMAT_15, FORMAT_10 = "15 Harian", "10 Harian"
PERIODE_PER_BULAN = {FORMAT_15: 2, FORMAT_10: 3}
SIFAT_MUSIM_SKOR = {"Kering": -1.0, "Normal": 0.0, "Basah": 1.0}


def periode_dari_tanggal(tanggal: pd.Series, fmt: str) -> pd.Series:
    """Indeks periode (1..2 atau 1..3) dari tanggal harian."""
    d = tanggal.dt.day
    if PERIODE_PER_BULAN.get(fmt, 2) == 3:
        return np.select([d <= 10, d <= 20], [1, 2], 3)
    return np.where(d <= 15, 1, 2)


def label_periode(bulan, idx) -> str:
    """'MM-PP' — mis. bulan 8 periode 1 -> '08-01' (sumbu-x grafik RTOW)."""
    return f"{int(bulan):02d}-{int(idx):02d}"


def awal_periode(tahun: int, bulan: int, idx: int, fmt: str) -> pd.Timestamp:
    """Tanggal awal sebuah periode (dipakai sebagai kunci waktu)."""
    if PERIODE_PER_BULAN.get(fmt, 2) == 3:
        hari = {1: 1, 2: 11, 3: 21}[int(idx)]
    else:
        hari = {1: 1, 2: 16}[int(idx)]
    return pd.Timestamp(int(tahun), int(bulan), hari)


def daftar_periode(fmt: str, bulan_awal: int = 1, bulan_akhir: int = 12) -> list:
    """Urutan label periode kalender: ['01-01','01-02',...] s.d. bulan_akhir."""
    n = PERIODE_PER_BULAN.get(fmt, 2)
    return [label_periode(b, i)
            for b in range(bulan_awal, bulan_akhir + 1)
            for i in range(1, n + 1)]


def akhir_periode(tahun: int, bulan: int, idx: int, fmt: str) -> pd.Timestamp:
    """Tanggal TERAKHIR sebuah periode (periode terakhir bulan = akhir bulan).

    Dipakai menentukan periode yang sudah LENGKAP datanya: periode dianggap
    lengkap bila akhir_periode <= tanggal data harian terakhir.
    """
    n = PERIODE_PER_BULAN.get(fmt, 2)
    if int(idx) < n:
        return awal_periode(tahun, bulan, int(idx) + 1, fmt) - pd.Timedelta(days=1)
    return pd.Timestamp(int(tahun), int(bulan), 1) + pd.offsets.MonthEnd(1)


def periode_berikutnya(tahun: int, bulan: int, idx: int, fmt: str,
                       n: int) -> list:
    """n periode kalender SETELAH (tahun, bulan, idx) -> [(tahun, bulan, idx)].

    Melewati batas tahun (12-02 15H -> 01-01 tahun berikutnya) sehingga
    horizon prediksi dapat bergulir bulanan tanpa terikat Agu-Des.
    """
    per_bln = PERIODE_PER_BULAN.get(fmt, 2)
    hasil, t, b, i = [], int(tahun), int(bulan), int(idx)
    for _ in range(n):
        i += 1
        if i > per_bln:
            i, b = 1, b + 1
            if b > 12:
                b, t = 1, t + 1
        hasil.append((t, b, i))
    return hasil


def agregasi_periode(df: pd.DataFrame, fmt: str, kolom_nilai: str,
                     cara: str = "mean") -> pd.DataFrame:
    """Agregasi deret harian 1 bendungan -> per periode sesuai format.

    Return: [tanggal (awal periode), periode (label 'MM-PP'), <kolom_nilai>]
    """
    d = df.copy()
    d["_idx"] = periode_dari_tanggal(d["tanggal"], fmt)
    d["_thn"] = d["tanggal"].dt.year
    d["_bln"] = d["tanggal"].dt.month
    g = (d.groupby(["_thn", "_bln", "_idx"])[kolom_nilai]
           .agg(cara).reset_index())
    g["tanggal"] = [awal_periode(t, b, i, fmt)
                    for t, b, i in zip(g["_thn"], g["_bln"], g["_idx"])]
    g["periode"] = [label_periode(b, i) for b, i in zip(g["_bln"], g["_idx"])]
    return (g[["tanggal", "periode", kolom_nilai]]
            .sort_values("tanggal").reset_index(drop=True))


def bulanan_ke_periode(df_bulanan: pd.DataFrame, fmt: str,
                       kolom_bulan: str = "bulan") -> pd.DataFrame:
    """Rentangkan nilai bulanan (BON/RTOW/neraca) ke tiap periode bulan itu."""
    n = PERIODE_PER_BULAN.get(fmt, 2)
    ulang = df_bulanan.loc[df_bulanan.index.repeat(n)].copy()
    ulang["idx_periode"] = list(range(1, n + 1)) * len(df_bulanan)
    ulang["periode"] = [label_periode(b, i) for b, i in
                        zip(ulang[kolom_bulan], ulang["idx_periode"])]
    return ulang.drop(columns="idx_periode").reset_index(drop=True)


def tahun_air(ts: pd.Timestamp) -> str:
    """Tahun air sifat musim (Nov-Okt): Nov 2024 & Mei 2025 -> '2024-2025'."""
    y = ts.year
    return f"{y}-{y + 1}" if ts.month >= 11 else f"{y - 1}-{y}"


def baca_format_rtow(cfg: dict, daftar: pd.DataFrame = None) -> pd.Series:
    """Map kode_bendungan -> format periode ('15 Harian'/'10 Harian').

    Sumber: fromat-rtow.csv (id;nama_bendungan;format). Nilai selain kedua
    format valid (mis. 'Tidak Melayani') atau id tak terdaftar -> default
    '15 Harian'.
    """
    daftar = daftar if daftar is not None else baca_daftar_bendungan(cfg)
    fp = path_root(cfg["bon"].get("file_format_rtow", "data_bon/fromat-rtow.csv"))
    sep = cfg["bon"].get("csv_sep", ";")
    f = pd.read_csv(fp, sep=sep)
    f.columns = [str(c).strip().lower() for c in f.columns]
    f["id_db"] = pd.to_numeric(f["id"], errors="coerce")
    f = f.dropna(subset=["id_db"])
    f["id_db"] = f["id_db"].astype(int)
    f = f.drop_duplicates("id_db")
    f["format"] = f["format"].astype(str).str.strip()
    f.loc[~f["format"].isin(PERIODE_PER_BULAN), "format"] = FORMAT_15
    m = daftar[["kode_bendungan", "id_db"]].merge(
        f[["id_db", "format"]], on="id_db", how="left")
    m["format"] = m["format"].fillna(FORMAT_15)
    m = m.drop_duplicates("kode_bendungan")
    return m.set_index("kode_bendungan")["format"]


def baca_sifat_musim(cfg: dict, daftar: pd.DataFrame = None) -> pd.DataFrame:
    """Sifat musim per bendungan per tahun air -> format panjang.

    sifat_musim.csv: id;nama_bendungan;2022-2023;...;2025-2026
    Return: [kode_bendungan, tahun_air, sifat ('Basah'/'Normal'/'Kering'),
             skor_musim (-1/0/+1)]
    """
    daftar = daftar if daftar is not None else baca_daftar_bendungan(cfg)
    fp = path_root(cfg["bon"].get("file_sifat_musim", "data_bon/sifat_musim.csv"))
    sep = cfg["bon"].get("csv_sep", ";")
    s = pd.read_csv(fp, sep=sep)
    s.columns = [str(c).strip() for c in s.columns]
    s["id_db"] = pd.to_numeric(s["id"], errors="coerce")
    s = s.dropna(subset=["id_db"])
    s["id_db"] = s["id_db"].astype(int)
    kolom_tahun = [c for c in s.columns if re.fullmatch(r"\d{4}-\d{4}", c)]
    m = daftar[["kode_bendungan", "id_db"]].merge(
        s[["id_db"] + kolom_tahun], on="id_db", how="left")
    panjang = m.melt(id_vars="kode_bendungan", value_vars=kolom_tahun,
                     var_name="tahun_air", value_name="sifat")
    panjang["sifat"] = (panjang["sifat"].astype(str).str.strip().str.title()
                        .where(lambda x: x.isin(SIFAT_MUSIM_SKOR), "Normal"))
    panjang["skor_musim"] = panjang["sifat"].map(SIFAT_MUSIM_SKOR)
    return panjang.sort_values(["kode_bendungan", "tahun_air"]).reset_index(drop=True)


def skor_musim_untuk(ts: pd.Timestamp, peta_musim: dict, kode: str) -> float:
    """Skor sifat musim utk 1 timestep; di luar rentang data -> tahun air
    terakhir yang diketahui (asumsi berlanjut), tanpa data -> 0 (Normal)."""
    tabel = peta_musim.get(kode)
    if not tabel:
        return 0.0
    ta = tahun_air(ts)
    if ta in tabel:
        return tabel[ta]
    kunci = sorted(tabel)
    if ta < kunci[0]:
        return 0.0
    return tabel[kunci[-1]]


def load_config(path: str = None) -> dict:
    """config.yaml (lokal, berisi kredensial, di-.gitignore) bila ada;
    fallback config.cloud.yaml (tanpa kredensial) saat deploy di
    Streamlit Community Cloud."""
    if path is None:
        path = os.path.join(ROOT, "config.yaml")
        if not os.path.exists(path):
            path = os.path.join(ROOT, "config.cloud.yaml")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # periode.akhir "auto" / kosong -> H-1 saat run (prediksi bergulir:
    # tiap run di awal bulan otomatis menarik realisasi s.d. kemarin)
    akhir = str(cfg.get("periode", {}).get("akhir", "") or "").strip().lower()
    if akhir in ("", "auto", "none", "null"):
        cfg.setdefault("periode", {})["akhir"] = (
            pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
        ).strftime("%Y-%m-%d")
    return cfg


def path_root(*parts) -> str:
    return os.path.join(ROOT, *parts)


def normalisasi_tanggal(df: pd.DataFrame, kolom: str = "tanggal") -> pd.DataFrame:
    """Paksa kolom tanggal ke datetime64[ns] (normalisasi ke tengah malam).

    Query ekstraksi memakai '"timestamp"::date' sehingga psycopg2 mengembalikan
    objek datetime.date -> parquet menyimpannya sebagai dtype object. Bila satu
    tabel sudah datetime64 dan tabel lain masih object, merge antar-tabel gagal:
    "You are trying to merge on datetime64[ns] and object columns".
    Semua pembaca/penulis parquet harian melewati fungsi ini agar dtype seragam.
    """
    if kolom in df.columns and not pd.api.types.is_datetime64_any_dtype(df[kolom]):
        df = df.copy()
        df[kolom] = pd.to_datetime(df[kolom], errors="coerce").dt.normalize()
    return df


def kode_ke_id(kode: str) -> int:
    """BD001 -> 1, BD015 -> 15 (mode strip_prefix)."""
    m = re.search(r"(\d+)$", str(kode).strip())
    if not m:
        raise ValueError(f"Tidak bisa memetakan kode '{kode}' ke id database")
    return int(m.group(1))


def baca_daftar_bendungan(cfg: dict) -> pd.DataFrame:
    """Daftar bendungan yang dimodelkan = HANYA yang ada di data BON A/B.

    daftar_bendungan.csv (delimiter ';'):
      id;kode_bendungan;nama_bendungan;nama_balai;nama_pulau
    Kolom 'id' (atau 'id_db' pada format lama) = id bendungan di database
    SINBAD, dipakai LANGSUNG untuk penarikan data TMA/inflow/outflow/storage
    curve. Bila keduanya tidak ada, jatuh ke pemetaan strip_prefix (BD001->1).
    """
    sep = cfg["bon"].get("csv_sep", ";")
    fp = path_root(cfg["bon"]["file_daftar"])
    df = pd.read_csv(fp, sep=sep, dtype={"kode_bendungan": str})
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.loc[:, ~df.columns.str.startswith("unnamed")]   # kolom ';' sisa Excel
    if "nama_bendungan" not in df.columns:
        raise ValueError(
            f"Kolom 'nama_bendungan' tidak ditemukan di {fp}. "
            f"Kolom terbaca: {list(df.columns)} -- cek delimiter (config bon.csv_sep='{sep}').")
    df["kode_bendungan"] = df["kode_bendungan"].str.strip()
    df["nama_bendungan"] = df["nama_bendungan"].astype(str).str.strip()
    if "id" in df.columns:
        df["id_db"] = pd.to_numeric(df["id"], errors="raise").astype(int)
        df = df.drop(columns="id")
    elif "id_db" in df.columns:
        df["id_db"] = df["id_db"].astype(int)
    else:
        df["id_db"] = df["kode_bendungan"].map(kode_ke_id)
    if df["id_db"].duplicated().any():
        dup = df[df["id_db"].duplicated(keep=False)]
        raise ValueError(f"id_db duplikat (mirip kasus GEMBONG di SINBAD):\n{dup}")
    if df["kode_bendungan"].duplicated().any():
        dup = df[df["kode_bendungan"].duplicated(keep=False)]
        raise ValueError(f"kode_bendungan duplikat di {fp}:\n{dup}")
    return df


def _baca_sheet_bulanan(cfg: dict, sheet: str, nama_nilai: str,
                        daftar: pd.DataFrame = None) -> pd.DataFrame:
    """Baca satu sheet berformat lebar (id, nama_bendungan, Jan..Des) dari file
    BON/RTOW/neraca air -> format panjang [kode_bendungan, bulan, <nama_nilai>].

    Kunci relasi = kolom 'id' (id database) yang dipetakan ke kode_bendungan
    lewat daftar_bendungan.csv; nama_bendungan resmi diambil dari daftar.
    Baris dengan id kosong/tak dikenal dilewati.
    """
    daftar = daftar if daftar is not None else baca_daftar_bendungan(cfg)
    fp = path_root(cfg["bon"]["file_bon"])
    df = pd.read_excel(fp, sheet_name=sheet)
    df.columns = [str(c).strip() for c in df.columns]
    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    df = df.dropna(subset=["id"])
    df["id_db"] = df["id"].astype(int)
    # nama resmi diambil dari daftar; kolom nama pada sheet dibuang agar tidak
    # bentrok saat merge
    df = df.drop(columns=[c for c in ("nama_bendungan",) if c in df.columns])
    df = df.merge(daftar[["id_db", "kode_bendungan", "nama_bendungan"]],
                  on="id_db", how="inner")
    df = df.melt(id_vars=["kode_bendungan", "nama_bendungan"],
                 value_vars=[b for b in BULAN_MAP if b in df.columns],
                 var_name="bulan_nama", value_name=nama_nilai)
    df["bulan"] = df["bulan_nama"].map(BULAN_MAP)
    df[nama_nilai] = pd.to_numeric(df[nama_nilai], errors="coerce")
    return (df[["kode_bendungan", "nama_bendungan", "bulan", nama_nilai]]
            .sort_values(["kode_bendungan", "bulan"]).reset_index(drop=True))


def baca_rtow(cfg: dict, daftar: pd.DataFrame = None) -> pd.DataFrame:
    """Elevasi RTOW (Rencana Tahunan Operasi Waduk) bulanan per bendungan.

    Return: [kode_bendungan, nama_bendungan, bulan(1-12), rtow]
    """
    sheet = cfg["bon"].get("sheet_rtow", "rtow")
    return _baca_sheet_bulanan(cfg, sheet, "rtow", daftar)


def baca_neraca_air(cfg: dict, daftar: pd.DataFrame = None) -> pd.DataFrame:
    """Ketersediaan vs kebutuhan air bulanan (m3) -> status surplus/defisit.

    neraca_m3 = ketersediaan - kebutuhan;
    kebutuhan > ketersediaan -> Defisit, selain itu Surplus.
    Ketersediaan negatif (salah entri) dipotong ke 0.
    Return: [kode_bendungan, nama_bendungan, bulan, ketersediaan_m3,
             kebutuhan_m3, neraca_m3, status_neraca]
    """
    daftar = daftar if daftar is not None else baca_daftar_bendungan(cfg)
    ket = _baca_sheet_bulanan(cfg, cfg["bon"].get("sheet_ketersediaan",
                                                  "ketersediaan_air"),
                              "ketersediaan_m3", daftar)
    keb = _baca_sheet_bulanan(cfg, cfg["bon"].get("sheet_kebutuhan",
                                                  "kebutuhan_air"),
                              "kebutuhan_m3", daftar)
    n = ket.merge(keb, on=["kode_bendungan", "nama_bendungan", "bulan"],
                  how="outer")
    n["ketersediaan_m3"] = n["ketersediaan_m3"].clip(lower=0)
    n["neraca_m3"] = n["ketersediaan_m3"] - n["kebutuhan_m3"]
    n["status_neraca"] = pd.Series(
        pd.NA, index=n.index, dtype="object")
    lengkap = n["ketersediaan_m3"].notna() & n["kebutuhan_m3"].notna()
    n.loc[lengkap & (n["kebutuhan_m3"] > n["ketersediaan_m3"]),
          "status_neraca"] = "Defisit"
    n.loc[lengkap & (n["kebutuhan_m3"] <= n["ketersediaan_m3"]),
          "status_neraca"] = "Surplus"
    n["status_neraca"] = n["status_neraca"].fillna("Tanpa Data")
    return n.sort_values(["kode_bendungan", "bulan"]).reset_index(drop=True)


def baca_bon(cfg: dict, daftar: pd.DataFrame = None) -> pd.DataFrame:
    """Baca BON A/B dari file referensi (sheet lebar berkunci id database).

    Sel BON kosong diisi interpolasi linear antar-bulan per bendungan
    (lalu ffill/bfill untuk ujung deret); kolom 'bon_terisi_otomatis'
    menandai nilai hasil pengisian.
    Return: [kode_bendungan, nama_bendungan, bulan(1-12), bon_a, bon_b,
             bon_terisi_otomatis]
    """
    daftar = daftar if daftar is not None else baca_daftar_bendungan(cfg)
    hasil = []
    for kunci, kolom, baku in [("sheet_bon_a", "bon_a", "bon a"),
                               ("sheet_bon_b", "bon_b", "bon b")]:
        df = _baca_sheet_bulanan(cfg, cfg["bon"].get(kunci, baku), kolom, daftar)
        df = df.sort_values(["kode_bendungan", "bulan"])
        df[f"{kolom}_isna"] = df[kolom].isna()
        df[kolom] = (df.groupby("kode_bendungan")[kolom]
                       .transform(lambda s: s.interpolate("linear").ffill().bfill()))
        hasil.append(df)

    bon = hasil[0].merge(hasil[1],
                         on=["kode_bendungan", "nama_bendungan", "bulan"])
    bon["bon_terisi_otomatis"] = bon["bon_a_isna"] | bon["bon_b_isna"]
    return (bon.drop(columns=["bon_a_isna", "bon_b_isna"])
               .sort_values(["kode_bendungan", "bulan"]).reset_index(drop=True))
