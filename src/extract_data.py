"""Ekstraksi data dari PostgreSQL SINBAD (skema terverifikasi DBeaver 2026-07).

  rawdata.sinbad_bendungan_tma           : sync_date, id, "timestamp", tma, volume
  rawdata.sinbad_bendungan_inflow        : sync_date, id, "timestamp", inflow
  rawdata.sinbad_bendungan_outflow       : sync_date, id, "timestamp",
                                           outflow_turbin, outflow_abaku,
                                           outflow_aindustri, outflow_irigasi,
                                           outflow_limpas, outflow_pemeliharaan
  rawdata.sinbad_bendungan_storagecurve  : sync_date, periode, id, tma, volume

Identitas bendungan = kolom id (int4). Pemetaan ke kode BDxxx dari
daftar_bendungan.csv (strip_prefix atau kolom id_db).
Kolom "timestamp" dikutip ganda karena reserved word.
Output: parquet harian di data/raw/ dengan kolom kode_bendungan (BDxxx).
"""
import os
import sys
import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (load_config, path_root, baca_daftar_bendungan,
                       normalisasi_tanggal)


def buat_engine(cfg):
    d = cfg["database"]
    url = (f"postgresql+psycopg2://{d['user']}:{d['password']}"
           f"@{d['host']}:{d['port']}/{d['dbname']}")
    return create_engine(url, pool_pre_ping=True)


def precheck_freshness(engine, cfg, id_list):
    """Pre-check halt: deteksi pipeline ingestion stall sebelum training."""
    c, t = cfg["columns"], cfg["tables"]
    q = text(f'''
        SELECT MIN("{c["waktu"]}") AS min_w, MAX("{c["waktu"]}") AS max_w
        FROM {t["tma"]}
        WHERE {c["id_bendungan"]} = ANY(:ids)
    ''')
    with engine.connect() as conn:
        r = conn.execute(q, {"ids": id_list}).mappings().one()
    print(f"[PRECHECK] rentang data TMA: {r['min_w']} s.d. {r['max_w']}")
    batas = pd.Timestamp(cfg["periode"]["akhir"]) - pd.Timedelta(days=30)
    if r["max_w"] is None or pd.Timestamp(r["max_w"]) < batas:
        raise RuntimeError(
            f"Data TMA terbaru ({r['max_w']}) lebih tua dari {batas.date()}. "
            "Indikasi pipeline ingestion SINBAD stall -- cek upstream dulu.")


def extract_tma(engine, cfg, id_list):
    c, t, p = cfg["columns"], cfg["tables"], cfg["periode"]
    q = text(f'''
        SELECT {c["id_bendungan"]}      AS id_db,
               "{c["waktu"]}"::date     AS tanggal,
               AVG({c["tma"]})          AS tma
        FROM {t["tma"]}
        WHERE {c["id_bendungan"]} = ANY(:ids)
          AND "{c["waktu"]}" BETWEEN :awal AND :akhir
          AND {c["tma"]} IS NOT NULL
        GROUP BY 1, 2 ORDER BY 1, 2
    ''')
    return pd.read_sql(q, engine, params={
        "ids": id_list, "awal": p["awal"], "akhir": p["akhir"]})


def extract_outflow(engine, cfg, id_list):
    """Outflow total = jumlah 6 komponen (m3/detik), NULL dianggap 0."""
    c, t, p = cfg["columns"], cfg["tables"], cfg["periode"]
    komponen = " + ".join(f"COALESCE({k}, 0)" for k in c["outflow_komponen"])
    q = text(f'''
        SELECT {c["id_bendungan"]}      AS id_db,
               "{c["waktu"]}"::date     AS tanggal,
               AVG({komponen})          AS outflow
        FROM {t["outflow"]}
        WHERE {c["id_bendungan"]} = ANY(:ids)
          AND "{c["waktu"]}" BETWEEN :awal AND :akhir
        GROUP BY 1, 2 ORDER BY 1, 2
    ''')
    return pd.read_sql(q, engine, params={
        "ids": id_list, "awal": p["awal"], "akhir": p["akhir"]})


def extract_inflow(engine, cfg, id_list):
    """Inflow tercatat (m3/detik) -- pembanding hasil neraca air."""
    c, t, p = cfg["columns"], cfg["tables"], cfg["periode"]
    q = text(f'''
        SELECT {c["id_bendungan"]}      AS id_db,
               "{c["waktu"]}"::date     AS tanggal,
               AVG({c["inflow"]})       AS inflow_tercatat
        FROM {t["inflow"]}
        WHERE {c["id_bendungan"]} = ANY(:ids)
          AND "{c["waktu"]}" BETWEEN :awal AND :akhir
        GROUP BY 1, 2 ORDER BY 1, 2
    ''')
    return pd.read_sql(q, engine, params={
        "ids": id_list, "awal": p["awal"], "akhir": p["akhir"]})


def extract_storagecurve(engine, cfg, id_list):
    """Storage curve: elevasi = kolom 'tma'. Jika ada > 1 versi periode per
    bendungan, dipakai periode dengan sync_date terbaru."""
    c, t = cfg["columns"], cfg["tables"]
    q = text(f'''
        SELECT {c["id_bendungan"]}      AS id_db,
               {c["periode_curve"]}     AS periode,
               MAX({c["sync_curve"]})   AS sync_terakhir,
               {c["elevasi_curve"]}     AS elevasi,
               {c["volume_curve"]}      AS volume
        FROM {t["storage_curve"]}
        WHERE {c["id_bendungan"]} = ANY(:ids)
        GROUP BY 1, 2, 4, 5 ORDER BY 1, 4
    ''')
    df = pd.read_sql(q, engine, params={"ids": id_list})
    if df.empty:
        return df[["id_db", "elevasi", "volume"]]
    # pilih periode terbaru per bendungan
    terbaru = (df.groupby(["id_db", "periode"])["sync_terakhir"].max()
                 .reset_index()
                 .sort_values("sync_terakhir")
                 .drop_duplicates("id_db", keep="last")[["id_db", "periode"]])
    df = df.merge(terbaru, on=["id_db", "periode"])
    return (df[["id_db", "elevasi", "volume"]]
            .drop_duplicates(["id_db", "elevasi"])
            .sort_values(["id_db", "elevasi"]).reset_index(drop=True))


def main():
    cfg = load_config()
    daftar = baca_daftar_bendungan(cfg)
    id_list = daftar["id_db"].tolist()
    peta = daftar.set_index("id_db")["kode_bendungan"]
    print(f"Ekstraksi {len(id_list)} bendungan (sesuai data BON A/B), "
          f"id {min(id_list)}..{max(id_list)}")

    engine = buat_engine(cfg)
    precheck_freshness(engine, cfg, id_list)

    out = path_root("data", "raw")
    os.makedirs(out, exist_ok=True)
    for nama, fn in [("tma", extract_tma), ("outflow", extract_outflow),
                     ("inflow", extract_inflow),
                     ("storagecurve", extract_storagecurve)]:
        df = fn(engine, cfg, id_list)
        df["kode_bendungan"] = df["id_db"].map(peta)
        df = df.drop(columns="id_db")
        # '::date' pada SQL -> psycopg2 mengembalikan datetime.date (dtype
        # object); diseragamkan ke datetime64 agar merge antar-tahap tidak gagal
        df = normalisasi_tanggal(df)
        # bendungan tanpa data sama sekali
        kosong = set(peta.values) - set(df["kode_bendungan"].unique())
        if kosong:
            print(f"  [{nama}] tanpa data: {len(kosong)} bendungan "
                  f"(contoh: {sorted(kosong)[:5]})")
        fp = os.path.join(out, f"{nama}_harian.parquet")
        df.to_parquet(fp, index=False)
        print(f"OK  {fp}  ({len(df):,} baris)")


if __name__ == "__main__":
    main()
