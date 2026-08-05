"""Dashboard Streamlit — Prediksi TMA Bulanan Agustus-Desember 2026 (LSTM).

Jalankan:  streamlit run app.py
Prasyarat: pipeline sudah dijalankan (python run_pipeline.py) sehingga
           output/prediksi/prediksi_tma_2026.parquet tersedia.

Menu:
  🏜️ Siaga Kekeringan        - ringkasan visual nasional + akurasi & metode
  📈 Detail Bendungan        - grafik sandingan + kecukupan air + unduh per bendungan
  🚨 Pemantauan Agustus 2026 - tabel pemantauan bulan fokus untuk semua bendungan
  🗂️ Rekap & Di bawah BON B  - rekap nasional, per balai, daftar kritis
  🧩 Data Belum Cocok        - bendungan yang datanya belum terhubung antar sumber
  ⬇️ Unduh Laporan           - Excel + ZIP grafik JPG (semua / hanya di bawah BON B)
"""
import io
import os
import sys
import time
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.utils import (load_config, path_root, baca_daftar_bendungan,
                       baca_neraca_air, baca_sifat_musim, daftar_periode)
import src.laporan as lap
import src.kecukupan as kc
from src.laporan import KRITIS, WASPADA, NORMAL, TANPA_BON, BULAN_ID, WARNA
from src.evaluasi import METODE_INFO, METODE_LSTM, ringkas as ringkas_evaluasi

st.set_page_config(page_title="DDEWOMS — PMB", page_icon="🌊",
                   layout="wide")

# latar pastel untuk sel tabel
WARNA_STATUS = {KRITIS: "#F8CBCB", WASPADA: "#FFEB9C",
                NORMAL: "#C6EFCE", TANPA_BON: "#E4E4E4",
                "Defisit": "#F8CBCB", "Surplus": "#C6EFCE",
                "Cukup": "#C6EFCE", "Belum cukup": "#F8CBCB",
                "Tanpa Data": "#E4E4E4"}
# isi tegas untuk mark grafik (selalu berpasangan dengan label + angka)
WARNA_GRAFIK = {KRITIS: "#C62828", WASPADA: "#D98E04",
                NORMAL: "#2E7D32", TANPA_BON: "#9E9E9E",
                "Defisit": "#C62828", "Surplus": "#2A9D8F",
                "Cukup": "#2E7D32", "Belum cukup": "#C62828"}
URUT_STATUS = [KRITIS, WASPADA, NORMAL, TANPA_BON]
# badge rekomendasi operasional
WARNA_BADGE = {"hijau": "#C6EFCE", "amber": "#FFEB9C",
               "merah": "#F8CBCB", "abu": "#E4E4E4"}


# ------------------------------------------------------------------ data
@st.cache_data(show_spinner="Memuat data prediksi…")
def muat_data():
    cfg = load_config()
    fp_pred = path_root(cfg["output"]["dir_prediksi"], "prediksi_tma_2026.parquet")
    if not os.path.exists(fp_pred):
        st.error("❌ Hasil prediksi belum ada: `output/prediksi/"
                 "prediksi_tma_2026.parquet` tidak ditemukan.\n\n"
                 "Jalankan pipeline dulu: `python run_pipeline.py` "
                 "(butuh akses jaringan internal PU untuk mode database), "
                 "lalu muat ulang halaman ini.")
        st.stop()
    df = pd.read_parquet(fp_pred)
    fp_hist = path_root("data", "processed", "tma_periode.parquet")
    hist = pd.read_parquet(fp_hist) if os.path.exists(fp_hist) else pd.DataFrame()
    fp_qc = path_root("data", "processed", "rekap_qc.xlsx")
    rekap_qc = pd.read_excel(fp_qc) if os.path.exists(fp_qc) else pd.DataFrame()
    daftar = baca_daftar_bendungan(cfg)
    df = lap.gabung_wilayah(df, daftar)
    neraca = lap.gabung_wilayah(baca_neraca_air(cfg, daftar), daftar)
    sifat_musim = baca_sifat_musim(cfg, daftar)
    fp_ev = path_root(cfg["output"]["dir_prediksi"], "evaluasi_metode.parquet")
    evaluasi = pd.read_parquet(fp_ev) if os.path.exists(fp_ev) else pd.DataFrame()
    # sumber baru: workbook RTOW periodik + elevasi kekeringan + storage curve
    kunci = daftar[["id_db", "kode_bendungan"]].drop_duplicates()
    rtow_p = kc.baca_rtow_periodik()
    if not rtow_p.empty:
        rtow_p = rtow_p.merge(kunci, on="id_db", how="inner")
    kek = kc.baca_elevasi_kekeringan()
    if not kek.empty:
        kek = kek.merge(kunci, on="id_db", how="inner")
    kurva_sc = kc.baca_storage_curve()
    return (df, hist, daftar, rekap_qc, neraca, evaluasi, sifat_musim,
            rtow_p, kek, kurva_sc)


@st.cache_data(show_spinner="Menyusun berkas Excel…")
def buat_excel(df, daftar, hist, hanya_kritis, bulan_fokus, filter_teks, waktu,
               rekap_qc=None, neraca=None):
    return lap.excel_laporan(df, daftar, hist, hanya_kritis=hanya_kritis,
                             bulan_fokus=bulan_fokus, filter_teks=filter_teks,
                             waktu=waktu, rekap_qc=rekap_qc, neraca=neraca)


@st.cache_data(show_spinner="Merender grafik JPG…")
def buat_jpg(g, hist, tampil_hist):
    return lap.fig_ke_jpg(lap.fig_bendungan(g, hist, tampil_hist))


@st.cache_data(show_spinner="Menyusun laporan PDF…")
def buat_pdf(df, neraca, evaluasi, filter_teks, waktu):
    return lap.pdf_laporan(df, neraca, evaluasi,
                           filter_teks=filter_teks, waktu=waktu)


def stamp():
    return dt.datetime.now().strftime("%Y%m%d_%H%M")


def waktu_panjang():
    return dt.datetime.now().strftime("%d %B %Y %H:%M")


def warnai_status(d: pd.DataFrame, kolom: str):
    """Styler pewarnaan baris tabel berdasarkan kolom status."""
    if kolom not in d.columns:
        return d

    def gaya(row):
        bg = WARNA_STATUS.get(row[kolom], "")
        return [f"background-color: {bg}" if bg else "" for _ in row]

    return d.style.apply(gaya, axis=1)


(df, hist, daftar, rekap_qc, neraca, evaluasi, sifat_musim,
 rtow_p, kek, kurva_sc) = muat_data()

# ------------------------------------------------------------------ header
st.markdown(
    f"<h2 style='color:{WARNA['navy']};margin-bottom:0'>🌊 Dams Drought Early "
    f"Warning and Operational Mitigation System (DDEWOMS)</h2>"
    f"<p style='color:{WARNA['biru']};margin-top:2px'>Pusat Monitoring Bendungan — LSTM per periode "
    f"10/15-harian (format RTOW) + sifat musim · prediksi Agustus–Desember 2026 · "
    f"sandingan Bon A / Bon B / RTOW · kecukupan air RTOW periodik</p>",
    unsafe_allow_html=True)

# ------------------------------------------------------------------ sidebar
st.sidebar.header("Menu")
menu = st.sidebar.radio(
    "Pilih halaman",
    ["🏜️ Siaga Kekeringan", "📈 Detail Bendungan", "🚨 Pemantauan Agustus 2026",
     "🗂️ Rekap & Di bawah BON B", "🧩 Data Belum Cocok", "⬇️ Unduh Laporan"],
    label_visibility="collapsed")

st.sidebar.header("Filter Wilayah")
pulau_semua = sorted(df["nama_pulau"].dropna().unique())
pulau = st.sidebar.multiselect("Pulau / wilayah", pulau_semua, default=pulau_semua)
df_p = df[df["nama_pulau"].isin(pulau)] if pulau else df

balai_semua = sorted(df_p["nama_balai"].dropna().unique())
# key mengikuti pilihan pulau -> pilihan balai otomatis di-reset saat pulau diubah
balai = st.sidebar.multiselect("Balai (BBWS/BWS)", balai_semua,
                               default=balai_semua,
                               key="balai_" + "|".join(pulau))
dff = df_p[df_p["nama_balai"].isin(balai)] if balai else df_p

st.sidebar.header("Filter Status Prediksi")
status_pilih = st.sidebar.multiselect(
    "Status terburuk prediksi", [KRITIS, WASPADA, NORMAL, TANPA_BON],
    default=[KRITIS, WASPADA, NORMAL, TANPA_BON])
if status_pilih:
    st_bdg = (dff[dff["jenis"] == "prediksi"]
              .groupby("kode_bendungan")["status_bon"].apply(lap.status_terburuk))
    dff = dff[dff["kode_bendungan"].isin(st_bdg[st_bdg.isin(status_pilih)].index)]

n_bdg = dff["kode_bendungan"].nunique()
filter_teks = (f"Pulau: {', '.join(pulau) if len(pulau) != len(pulau_semua) else 'semua'}"
               f" · Balai: {len(balai)} dari {len(balai_semua)}"
               f" · {n_bdg} bendungan")
st.sidebar.caption(f"**{n_bdg}** bendungan lolos filter "
                   f"(dari {df['kode_bendungan'].nunique()} total).")

if n_bdg == 0:
    st.warning("Tidak ada bendungan yang cocok dengan filter. Longgarkan filter "
               "pada panel kiri.")
    st.stop()

pred_dff = dff[dff["jenis"] == "prediksi"]
kode_kritis = sorted(pred_dff[pred_dff["status_bon"] == KRITIS]["kode_bendungan"].unique())
neraca_dff = neraca[neraca["kode_bendungan"].isin(dff["kode_bendungan"].unique())]


# ================================================================== RTOW periodik
# Helper sumber data baru (workbook per periode + storage curve). Nilai kosong
# DIBIARKAN kosong — grafik/metrik menampilkan placeholder, tanpa interpolasi.
KOLOM_PERIODIK = ("bon_a", "bon_b", "rtow", "ketersediaan_m3s", "kebutuhan_m3s")


def ambil_periodik(kode: str, fmt: str):
    """Data RTOW periodik 1 bendungan.

    Return (pb, kek_b, kurva_b): pb = DataFrame berindeks label periode penuh
    setahun (kolom KOLOM_PERIODIK, NaN bila belum tersedia); kek_b = baris
    elevasi kekeringan/dasar waduk (None bila tidak ada); kurva_b = storage
    curve bendungan (DataFrame kosong bila tidak ada).
    """
    urut = daftar_periode(fmt, 1, 12)
    b = (rtow_p[rtow_p["kode_bendungan"] == kode]
         if not rtow_p.empty else pd.DataFrame())
    pb = (b.drop_duplicates("periode").set_index("periode").reindex(urut)
          if not b.empty else pd.DataFrame(index=urut))
    for c in KOLOM_PERIODIK:
        if c not in pb.columns:
            pb[c] = np.nan
    kek_b = None
    if not kek.empty:
        kb = kek[kek["kode_bendungan"] == kode]
        if not kb.empty:
            kek_b = kb.iloc[0]
    kurva_b = kc.kurva_bendungan(kurva_sc, kode)
    return pb, kek_b, kurva_b


def badge_html(teks: str, latar: str) -> str:
    return (f"<span style='background:{latar};color:#1A1A2E;padding:2px 12px;"
            f"border-radius:12px;font-weight:600;font-size:0.92em;"
            f"white-space:nowrap'>{teks}</span>")


def fig_sandingan(g: pd.DataFrame, tampil_hist: bool = False):
    """Grafik sandingan per periode (dipakai Detail Bendungan & kartu carousel).

    Bon A/Bon B/RTOW/ketersediaan/kebutuhan diambil PER PERIODE dari workbook
    RTOW periodik (bukan interpolasi bulanan). Ketersediaan & kebutuhan air
    (m3/s) dikonversi ke volume (x hari x 86400) lalu ke elevasi mdpl lewat
    interpolasi storage curve penuh. Ditambah garis flat elevasi kekeringan &
    elevasi dasar waduk (fallback kolom SINBAD). Data kosong -> garis tidak
    digambar (tanpa interpolasi pengganti).

    Return (fig, catatan): catatan = daftar pesan data yang belum tersedia.
    """
    kode = g["kode_bendungan"].iloc[0]
    nama = g["nama_bendungan"].iloc[0]
    fmt = (g["format_periode"].iloc[0]
           if "format_periode" in g.columns else "15 Harian")
    urut = daftar_periode(fmt, 1, 12)
    real = g[g["jenis"] == "realisasi"]
    pred = g[g["jenis"] == "prediksi"]
    pb, kek_b, kurva_b = ambil_periodik(kode, fmt)
    catatan = []

    fig = go.Figure()
    if tampil_hist and not hist.empty:
        h = hist[(hist["kode_bendungan"] == kode)
                 & (hist["tanggal"] < "2026-01-01")]
        pertama = True
        for thn, ht in h.groupby(h["tanggal"].dt.year):
            s = ht.set_index("periode")["tma"].reindex(urut)
            fig.add_trace(go.Scatter(
                x=urut, y=s.values, mode="lines",
                line=dict(color="#D5DDE5", width=1.2),
                name="Historis per tahun (2018–2025)", legendgroup="hist",
                showlegend=pertama, hovertemplate=f"{thn} · %{{x}}: %{{y:.2f}} mdpl"))
            pertama = False

    # --- garis per periode dari workbook RTOW periodik (sumber baru) ---
    if pb["rtow"].notna().any():
        fig.add_trace(go.Scatter(x=urut, y=pb["rtow"].values, mode="lines",
                                 line=dict(color=WARNA["biru"], width=2, dash="dot"),
                                 name="RTOW per periode (rencana operasi)"))
    else:
        catatan.append("Elevasi rencana RTOW per periode belum tersedia — "
                       "garis RTOW tidak ditampilkan.")
    if pb["bon_a"].notna().any():
        fig.add_trace(go.Scatter(x=urut, y=pb["bon_a"].values, mode="lines",
                                 line=dict(color=WARNA["amber"], width=2, dash="dashdot"),
                                 name="Bon A per periode"))
    if pb["bon_b"].notna().any():
        fig.add_trace(go.Scatter(x=urut, y=pb["bon_b"].values, mode="lines",
                                 line=dict(color=WARNA["coral"], width=2, dash="dashdot"),
                                 name="Bon B per periode"))
    if pb["bon_a"].isna().all() and pb["bon_b"].isna().all():
        catatan.append("Nilai Bon A/Bon B per periode belum tersedia — "
                       "garis Bon tidak ditampilkan.")

    # --- ketersediaan & kebutuhan air: m3/s -> volume -> elevasi ---
    ada_kk = (pb["ketersediaan_m3s"].notna().any()
              or pb["kebutuhan_m3s"].notna().any())
    if ada_kk and kurva_b.empty:
        catatan.append("Storage curve bendungan ini tidak tersedia — "
                       "ketersediaan/kebutuhan air tidak dapat dikonversi "
                       "ke elevasi.")
    elif ada_kk:
        for kol, label, warna in (
                ("ketersediaan_m3s", "Ketersediaan air (konversi elevasi)",
                 WARNA["teal"]),
                ("kebutuhan_m3s", "Kebutuhan air (konversi elevasi)",
                 "#6A1B9A")):
            s = pd.to_numeric(pb[kol], errors="coerce")
            if s.notna().any():
                vol = kc.debit_ke_volume(s, fmt)
                elev = kc.volume_ke_elevasi(kurva_b, vol.values)
                elev = np.where(np.isnan(vol.values), np.nan, elev)
                fig.add_trace(go.Scatter(
                    x=urut, y=elev, mode="lines",
                    line=dict(color=warna, width=1.8, dash="dash"),
                    name=label,
                    customdata=np.stack([s.values, vol.values / 1e6], axis=-1),
                    hovertemplate="%{x}: %{y:.2f} mdpl "
                                  "(%{customdata[0]:.3f} m³/s ≈ "
                                  "%{customdata[1]:.2f} juta m³)"
                                  f"<extra>{label}</extra>"))
    else:
        catatan.append("Data ketersediaan/kebutuhan air per periode belum "
                       "tersedia.")

    # --- garis flat elevasi kekeringan & dasar waduk (fallback SINBAD) ---
    if kek_b is not None and pd.notna(kek_b["elevasi_kekeringan"]):
        fig.add_trace(go.Scatter(
            x=urut, y=[float(kek_b["elevasi_kekeringan"])] * len(urut),
            mode="lines", line=dict(color="#AD1457", width=2, dash="longdash"),
            name=f"Elevasi kekeringan ({kek_b['sumber_kekeringan']})"))
    else:
        catatan.append("Elevasi kekeringan belum tersedia (RTOW maupun SINBAD).")
    if kek_b is not None and pd.notna(kek_b["elevasi_dasar"]):
        fig.add_trace(go.Scatter(
            x=urut, y=[float(kek_b["elevasi_dasar"])] * len(urut),
            mode="lines", line=dict(color="#6D4C41", width=2, dash="longdashdot"),
            name=f"Elevasi dasar waduk ({kek_b['sumber_dasar']})"))
    else:
        catatan.append("Elevasi dasar waduk belum tersedia (RTOW maupun SINBAD).")

    # --- realisasi & prediksi (metode LSTM TIDAK diubah — hanya tampilan) ---
    if not real.empty:
        fig.add_trace(go.Scatter(x=real["periode"], y=real["tma"],
                                 mode="lines+markers",
                                 line=dict(color=WARNA["hijau"], width=3),
                                 marker=dict(size=7), name="TMA Realisasi 2026"))
    if not pred.empty:
        x_pred = ([real["periode"].iloc[-1]] if not real.empty else []) + \
                 list(pred["periode"])
        y_pred = ([real["tma"].iloc[-1]] if not real.empty else []) + list(pred["tma"])
        fig.add_trace(go.Scatter(x=x_pred, y=y_pred, mode="lines+markers",
                                 line=dict(color=WARNA["merah"], width=3, dash="dash"),
                                 marker=dict(size=7, symbol="square"),
                                 name="TMA Prediksi LSTM (Agu–Des 2026)"))
        krit = pred[pred["status_bon"] == KRITIS]
        if not krit.empty:
            fig.add_trace(go.Scatter(
                x=krit["periode"], y=krit["tma"], mode="markers",
                marker=dict(size=14, symbol="circle-open",
                            line=dict(color=WARNA["merah"], width=2.5)),
                name="Periode di bawah BON B", hoverinfo="skip"))
        if not real.empty:
            fig.add_vline(x=real["periode"].iloc[-1],
                          line=dict(color="grey", dash="dot"))
            fig.add_annotation(x=real["periode"].iloc[-1], yref="paper", y=1.04,
                               text="realisasi | prediksi", showarrow=False,
                               font=dict(size=11, color="grey"))
    fig.update_layout(
        title=dict(text=f"Sandingan TMA per Periode {fmt} 2026 — Bendungan {nama}",
                   font=dict(size=18, color=WARNA["navy"])),
        yaxis_title="TMA (mdpl)", xaxis_title="Periode (bulan-periode)",
        hovermode="x unified", height=560,
        legend=dict(orientation="h", y=-0.22), plot_bgcolor="white")
    fig.update_xaxes(showgrid=True, gridcolor="#EEE", type="category",
                     categoryorder="array", categoryarray=urut,
                     tickangle=-90, tickfont=dict(size=9))
    fig.update_yaxes(showgrid=True, gridcolor="#EEE")
    return fig, catatan


def blok_status_zona(kode: str, fmt: str, g: pd.DataFrame, pb: pd.DataFrame):
    """Keputusan 3 zona memakai Bon A/Bon B PER PERIODE RTOW berjalan
    (tanggal berjalan; bila periode itu tanpa data -> periode terakhir yang
    tersedia). Return (status, label_periode, tma_terkini)."""
    real = g[g["jenis"] == "realisasi"]
    tma = float(real["tma"].iloc[-1]) if not real.empty else np.nan
    punya = pb.index[pb["bon_a"].notna() & pb["bon_b"].notna()]
    label = kc.periode_aktif(fmt, tersedia=list(punya) if len(punya) else None)
    ba = pb.loc[label, "bon_a"] if label in pb.index else np.nan
    bb = pb.loc[label, "bon_b"] if label in pb.index else np.nan
    status = kc.status_tiga_zona(tma, ba, bb)
    ikon = {NORMAL: "✅", WASPADA: "⚠️", KRITIS: "🚨", TANPA_BON: "➖"}[status]
    if status == TANPA_BON:
        st.markdown(
            f"**Keputusan 3 zona — periode berjalan {label}:** "
            + badge_html(f"{ikon} {TANPA_BON}", WARNA_STATUS[TANPA_BON])
            + " <small>data Bon A/Bon B per periode atau TMA terkini belum "
              "tersedia — keputusan tidak dipaksakan.</small>",
            unsafe_allow_html=True)
    else:
        st.markdown(
            f"**Keputusan 3 zona — periode berjalan {label}:** "
            + badge_html(f"{ikon} {status}", WARNA_STATUS[status])
            + f" <small>TMA terkini {tma:.2f} mdpl · Bon A {ba:.2f} mdpl · "
              f"Bon B {bb:.2f} mdpl</small>",
            unsafe_allow_html=True)
    return status, label, tma


def blok_kecukupan(kode: str, fmt: str, pb: pd.DataFrame):
    """Blok Kecukupan Air periode berjalan: volume ketersediaan vs kebutuhan
    (konversi m3/s -> m3), deviasi (m3 & %), status Cukup/Belum cukup.
    Return (kecukupan_dict_atau_None, label_periode)."""
    punya = pb.index[pb["ketersediaan_m3s"].notna() & pb["kebutuhan_m3s"].notna()]
    label = kc.periode_aktif(fmt, tersedia=list(punya) if len(punya) else None)
    baris = pb.loc[label] if label in pb.index else None
    kck = (kc.hitung_kecukupan(baris["ketersediaan_m3s"],
                               baris["kebutuhan_m3s"], fmt)
           if baris is not None else None)
    st.markdown(f"**💧 Kecukupan Air — periode berjalan {label}**")
    if kck is None:
        st.info("ℹ️ Data ketersediaan/kebutuhan air RTOW periode ini belum "
                "tersedia — kecukupan air tidak dihitung.")
        return None, label
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ketersediaan air", f"{kck['ketersediaan_m3']:,.0f} m³",
              help="Konversi dari m³/s × jumlah hari periode × 86.400")
    c2.metric("Kebutuhan air", f"{kck['kebutuhan_m3']:,.0f} m³",
              help="Konversi dari m³/s × jumlah hari periode × 86.400")
    teks_pct = ("—" if pd.isna(kck["deviasi_pct"])
                else f"{kck['deviasi_pct']:+.1f}%")
    c3.metric("Deviasi (ketersediaan − kebutuhan)",
              f"{kck['deviasi_m3']:+,.0f} m³", delta=teks_pct,
              delta_color="normal" if kck["deviasi_m3"] >= 0 else "inverse")
    ikon = "✅" if kck["status"] == "Cukup" else "🔴"
    c4.markdown("Status<br>" + badge_html(f"{ikon} {kck['status']}",
                                          WARNA_STATUS[kck["status"]]),
                unsafe_allow_html=True)
    return kck, label


def html_infografis(nama, kode, balai, label, status_zona, kck, rek,
                    k, ketahanan, waktu) -> str:
    """Infografis PMB (layout kartu, badge status, nilai kunci) — pratinjau
    di layar & dapat diunduh sebagai HTML mandiri."""
    warna_badge = {"hijau": "#2E7D32", "amber": "#D98E04",
                   "merah": "#C62828", "abu": "#9E9E9E"}[rek["warna"]]
    if kck:
        baris_kck = (f"<div class='n'><b>{kck['ketersediaan_m3']:,.0f}</b> m³ "
                     f"ketersediaan</div>"
                     f"<div class='n'><b>{kck['kebutuhan_m3']:,.0f}</b> m³ "
                     f"kebutuhan</div>"
                     f"<div class='n'><b>{kck['deviasi_m3']:+,.0f}</b> m³ "
                     f"({'—' if pd.isna(kck['deviasi_pct']) else format(kck['deviasi_pct'], '+.1f') + '%'}) deviasi</div>")
    else:
        baris_kck = "<div class='n'>Data kecukupan air belum tersedia</div>"
    if ketahanan is not None and not (isinstance(ketahanan, float)
                                      and np.isnan(ketahanan)):
        teks_tahan = ("∞ (tanpa pengurasan bersih)" if np.isinf(ketahanan)
                      else f"~{ketahanan:.0f} hari")
    else:
        teks_tahan = "—"
    langkah_html = "".join(
        f"<li>{ik} {tx.replace('**', '')}</li>" for ik, tx in rek["langkah"])
    return f"""
<div style="max-width:640px;font-family:Segoe UI,Arial,sans-serif;border:2px solid {WARNA['navy']};border-radius:14px;overflow:hidden;background:#fff">
  <div style="background:{WARNA['navy']};color:#fff;padding:14px 20px">
    <div style="font-size:0.8em;letter-spacing:1px">PUSAT MONITORING BENDUNGAN · DDEWOMS</div>
    <div style="font-size:1.3em;font-weight:700">🧭 Rekomendasi Operasional — {nama} ({kode})</div>
    <div style="font-size:0.85em">{balai} · periode {label} · {waktu}</div>
  </div>
  <div style="padding:16px 20px">
    <div style="margin-bottom:10px">
      <span style="background:{warna_badge};color:#fff;padding:4px 14px;border-radius:14px;font-weight:700">{rek['badge']}</span>
      <span style="background:{WARNA_STATUS.get(status_zona, '#E4E4E4')};padding:4px 14px;border-radius:14px;font-weight:600;margin-left:6px">{status_zona}</span>
    </div>
    <div style="display:flex;gap:14px;flex-wrap:wrap;margin-bottom:10px">
      <div class='n' style="font-size:0.95em">Faktor K rekomendasi: <b>{('—' if pd.isna(rek['k_rekomendasi']) else format(rek['k_rekomendasi'], '.2f'))}</b></div>
      <div class='n' style="font-size:0.95em">Ketahanan air: <b>{teks_tahan}</b></div>
    </div>
    <div style="display:flex;gap:14px;flex-wrap:wrap;font-size:0.95em;margin-bottom:10px">{baris_kck}</div>
    <ol style="font-size:0.9em;line-height:1.5;padding-left:18px;margin:0">{langkah_html}</ol>
  </div>
  <div style="background:#F0F4F8;color:{WARNA['navy']};padding:8px 20px;font-size:0.75em">
    Subdit OP Bendungan dan Danau · Dit. Bina OP · Ditjen SDA — rekomendasi rule-based Skenario 3 di atas data RTOW periodik (bukan keluaran model LSTM)
  </div>
</div>"""


def blok_rekomendasi(kode: str, nama: str, balai: str, fmt: str,
                     g: pd.DataFrame, pb: pd.DataFrame, kek_b, kurva_b,
                     status_zona: str, label: str, tma: float):
    """Box Rekomendasi Operasional (Skenario 3, rule-based — tanpa LSTM)."""
    with st.container(border=True):
        st.markdown("##### 🧭 Rekomendasi Operasional — Pusat Monitoring "
                    "Bendungan")
        if pb[list(KOLOM_PERIODIK)].isna().all().all():
            st.info("ℹ️ Data RTOW belum tersedia untuk bendungan ini — "
                    "rekomendasi tidak dihitung.")
            return
        minta = st.toggle(
            "Permintaan pelayanan resmi (UPB/UPI, Kementan, atau pihak lain — "
            "surat/permohonan resmi)",
            key=f"minta_{kode}",
            help="Default 'Tidak' bila belum ada data. Penanda disimpan "
                 "selama sesi dashboard terbuka.")
        baris = pb.loc[label] if label in pb.index else None
        ket = baris["ketersediaan_m3s"] if baris is not None else np.nan
        keb = baris["kebutuhan_m3s"] if baris is not None else np.nan
        bon_a = baris["bon_a"] if baris is not None else np.nan
        bon_b = baris["bon_b"] if baris is not None else np.nan
        elev_kek = (kek_b["elevasi_kekeringan"]
                    if kek_b is not None else np.nan)
        k = kc.faktor_k(ket, keb)
        kck = kc.hitung_kecukupan(ket, keb, fmt)
        ketahanan = kc.estimasi_ketahanan_hari(kurva_b, tma, elev_kek, ket, keb)
        rek = kc.rekomendasi_skenario3(
            ada_permintaan_resmi=bool(minta), k=k, tma=tma, bon_a=bon_a,
            bon_b=bon_b, elevasi_kekeringan=elev_kek, kecukupan=kck,
            ketahanan_hari=ketahanan)

        st.markdown(badge_html(rek["badge"], WARNA_BADGE[rek["warna"]]),
                    unsafe_allow_html=True)
        for ikon, teks in rek["langkah"]:
            st.markdown(f"{ikon} {teks}")

        n1, n2, n3 = st.columns(3)
        n1.metric("Faktor K rekomendasi",
                  "—" if pd.isna(rek["k_rekomendasi"])
                  else f"{rek['k_rekomendasi']:.2f}")
        if kck:
            teks_pct = ("—" if pd.isna(kck["deviasi_pct"])
                        else f"{kck['deviasi_pct']:+.1f}%")
            n2.metric("Deviasi kecukupan air",
                      f"{kck['deviasi_m3']:+,.0f} m³", delta=teks_pct,
                      delta_color="normal" if kck["deviasi_m3"] >= 0
                      else "inverse")
        else:
            n2.metric("Deviasi kecukupan air", "—")
        if ketahanan is not None and not (isinstance(ketahanan, float)
                                          and np.isnan(ketahanan)):
            n3.metric("Estimasi ketahanan air",
                      "∞" if np.isinf(ketahanan) else f"~{ketahanan:.0f} hari")
        else:
            n3.metric("Estimasi ketahanan air", "—")

        if st.button("🖼️ Susun infografis", key=f"btn_info_{kode}"):
            st.session_state[f"tampil_info_{kode}"] = True
        if st.session_state.get(f"tampil_info_{kode}"):
            html = html_infografis(nama, kode, balai, label, status_zona,
                                   kck, rek, k, ketahanan, waktu_panjang())
            st.markdown(html, unsafe_allow_html=True)
            st.download_button(
                "⬇️ Unduh infografis (HTML, siap dibagikan)",
                html.encode("utf-8"),
                file_name=f"infografis_pmb_{kode}_{stamp()}.html",
                mime="text/html", key=f"dl_info_{kode}")


def kartu_bendungan(kode: str):
    """Kartu pemantauan mirip Detail Bendungan (dipakai carousel depan)."""
    g = df[df["kode_bendungan"] == kode].sort_values("tanggal")
    if g.empty:
        st.info(f"Data bendungan {kode} tidak ditemukan.")
        return
    nama = g["nama_bendungan"].iloc[0]
    balai = g["nama_balai"].iloc[0]
    fmt = (g["format_periode"].iloc[0]
           if "format_periode" in g.columns else "15 Harian")
    st.markdown(f"##### 🏞️ {nama} ({kode}) — {balai} · {g['nama_pulau'].iloc[0]}")
    fig, catatan = fig_sandingan(g, tampil_hist=False)
    if catatan:
        st.caption("ℹ️ Data belum tersedia: " + " · ".join(catatan))
    st.plotly_chart(fig, use_container_width=True, key=f"car_fig_{kode}")
    pb, kek_b, kurva_b = ambil_periodik(kode, fmt)
    status_zona, label, tma = blok_status_zona(kode, fmt, g, pb)
    blok_kecukupan(kode, fmt, pb)
    blok_rekomendasi(kode, nama, balai, fmt, g, pb, kek_b, kurva_b,
                     status_zona, label, tma)


# ============================================================ 0. SIAGA KEKERINGAN
if menu == "🏜️ Siaga Kekeringan":
    st.subheader("🏜️ Dashboard Siaga Kekeringan — Musim Kering Agustus–Desember 2026")

    # -- agregat per bendungan: status BON terburuk PER BULAN (worst periode
    #    dalam bulan) agar bendungan format 10-harian & 15-harian sebanding
    sbm = lap.status_per_bulan(pred_dff)
    per_bdg = (pred_dff.groupby(["kode_bendungan", "nama_bendungan",
                                 "nama_balai", "nama_pulau"])
               .agg(status_terburuk=("status_bon", lap.status_terburuk),
                    tma_min=("tma", "min"))
               .reset_index())
    bulan_status = (sbm.assign(k=sbm["status_bon"] == KRITIS,
                               w=sbm["status_bon"] == WASPADA)
                    .groupby("kode_bendungan")
                    .agg(bulan_kritis=("k", "sum"), bulan_waspada=("w", "sum"))
                    .astype(int))
    per_bdg = per_bdg.merge(bulan_status, on="kode_bendungan", how="left")
    per_bdg[["bulan_kritis", "bulan_waspada"]] = (
        per_bdg[["bulan_kritis", "bulan_waspada"]].fillna(0).astype(int))
    if "rtow" in pred_dff.columns:
        rtow_bdg = (pred_dff.assign(bawah=pred_dff["tma"] < pred_dff["rtow"],
                                    bulan=pred_dff["tanggal"].dt.month)
                    .groupby(["kode_bendungan", "bulan"])["bawah"].any()
                    .groupby("kode_bendungan").sum().astype(int)
                    .rename("bulan_dibawah_rtow"))
        per_bdg = per_bdg.merge(rtow_bdg, on="kode_bendungan", how="left")
        per_bdg["bulan_dibawah_rtow"] = per_bdg["bulan_dibawah_rtow"].fillna(0).astype(int)
    else:
        per_bdg["bulan_dibawah_rtow"] = 0

    n_kering = neraca_dff[neraca_dff["bulan"] >= 8]
    def_bdg = (n_kering.assign(defisit=n_kering["status_neraca"] == "Defisit")
               .groupby("kode_bendungan")
               .agg(bulan_defisit_agu_des=("defisit", "sum"),
                    defisit_terbesar_juta_m3=("neraca_m3",
                                              lambda s: round(max(0.0, -s.min()) / 1e6, 2)))
               .reset_index())
    def_bdg["bulan_defisit_agu_des"] = def_bdg["bulan_defisit_agu_des"].astype(int)
    per_bdg = per_bdg.merge(def_bdg, on="kode_bendungan", how="left")
    per_bdg[["bulan_defisit_agu_des", "defisit_terbesar_juta_m3"]] = (
        per_bdg[["bulan_defisit_agu_des", "defisit_terbesar_juta_m3"]].fillna(0))

    per_bdg["skor_prioritas"] = (3 * per_bdg["bulan_kritis"]
                                 + 1 * per_bdg["bulan_waspada"]
                                 + 2 * per_bdg["bulan_defisit_agu_des"]
                                 + 1 * per_bdg["bulan_dibawah_rtow"]).astype(int)

    # -- KPI (rekapitulasi 3 kategori BON + RTOW)
    m0, m1, m2, m3, m4 = st.columns(5)
    m0.metric("Bendungan dipantau", len(per_bdg))
    m1.metric("✅ Di atas BON A", int((per_bdg["status_terburuk"] == NORMAL).sum()))
    m2.metric("⚠️ Di antara BON A–B", int((per_bdg["status_terburuk"] == WASPADA).sum()))
    m3.metric("🚨 Di bawah BON B", int((per_bdg["status_terburuk"] == KRITIS).sum()))
    m4.metric("📉 Di bawah RTOW", int((per_bdg["bulan_dibawah_rtow"] > 0).sum()))

    # -- baris 1: donut status + batang per bulan
    c_kiri, c_kanan = st.columns([2, 3])
    with c_kiri:
        cnt = (per_bdg["status_terburuk"].value_counts()
               .reindex(URUT_STATUS).dropna().astype(int))
        fig_p = go.Figure(go.Pie(
            labels=cnt.index, values=cnt.values, hole=0.55, sort=False,
            marker=dict(colors=[WARNA_GRAFIK[s] for s in cnt.index],
                        line=dict(color="white", width=2)),
            texttemplate="%{label}<br>%{value} (%{percent})",
            textposition="outside"))
        fig_p.update_layout(
            title=dict(text="Status terburuk prediksi Agu–Des 2026",
                       font=dict(size=14, color=WARNA["navy"])),
            height=380, showlegend=False,
            annotations=[dict(text=f"{len(per_bdg)}<br>bendungan",
                              x=0.5, y=0.5, showarrow=False, font=dict(size=15))])
        st.plotly_chart(fig_p, use_container_width=True)
    with c_kanan:
        per_bulan = (sbm.groupby(["bulan", "status_bon"])
                     .size().unstack(fill_value=0)
                     .reindex(columns=URUT_STATUS, fill_value=0))
        fig_b = go.Figure()
        for s in URUT_STATUS:
            if s not in per_bulan.columns or per_bulan[s].sum() == 0:
                continue
            fig_b.add_trace(go.Bar(
                x=[BULAN_ID[b] for b in per_bulan.index], y=per_bulan[s],
                name=s, marker_color=WARNA_GRAFIK[s],
                text=per_bulan[s], textposition="inside"))
        fig_b.update_layout(
            barmode="stack", height=380, plot_bgcolor="white",
            title=dict(text="Jumlah bendungan per status per bulan "
                            "(periode terburuk dalam bulan)",
                       font=dict(size=14, color=WARNA["navy"])),
            yaxis_title="Jumlah bendungan", legend=dict(orientation="h", y=-0.18))
        fig_b.update_yaxes(showgrid=True, gridcolor="#EEE")
        st.plotly_chart(fig_b, use_container_width=True)

    # -- baris 2: per balai + kalender kritis
    #    (dua kartu neraca lama — jumlah bendungan defisit vs surplus per bulan
    #     dan 15 defisit air bulanan terbesar Agu–Des — DIHAPUS; bagian
    #     Kecukupan Air kini berbasis RTOW periodik pada kartu pemantauan)
    c_kiri, c_kanan = st.columns(2)
    with c_kiri:
        per_balai = (per_bdg.groupby("nama_balai")["status_terburuk"]
                     .value_counts().unstack(fill_value=0)
                     .reindex(columns=URUT_STATUS, fill_value=0))
        per_balai["_kritis"] = per_balai.get(KRITIS, 0)
        per_balai = (per_balai.sort_values(["_kritis", WASPADA], ascending=True)
                     .drop(columns="_kritis").tail(15))
        fig_bl = go.Figure()
        for s in URUT_STATUS:
            if s not in per_balai.columns or per_balai[s].sum() == 0:
                continue
            fig_bl.add_trace(go.Bar(
                y=per_balai.index, x=per_balai[s], orientation="h",
                name=s, marker_color=WARNA_GRAFIK[s],
                text=[v if v else "" for v in per_balai[s]],
                textposition="inside"))
        fig_bl.update_layout(
            barmode="stack", height=max(380, 26 * len(per_balai) + 120),
            plot_bgcolor="white",
            title=dict(text="Status per balai (15 balai dengan kondisi terberat)",
                       font=dict(size=14, color=WARNA["navy"])),
            xaxis_title="Jumlah bendungan", legend=dict(orientation="h", y=-0.12),
            yaxis=dict(tickfont=dict(size=10)))
        fig_bl.update_xaxes(showgrid=True, gridcolor="#EEE")
        st.plotly_chart(fig_bl, use_container_width=True)
    with c_kanan:
        kritis_bdg = per_bdg[per_bdg["bulan_kritis"] > 0].nlargest(30, "skor_prioritas")
        if kritis_bdg.empty:
            st.success("✅ Tidak ada bendungan di bawah BON B pada cakupan filter ini.")
        else:
            bobot = {NORMAL: 0, TANPA_BON: 0, WASPADA: 1, KRITIS: 2}
            piv = (sbm[sbm["kode_bendungan"].isin(kritis_bdg["kode_bendungan"])]
                   .assign(b=lambda d: d["status_bon"].map(bobot))
                   .pivot_table(index="nama_bendungan", columns="bulan",
                                values="b", aggfunc="max"))
            piv = piv.loc[piv.max(axis=1).sort_values().index]
            fig_h = go.Figure(go.Heatmap(
                z=piv.values, x=[BULAN_ID[b] for b in piv.columns], y=piv.index,
                zmin=0, zmax=2, showscale=False, xgap=2, ygap=2,
                colorscale=[[0, "#C6EFCE"], [0.5, WARNA_GRAFIK[WASPADA]],
                            [1, WARNA_GRAFIK[KRITIS]]],
                hovertemplate="%{y} · %{x}: %{customdata}<extra></extra>",
                customdata=piv.replace({0: "Di atas BON A", 1: "Di antara BON A-B",
                                        2: "Di bawah BON B"}).values))
            fig_h.update_layout(
                height=max(380, 16 * len(piv) + 130), plot_bgcolor="white",
                title=dict(text=f"Kalender kritis {len(piv)} bendungan prioritas "
                                f"(merah = di bawah BON B)",
                           font=dict(size=14, color=WARNA["navy"])),
                yaxis=dict(tickfont=dict(size=9)))
            st.plotly_chart(fig_h, use_container_width=True)

    # -- pilih bendungan dalam pemantauan + carousel kartu
    st.markdown("#### 📡 Pilih bendungan dalam pemantauan")
    st.caption("Hanya bendungan terpilih yang ditampilkan sebagai kartu "
               "pemantauan di bawah — kartu bergeser otomatis dan dapat "
               "dijeda / dinavigasi manual.")
    daf_pantau = (dff[["kode_bendungan", "nama_bendungan", "nama_balai"]]
                  .drop_duplicates("kode_bendungan")
                  .sort_values("nama_bendungan"))
    peta_pantau = daf_pantau.set_index("kode_bendungan")
    opsi_pantau = list(daf_pantau["kode_bendungan"])
    bawaan = [k for k in st.session_state.get(
        "pantau_simpan", kode_kritis[:3] or opsi_pantau[:1])
        if k in opsi_pantau]
    pilih_pantau = st.multiselect(
        "Bendungan dalam pemantauan", opsi_pantau, default=bawaan,
        format_func=lambda k: (f"{peta_pantau.loc[k, 'nama_bendungan']} ({k}) "
                               f"— {peta_pantau.loc[k, 'nama_balai']}"))
    st.session_state["pantau_simpan"] = pilih_pantau

    INTERVAL_CAROUSEL = 12          # detik per kartu saat putar otomatis

    def _geser_kartu(langkah: int, n: int):
        st.session_state["car_idx"] = (
            st.session_state.get("car_idx", 0) + langkah) % n
        st.session_state["car_due"] = time.monotonic() + INTERVAL_CAROUSEL

    if not pilih_pantau:
        st.info("Pilih minimal satu bendungan untuk menampilkan kartu "
                "pemantauan.")
    else:
        putar = st.toggle("▶️ Putar otomatis (geser antar kartu)",
                          value=True, key="car_play")

        @st.fragment(run_every=3 if putar and len(pilih_pantau) > 1 else None)
        def _karusel():
            n = len(pilih_pantau)
            kini = time.monotonic()
            jatuh_tempo = st.session_state.get("car_due")
            if jatuh_tempo is None:
                st.session_state["car_due"] = kini + INTERVAL_CAROUSEL
            elif st.session_state.get("car_play") and kini >= jatuh_tempo:
                st.session_state["car_idx"] = (
                    st.session_state.get("car_idx", 0) + 1) % n
                st.session_state["car_due"] = kini + INTERVAL_CAROUSEL
            idx = st.session_state.get("car_idx", 0) % n
            k1, k2, k3 = st.columns([1, 1, 4])
            k1.button("⏮️ Sebelumnya", key="car_prev",
                      on_click=_geser_kartu, args=(-1, n))
            k2.button("⏭️ Berikutnya", key="car_next",
                      on_click=_geser_kartu, args=(+1, n))
            k3.caption(f"Kartu **{idx + 1} dari {n}** — "
                       + (f"bergeser otomatis tiap {INTERVAL_CAROUSEL} detik "
                          f"(jeda lewat tombol di atas)"
                          if st.session_state.get("car_play")
                          else "putar otomatis dijeda"))
            kartu_bendungan(pilih_pantau[idx])

        _karusel()

    # -- daftar prioritas gabungan
    st.markdown("#### 🎯 Daftar Prioritas Perhatian")
    st.caption("Skor prioritas = 3×bulan di bawah BON B + 1×bulan di antara "
               "BON A–B + 2×bulan defisit air (Agu–Des) + 1×bulan di bawah "
               "RTOW (bulan dinilai dari periode terburuknya). "
               "Semakin tinggi semakin butuh perhatian.")
    prio = (per_bdg[per_bdg["skor_prioritas"] > 0]
            .sort_values(["skor_prioritas", "bulan_kritis"], ascending=False)
            .head(30))
    if prio.empty:
        st.success("✅ Tidak ada bendungan berskor prioritas > 0 pada filter ini.")
    else:
        st.dataframe(warnai_status(
            prio[["kode_bendungan", "nama_bendungan", "nama_balai", "nama_pulau",
                  "status_terburuk", "bulan_kritis", "bulan_waspada",
                  "bulan_dibawah_rtow", "bulan_defisit_agu_des",
                  "defisit_terbesar_juta_m3", "tma_min", "skor_prioritas"]],
            "status_terburuk"), width="stretch", hide_index=True)

    # -- akurasi & metode
    st.markdown("#### 📐 Akurasi Prediksi & Perbandingan Metode")
    if evaluasi.empty:
        st.info("Jalankan `python src/evaluasi.py` untuk menghasilkan statistik "
                "akurasi (backtest beberapa metode).")
    else:
        ev = evaluasi[evaluasi["kode_bendungan"].isin(dff["kode_bendungan"].unique())]
        r = ringkas_evaluasi(ev)
        c_kiri, c_kanan = st.columns([3, 2])
        with c_kiri:
            fig_e = go.Figure()
            metode_urut = (r.groupby("metode")["mae_m"].mean()
                           .sort_values().index.tolist())
            warna_periode = {"Agu-Des 2025": WARNA["biru"],
                             "Mar-Jul 2026": WARNA["teal"]}
            for per in r["periode_uji"].unique():
                d = (r[r["periode_uji"] == per].set_index("metode")
                     .reindex(metode_urut))
                fig_e.add_trace(go.Bar(
                    x=metode_urut, y=d["mae_m"], name=f"Uji {per}",
                    marker_color=warna_periode.get(per, WARNA["abu"]),
                    text=d["mae_m"].round(2), textposition="outside"))
            fig_e.update_layout(
                barmode="group", height=400, plot_bgcolor="white",
                title=dict(text="MAE per metode (meter) — makin rendah makin akurat",
                           font=dict(size=14, color=WARNA["navy"])),
                yaxis_title="MAE (m)", legend=dict(orientation="h", y=-0.25))
            fig_e.update_yaxes(showgrid=True, gridcolor="#EEE")
            st.plotly_chart(fig_e, use_container_width=True)
        with c_kanan:
            terbaik = r.loc[r.groupby("periode_uji")["mae_m"].idxmin()]
            for _, b in terbaik.iterrows():
                st.metric(f"Metode terbaik — uji {b['periode_uji']}",
                          b["metode"], delta=f"MAE {b['mae_m']:.2f} m",
                          delta_color="off")
            lstm_r = r[r["metode"] == METODE_LSTM]
            if not lstm_r.empty:
                st.caption(
                    f"LSTM produksi: MAE {lstm_r['mae_m'].mean():.2f} m · RMSE "
                    f"{lstm_r['rmse_m'].mean():.2f} m · bias "
                    f"{lstm_r['bias_m'].mean():+.2f} m · MAPE thd rentang TMA "
                    f"{lstm_r['mape_rentang_pct'].mean():.1f}% "
                    f"(rata-rata {int(lstm_r['n_bendungan'].max())} bendungan, "
                    f"2 periode uji).")

        st.dataframe(r.rename(columns={
            "periode_uji": "Periode Uji", "metode": "Metode",
            "n_prediksi": "Jumlah Prediksi", "n_bendungan": "Jumlah Bendungan",
            "mae_m": "MAE (m)", "rmse_m": "RMSE (m)", "bias_m": "Bias (m)",
            "mape_rentang_pct": "MAPE thd Rentang (%)"}),
            width="stretch", hide_index=True)

        with st.expander("ℹ️ Penjelasan metode & cara membaca metrik"):
            for metode, desk in METODE_INFO.items():
                st.markdown(f"- **{metode}** — {desk}")
            st.markdown(
                "**Metrik:** MAE = rata-rata selisih absolut prediksi vs realisasi "
                "(meter); RMSE = akar rata-rata kuadrat error, lebih menghukum error "
                "besar; Bias = rata-rata error bertanda (positif = prediksi terlalu "
                "tinggi); MAPE thd rentang = MAE dibagi rentang TMA historis "
                "bendungan (persen), agar bendungan dangkal dan dalam sebanding.\n\n"
                "**Cara uji (backtest):** model dimundurkan ke titik waktu tertentu, "
                "memprediksi 5 bulan berikutnya hanya dari data sebelum titik itu, "
                "lalu dibandingkan dengan realisasi. Dua periode uji: Agu–Des 2025 "
                "(kalender sama dengan horizon prediksi 2026) dan Mar–Jul 2026 "
                "(terbaru). Catatan: periode uji masih berada dalam rentang data "
                "pelatihan model produksi sehingga skor LSTM cenderung optimis; "
                "baseline menjadi pembanding yang adil.")

    # -- kesimpulan menyeluruh + unduh PDF
    st.markdown("#### 📝 Kesimpulan")
    for i, p in enumerate(lap.kesimpulan_nasional(dff, neraca_dff, evaluasi), 1):
        st.markdown(f"**{i}.** {p}")

    c_pdf1, c_pdf2 = st.columns(2)
    if c_pdf1.button("⚙️ Siapkan laporan PDF (kesimpulan + rekap + grafik "
                     "bendungan paling rawan)", width="stretch"):
        with st.spinner("Menyusun laporan PDF…"):
            st.session_state["pdf_siaga"] = buat_pdf(
                dff, neraca_dff, evaluasi, filter_teks, waktu_panjang())
    if "pdf_siaga" in st.session_state:
        c_pdf2.download_button(
            f"📄 Unduh PDF Siaga Kekeringan "
            f"({len(st.session_state['pdf_siaga']) / 1e6:.1f} MB)",
            st.session_state["pdf_siaga"],
            file_name=f"laporan_siaga_kekeringan_{stamp()}.pdf",
            mime="application/pdf", width="stretch")


# ============================================================ 1. DETAIL
elif menu == "📈 Detail Bendungan":
    daf = (dff[["kode_bendungan", "nama_bendungan", "nama_balai"]]
           .drop_duplicates("kode_bendungan").sort_values("nama_bendungan"))
    peta = daf.set_index("kode_bendungan")
    c_sel, c_hist = st.columns([3, 1])
    pilihan = c_sel.selectbox(
        "Pilih Bendungan", daf["kode_bendungan"],
        format_func=lambda k: f"{peta.loc[k, 'nama_bendungan']} ({k}) — {peta.loc[k, 'nama_balai']}")
    tampil_hist = c_hist.checkbox("Tampilkan historis 2018–2025", value=False)

    g = df[df["kode_bendungan"] == pilihan].sort_values("tanggal")
    nama = g["nama_bendungan"].iloc[0]
    fmt = g["format_periode"].iloc[0] if "format_periode" in g.columns else "15 Harian"
    real = g[g["jenis"] == "realisasi"]
    pred = g[g["jenis"] == "prediksi"]
    urut = daftar_periode(fmt, 1, 12)                 # '01-01' ... '12-02'/'12-03'

    st.caption(f"{g['nama_balai'].iloc[0]} · {g['nama_pulau'].iloc[0]} · "
               f"skala periode **{fmt}** ({len(urut)} periode/tahun, sesuai "
               f"format RTOW) · Bon A/B, RTOW, ketersediaan & kebutuhan air "
               f"per periode dari workbook RTOW periodik")

    fig, catatan = fig_sandingan(g, tampil_hist)
    if catatan:
        st.info("ℹ️ " + " · ".join(catatan))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    if not real.empty:
        c1.metric(f"TMA realisasi terakhir (periode {real['periode'].iloc[-1]})",
                  f"{real['tma'].iloc[-1]:.2f} mdpl")
    if not pred.empty:
        c2.metric(f"TMA awal Agustus (periode {pred['periode'].iloc[0]})",
                  f"{pred['tma'].iloc[0]:.2f} mdpl",
                  delta=f"{pred['tma'].iloc[0] - real['tma'].iloc[-1]:+.2f} m"
                  if not real.empty else None, delta_color="inverse")
        c3.metric(f"TMA akhir Desember (periode {pred['periode'].iloc[-1]})",
                  f"{pred['tma'].iloc[-1]:.2f} mdpl",
                  delta=f"{pred['tma'].iloc[-1] - real['tma'].iloc[-1]:+.2f} m"
                  if not real.empty else None, delta_color="inverse")
        kritis = pred[lap.dibawah_bon(pred["status_bon"])]
        c4.metric("Periode prediksi di bawah BON A",
                  f"{len(kritis)} dari {len(pred)}")

    # ---------- keputusan 3 zona periode RTOW berjalan (workbook baru) ----------
    pb, kek_b, kurva_b = ambil_periodik(pilihan, fmt)
    status_zona, label_aktif, tma_kini = blok_status_zona(pilihan, fmt, g, pb)

    # ---------- hasil analisis gabungan ----------
    st.markdown("##### 🔎 Hasil Analisis")
    penjelasan = lap.narasi_bendungan(g)
    status = lap.status_terburuk(pred["status_bon"]) if not pred.empty else TANPA_BON
    if status == KRITIS:
        st.error(f"🚨 {penjelasan}")
    elif status == WASPADA:
        st.warning(f"⚠️ {penjelasan}")
    elif status == NORMAL:
        st.success(f"✅ {penjelasan}")
    else:
        st.info(f"ℹ️ {penjelasan}")

    info_tambahan = []
    sm_b = sifat_musim[sifat_musim["kode_bendungan"] == pilihan]
    if not sm_b.empty:
        urutan_sm = " → ".join(f"{r.tahun_air}: {r.sifat}"
                               for r in sm_b.itertuples())
        info_tambahan.append(f"**Sifat musim** (fitur pembelajaran LSTM): "
                             f"{urutan_sm}.")
    if not evaluasi.empty:
        ev_b = evaluasi[(evaluasi["kode_bendungan"] == pilihan)
                        & evaluasi["error"].notna()].copy()
        if not ev_b.empty:
            ev_b["abs"] = ev_b["error"].abs()
            mae_b = ev_b.groupby("metode")["abs"].mean().sort_values()
            info_tambahan.append(
                f"**Akurasi backtest bendungan ini**: MAE terbaik "
                f"{mae_b.index[0]} = {mae_b.iloc[0]:.2f} m "
                f"(LSTM: {mae_b.get(METODE_LSTM, float('nan')):.2f} m).")
    ringkas_status = (
        f"**Rekap periode prediksi**: {int((pred['status_bon'] == NORMAL).sum())} "
        f"di atas BON A · {int((pred['status_bon'] == WASPADA).sum())} di antara "
        f"BON A–B · {int((pred['status_bon'] == KRITIS).sum())} di bawah BON B "
        f"(dari {len(pred)} periode).")
    st.markdown(ringkas_status)
    for t in info_tambahan:
        st.markdown(t)

    kolom_tampil = [c for c in ("periode", "tanggal", "tma", "jenis", "bon_a",
                                "bon_b", "rtow", "status_bon") if c in g.columns]
    tampil = g[kolom_tampil].copy()
    tampil["tanggal"] = tampil["tanggal"].dt.strftime("%d %b %Y")
    st.dataframe(warnai_status(tampil, "status_bon"), width="stretch",
                 hide_index=True)

    # ---------- kecukupan air: kebutuhan vs ketersediaan (RTOW periodik) ----------
    st.markdown("##### 💧 Kecukupan Air: Kebutuhan vs Ketersediaan "
                "(RTOW per periode)")
    ada_kck = (pb["ketersediaan_m3s"].notna() & pb["kebutuhan_m3s"].notna())
    if not ada_kck.any():
        st.info("ℹ️ Data ketersediaan/kebutuhan air per periode belum tersedia "
                "pada workbook RTOW — kecukupan air belum dapat dinilai.")
    else:
        blok_kecukupan(pilihan, fmt, pb)

        tk = pb.copy()
        tk["ketersediaan_m3"] = kc.debit_ke_volume(
            pd.to_numeric(tk["ketersediaan_m3s"], errors="coerce"), fmt)
        tk["kebutuhan_m3"] = kc.debit_ke_volume(
            pd.to_numeric(tk["kebutuhan_m3s"], errors="coerce"), fmt)
        tk["deviasi_m3"] = tk["ketersediaan_m3"] - tk["kebutuhan_m3"]
        tk["deviasi_pct"] = np.where(
            tk["kebutuhan_m3"] > 0,
            tk["deviasi_m3"] / tk["kebutuhan_m3"] * 100, np.nan)
        lengkap = tk["ketersediaan_m3"].notna() & tk["kebutuhan_m3"].notna()
        tk["status_kecukupan"] = np.select(
            [lengkap & (tk["deviasi_m3"] >= 0), lengkap],
            ["Cukup", "Belum cukup"], "Tanpa Data")

        fig_k = go.Figure()
        fig_k.add_trace(go.Bar(
            x=tk.index, y=(tk["ketersediaan_m3"] / 1e6).round(3),
            name="Ketersediaan air", marker_color=WARNA["teal"]))
        fig_k.add_trace(go.Bar(
            x=tk.index, y=(tk["kebutuhan_m3"] / 1e6).round(3),
            name="Kebutuhan air", marker_color=WARNA["navy"]))
        blm = tk[tk["status_kecukupan"] == "Belum cukup"]
        if not blm.empty:
            fig_k.add_trace(go.Scatter(
                x=blm.index,
                y=(blm[["ketersediaan_m3", "kebutuhan_m3"]].max(axis=1) / 1e6) * 1.06,
                mode="text", text="▼ belum cukup",
                textfont=dict(color=WARNA["merah"], size=11),
                showlegend=False, hoverinfo="skip"))
        fig_k.update_layout(
            barmode="group", height=380, plot_bgcolor="white",
            yaxis_title="Volume (juta m³/periode)", hovermode="x unified",
            legend=dict(orientation="h", y=-0.25),
            title=dict(text=f"Kecukupan Air per Periode {fmt} — Bendungan {nama}",
                       font=dict(size=15, color=WARNA["navy"])))
        fig_k.update_xaxes(type="category", categoryorder="array",
                           categoryarray=urut, tickangle=-90,
                           tickfont=dict(size=9))
        fig_k.update_yaxes(showgrid=True, gridcolor="#EEE")
        st.plotly_chart(fig_k, use_container_width=True)

        tampil_k = tk.reset_index().rename(columns={"index": "periode"})
        tampil_k = tampil_k[["periode", "ketersediaan_m3s", "kebutuhan_m3s",
                             "ketersediaan_m3", "kebutuhan_m3", "deviasi_m3",
                             "deviasi_pct", "status_kecukupan"]].round(
            {"ketersediaan_m3s": 3, "kebutuhan_m3s": 3, "ketersediaan_m3": 0,
             "kebutuhan_m3": 0, "deviasi_m3": 0, "deviasi_pct": 1})
        st.dataframe(warnai_status(tampil_k, "status_kecukupan"),
                     width="stretch", hide_index=True)

    # ---------- rekomendasi operasional (Skenario 3, rule-based) ----------
    blok_rekomendasi(pilihan, nama, g["nama_balai"].iloc[0], fmt, g, pb,
                     kek_b, kurva_b, status_zona, label_aktif, tma_kini)

    st.markdown("##### Unduh untuk bendungan ini")
    u1, u2 = st.columns(2)
    u1.download_button(
        "🖼️ Grafik JPG", buat_jpg(g, hist, tampil_hist),
        file_name=f"grafik_{pilihan}_{nama.replace('/', '-')}_{stamp()}.jpg",
        mime="image/jpeg", width="stretch")
    u2.download_button(
        "📊 Excel bendungan ini", buat_excel(
            g, daftar, hist, False, 8, f"Bendungan {nama} ({pilihan})",
            waktu_panjang(), rekap_qc, neraca),
        file_name=f"prediksi_tma_{pilihan}_{stamp()}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch")


# ============================================================ 2. AGUSTUS
elif menu == "🚨 Pemantauan Agustus 2026":
    bulan_tersedia = sorted(pred_dff["tanggal"].dt.month.unique())
    c_b, c_n = st.columns([1, 1])
    bulan = c_b.selectbox("Bulan prediksi yang dipantau", bulan_tersedia,
                          index=bulan_tersedia.index(8) if 8 in bulan_tersedia else 0,
                          format_func=lambda m: f"{BULAN_ID[m]} 2026")
    n_peringkat = c_n.slider("Jumlah bendungan pada grafik peringkat", 5, 50, 25, 5)

    st.subheader(f"Pemantauan Bendungan — {BULAN_ID[bulan]} 2026")
    tab = lap.tabel_pemantauan_bulan(dff, bulan)
    if tab.empty:
        st.warning(f"Tidak ada data prediksi {BULAN_ID[bulan]} pada filter ini.")
        st.stop()

    n_kritis = int((tab["status_bon"] == KRITIS).sum())
    n_waspada = int((tab["status_bon"] == WASPADA).sum())
    n_normal = int((tab["status_bon"] == NORMAL).sum())
    n_tanpa = int((tab["status_bon"] == TANPA_BON).sum())
    n_rtow = (int((tab["selisih_ke_rtow_m"] < 0).sum())
              if "selisih_ke_rtow_m" in tab.columns else 0)
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Bendungan dipantau", len(tab))
    m2.metric("🚨 Di bawah BON B", n_kritis)
    m3.metric("⚠️ Di antara BON A–B", n_waspada)
    m4.metric("✅ Di atas BON A", n_normal)
    m5.metric("➖ Tanpa data BON", n_tanpa)
    m6.metric("📉 Di bawah RTOW", n_rtow)
    st.caption("Nilai per bendungan diambil dari **periode terburuk** (TMA "
               "terendah) dalam bulan tinjauan — kolom `periode_terburuk`.")

    if n_kritis:
        nm = tab[tab["status_bon"] == KRITIS]
        st.error(f"🚨 **{n_kritis} bendungan** diprediksi di bawah BON B pada "
                 f"{BULAN_ID[bulan]} 2026, terparah **{nm['nama_bendungan'].iloc[0]}** "
                 f"({nm['nama_balai'].iloc[0]}) dengan defisit "
                 f"{abs(nm['selisih_ke_bon_b_m'].iloc[0]):.2f} m di bawah BON B.")
    else:
        st.success(f"✅ Tidak ada bendungan di bawah BON B pada {BULAN_ID[bulan]} 2026 "
                   f"pada cakupan filter ini.")

    t_semua, t_kritis, t_balai = st.tabs(
        [f"📋 Semua ({len(tab)})", f"🚨 Di bawah BON B ({n_kritis})", "🏢 Per Balai"])
    with t_semua:
        st.dataframe(warnai_status(tab, "status_bon"), width="stretch",
                     hide_index=True)
    with t_kritis:
        kr = tab[tab["status_bon"] == KRITIS]
        if kr.empty:
            st.info("Tidak ada bendungan berstatus di bawah BON B.")
        else:
            st.dataframe(warnai_status(kr, "status_bon"), width="stretch",
                         hide_index=True)
    with t_balai:
        st.dataframe(lap.rekap_per_balai(dff, bulan), width="stretch",
                     hide_index=True)

    st.markdown("##### Grafik peringkat kerawanan")
    fig_p = lap.fig_peringkat_bulan(tab, bulan, n_peringkat)
    if fig_p is None:
        st.info("Belum ada nilai BON B pada bendungan terfilter, grafik peringkat "
                "tidak dapat dibuat.")
    else:
        jpg_p = lap.fig_ke_jpg(fig_p)
        st.image(jpg_p, width="stretch")
        st.download_button(f"🖼️ Unduh grafik peringkat {BULAN_ID[bulan]} (JPG)", jpg_p,
                           file_name=f"peringkat_{BULAN_ID[bulan].lower()}_2026_{stamp()}.jpg",
                           mime="image/jpeg")

    st.markdown("##### Unduh pemantauan bulan ini")
    d1, d2 = st.columns(2)
    d1.download_button(
        f"📊 Excel pemantauan {BULAN_ID[bulan]} — semua bendungan terfilter",
        buat_excel(dff, daftar, hist, False, bulan, filter_teks, waktu_panjang(),
                   rekap_qc, neraca),
        file_name=f"pemantauan_{BULAN_ID[bulan].lower()}_2026_semua_{stamp()}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch")
    d2.download_button(
        f"📊 Excel pemantauan {BULAN_ID[bulan]} — hanya di bawah BON B "
        f"({len(kode_kritis)} bendungan)",
        buat_excel(dff, daftar, hist, True, bulan, filter_teks, waktu_panjang(),
                   rekap_qc, neraca) if kode_kritis else b"",
        file_name=f"pemantauan_{BULAN_ID[bulan].lower()}_2026_dibawah_bon_b_{stamp()}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        disabled=not len(kode_kritis), width="stretch")


# ============================================================ 3. REKAP
elif menu == "🗂️ Rekap & Di bawah BON B":
    st.subheader("Rekapitulasi Status BON — Agustus–Desember 2026")
    rekap = lap.rekap_bendungan(dff)
    n_atas = int((rekap["status_terburuk"] == NORMAL).sum())
    n_antara = int((rekap["status_terburuk"] == WASPADA).sum())
    n_bawah = int((rekap["periode_dibawah_bon_b"] > 0).sum())
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Bendungan", len(rekap))
    m2.metric("✅ Di atas BON A (selalu)", n_atas)
    m3.metric("⚠️ Di antara BON A dan BON B", n_antara)
    m4.metric("🚨 Di bawah BON B", n_bawah)
    m5.metric("➖ Tanpa data BON",
              int((rekap["status_terburuk"] == TANPA_BON).sum()))
    st.caption("Kategori dinilai dari **status terburuk** seluruh periode "
               "prediksi Agu–Des 2026 per bendungan.")

    t1, t2, t3, t4, t5 = st.tabs(
        [f"🗂️ Semua ({len(rekap)})", f"🚨 Di bawah BON B ({n_bawah})",
         f"⚠️ Di antara BON A–B ({n_antara})", f"✅ Di atas BON A ({n_atas})",
         "🏢 Per Balai"])
    with t1:
        st.dataframe(warnai_status(rekap, "status_terburuk"), width="stretch",
                     hide_index=True,
                     column_config={"penjelasan": st.column_config.TextColumn(
                         "penjelasan", width="large")})
    with t2:
        kr = rekap[rekap["periode_dibawah_bon_b"] > 0]
        if kr.empty:
            st.success("✅ Tidak ada bendungan yang diprediksi menembus BON B "
                       "pada cakupan filter ini.")
        else:
            st.write(f"**{len(kr)} bendungan** menembus BON B pada satu periode "
                     f"atau lebih (Agustus–Desember 2026).")
            st.dataframe(warnai_status(kr, "status_terburuk"), width="stretch",
                         hide_index=True)
            detail = pred_dff[pred_dff["status_bon"] == KRITIS][
                ["kode_bendungan", "nama_bendungan", "nama_balai", "periode",
                 "tanggal", "tma", "bon_b", "status_bon"]].copy()
            detail["defisit_m"] = (detail["bon_b"] - detail["tma"]).round(2)
            detail["tanggal"] = detail["tanggal"].dt.strftime("%d %b %Y")
            st.markdown("**Rincian periode yang menembus BON B**")
            st.dataframe(detail.sort_values("defisit_m", ascending=False),
                         width="stretch", hide_index=True)
    with t3:
        ka = rekap[rekap["status_terburuk"] == WASPADA]
        if ka.empty:
            st.info("Tidak ada bendungan pada kategori ini.")
        else:
            st.write(f"**{len(ka)} bendungan** turun di bawah BON A tetapi "
                     f"masih di atas BON B — perlu pengetatan alokasi dan "
                     f"pemantauan lebih rapat.")
            st.dataframe(warnai_status(ka, "status_terburuk"), width="stretch",
                         hide_index=True)
    with t4:
        na_ = rekap[rekap["status_terburuk"] == NORMAL]
        if na_.empty:
            st.info("Tidak ada bendungan pada kategori ini.")
        else:
            st.write(f"**{len(na_)} bendungan** diprediksi tetap di atas BON A "
                     f"sepanjang Agu–Des 2026 — pasokan air aman sesuai pola "
                     f"operasi normal.")
            st.dataframe(warnai_status(na_, "status_terburuk"), width="stretch",
                         hide_index=True)
    with t5:
        st.dataframe(lap.rekap_per_balai(dff), width="stretch", hide_index=True)


# ============================================================ 4. BELUM COCOK
elif menu == "🧩 Data Belum Cocok":
    st.subheader("Data Bendungan yang Belum Cocok / Belum Lengkap")
    st.caption("Dihitung dari **seluruh** bendungan (tidak mengikuti filter wilayah) "
               "supaya tidak ada masalah data yang tersembunyi.")
    masalah = lap.daftar_belum_cocok(daftar, df, hist, rekap_qc)

    if masalah.empty:
        st.success("✅ Seluruh bendungan pada daftar BON cocok dengan data database, "
                   "punya hasil prediksi, dan nilai BON lengkap.")
    else:
        kat = sorted(masalah["kategori_masalah"].unique())
        m = st.columns(min(len(kat), 5))
        for i, k in enumerate(kat):
            m[i % len(m)].metric(k.split(". ", 1)[-1],
                                 int((masalah["kategori_masalah"] == k).sum()))

        pilih_kat = st.multiselect("Kategori masalah", kat, default=kat)
        tampil = masalah[masalah["kategori_masalah"].isin(pilih_kat)]
        st.dataframe(tampil, width="stretch", hide_index=True,
                     column_config={
                         "keterangan": st.column_config.TextColumn(width="large"),
                         "tindak_lanjut": st.column_config.TextColumn(width="large")})

        with st.expander("Cara membaca kategori masalah"):
            st.markdown(
                "- **1a. Tidak ada di database** — kode/nama bendungan ada di daftar BON "
                "tetapi tidak ditemukan pada data TMA hasil ekstraksi SINBAD. Umumnya "
                "karena pemetaan `BDxxx → id` tidak cocok (tambahkan kolom `id_db` pada "
                "`daftar_bendungan.csv`) atau bendungan belum mengirim data.\n"
                "- **1b. Data ditolak QC** — data TMA ada di database, tetapi seluruh "
                "nilainya berada di luar rentang elevasi storage curve bendungan "
                "tersebut sehingga ditolak QC. Indikasi id database atau satuan tidak "
                "cocok — verifikasi dengan `SELECT DISTINCT id FROM "
                "rawdata.sinbad_bendungan_tma`.\n"
                "- **2. Tidak ada hasil prediksi** — histori ada tetapi kurang dari 12 "
                "bulan (panjang sekuens input model).\n"
                "- **3. Tidak ada di daftar BON** — kode muncul di data TMA/prediksi "
                "tetapi belum terdaftar di `daftar_bendungan.csv`.\n"
                "- **4. Tanpa nilai BON A/B** — tetap diprediksi, tetapi status tidak "
                "dapat dinilai karena BON belum diisi.\n"
                "- **5. BON terisi otomatis** — sebagian bulan BON kosong lalu "
                "diinterpolasi linear; status bulan tersebut bersifat perkiraan.")

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
            masalah.to_excel(w, sheet_name="Data Belum Cocok", index=False)
        st.download_button("📊 Unduh daftar data belum cocok (Excel)", buf.getvalue(),
                           file_name=f"data_bendungan_belum_cocok_{stamp()}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument."
                                "spreadsheetml.sheet")

    st.divider()
    st.markdown("##### Cakupan data saat ini")
    c1, c2, c3 = st.columns(3)
    c1.metric("Bendungan di daftar BON", len(daftar))
    c2.metric("Punya histori TMA (database)",
              int(hist["kode_bendungan"].nunique()) if not hist.empty else 0)
    c3.metric("Punya hasil prediksi",
              int(df[df["jenis"] == "prediksi"]["kode_bendungan"].nunique()))


# ============================================================ 5. UNDUH
else:
    st.subheader("Unduh Laporan Prediksi TMA 2026")
    st.caption(f"Cakupan mengikuti filter panel kiri — saat ini: {filter_teks}.")

    kode_terfilter = sorted(dff["kode_bendungan"].unique())
    st.markdown("#### 1. Laporan Excel (informasi + penjelasan)")
    st.markdown(
        "Berisi lembar **Penjelasan** (metodologi, arti status BON, definisi kolom, "
        "batasan), **Ringkasan Bendungan** (termasuk narasi otomatis per bendungan), "
        "**Pemantauan Agustus**, **Prediksi Bulanan** lengkap, **Di bawah BON B**, "
        "**Rekap per Balai**, dan **Data Belum Cocok**.")
    e1, e2 = st.columns(2)
    e1.download_button(
        f"📊 Excel — semua bendungan ({len(kode_terfilter)})",
        buat_excel(dff, daftar, hist, False, 8, filter_teks, waktu_panjang(),
                   rekap_qc, neraca),
        file_name=f"laporan_prediksi_tma_2026_semua_{stamp()}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch")
    e2.download_button(
        f"📊 Excel — hanya di bawah BON B ({len(kode_kritis)})",
        buat_excel(dff, daftar, hist, True, 8, filter_teks, waktu_panjang(),
                   rekap_qc, neraca) if kode_kritis else b"",
        file_name=f"laporan_prediksi_tma_2026_dibawah_bon_b_{stamp()}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        disabled=not kode_kritis, width="stretch")

    st.divider()
    st.markdown("#### 2. Grafik JPG (ZIP per bendungan)")
    st.markdown("Setiap bendungan satu berkas JPG grafik sandingan realisasi vs "
                "prediksi + BON A/B, dikelompokkan ke folder per status "
                "(`KRITIS/`, `WASPADA/`, `NORMAL/`, `TANPA-BON/`).")
    z1, z2 = st.columns(2)
    hist_zip = st.checkbox("Sertakan garis historis 2018–2025 pada grafik", value=False)

    def siapkan_zip(kunci, kodes, label):
        bar = st.progress(0.0, text=f"Merender {label}…")
        data = lap.zip_grafik(
            dff, kodes, hist, hist_zip,
            progress=lambda i, n, t: bar.progress(i / n, text=f"[{i}/{n}] {t}"))
        bar.empty()
        st.session_state[f"zip_{kunci}"] = data

    if z1.button(f"⚙️ Siapkan ZIP — semua bendungan ({len(kode_terfilter)})",
                 width="stretch"):
        siapkan_zip("semua", kode_terfilter, f"{len(kode_terfilter)} grafik")
    if "zip_semua" in st.session_state:
        z1.download_button(
            f"🖼️ Unduh ZIP grafik semua bendungan "
            f"({len(st.session_state['zip_semua']) / 1e6:.1f} MB)",
            st.session_state["zip_semua"],
            file_name=f"grafik_prediksi_tma_2026_semua_{stamp()}.zip",
            mime="application/zip", width="stretch")

    if z2.button(f"⚙️ Siapkan ZIP — di bawah BON B ({len(kode_kritis)})",
                 disabled=not kode_kritis, width="stretch"):
        siapkan_zip("kritis", kode_kritis, f"{len(kode_kritis)} grafik kritis")
    if "zip_kritis" in st.session_state:
        z2.download_button(
            f"🖼️ Unduh ZIP grafik di bawah BON B "
            f"({len(st.session_state['zip_kritis']) / 1e6:.1f} MB)",
            st.session_state["zip_kritis"],
            file_name=f"grafik_prediksi_tma_2026_dibawah_bon_b_{stamp()}.zip",
            mime="application/zip", width="stretch")

    st.divider()
    st.markdown("#### 3. Laporan PDF (kesimpulan + rekap + grafik)")
    st.markdown("Satu berkas PDF siap cetak: kesimpulan otomatis seluruh data, "
                "rekap status per bulan & per balai, tabel bendungan di bawah "
                "BON B, dan grafik sandingan bendungan paling rawan.")
    p1, p2 = st.columns(2)
    if p1.button("⚙️ Siapkan laporan PDF", width="stretch", key="pdf_unduh_btn"):
        with st.spinner("Menyusun laporan PDF…"):
            st.session_state["pdf_siaga"] = buat_pdf(
                dff, neraca_dff, evaluasi, filter_teks, waktu_panjang())
    if "pdf_siaga" in st.session_state:
        p2.download_button(
            f"📄 Unduh PDF ({len(st.session_state['pdf_siaga']) / 1e6:.1f} MB)",
            st.session_state["pdf_siaga"],
            file_name=f"laporan_siaga_kekeringan_{stamp()}.pdf",
            mime="application/pdf", width="stretch")

    st.divider()
    st.markdown("#### 4. Pratinjau isi laporan")
    with st.expander("Ringkasan bendungan (baris pertama)"):
        st.dataframe(lap.rekap_bendungan(dff).head(15), width="stretch",
                     hide_index=True)
    with st.expander("Pemantauan Agustus 2026 (baris pertama)"):
        st.dataframe(lap.tabel_pemantauan_bulan(dff, 8).head(15), width="stretch",
                     hide_index=True)

st.caption("Model: LSTM encoder-decoder per periode 10/15-harian sesuai format "
           "RTOW (input 1 tahun periode → output Agu–Des) · fitur: TMA & inflow "
           "per periode hasil QC, posisi musiman sin/cos, dan sifat musim tahun "
           "air (Basah/Normal/Kering) · sumber: SINBAD (rawdata) · "
           "Subdit OP Bendungan dan Danau, Dit. Bina OP.")
