"""Dashboard Streamlit — Prediksi TMA Bulanan Agustus-Desember 2026 (LSTM).

Jalankan:  streamlit run app.py
Prasyarat: pipeline sudah dijalankan (python run_pipeline.py) sehingga
           output/prediksi/prediksi_tma_2026.parquet tersedia.

Menu:
  🏜️ Siaga Kekeringan        - ringkasan visual nasional + akurasi & metode
  📈 Detail Bendungan        - grafik sandingan + neraca air + unduh per bendungan
  🚨 Pemantauan Agustus 2026 - tabel pemantauan bulan fokus untuk semua bendungan
  🗂️ Rekap & Di bawah BON B  - rekap nasional, per balai, daftar kritis
  🧩 Data Belum Cocok        - bendungan yang datanya belum terhubung antar sumber
  ⬇️ Unduh Laporan           - Excel + ZIP grafik JPG (semua / hanya di bawah BON B)
"""
import io
import os
import sys
import datetime as dt
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.utils import (load_config, path_root, baca_daftar_bendungan,
                       baca_neraca_air, baca_sifat_musim)
import src.laporan as lap
from src.laporan import KRITIS, WASPADA, NORMAL, TANPA_BON, BULAN_ID, WARNA
from src.evaluasi import METODE_INFO, METODE_LSTM, ringkas as ringkas_evaluasi

st.set_page_config(page_title="Prediksi TMA 2026 — PMB", page_icon="🌊",
                   layout="wide")

# latar pastel untuk sel tabel
WARNA_STATUS = {KRITIS: "#F8CBCB", WASPADA: "#FFEB9C",
                NORMAL: "#C6EFCE", TANPA_BON: "#E4E4E4",
                "Defisit": "#F8CBCB", "Surplus": "#C6EFCE",
                "Tanpa Data": "#E4E4E4"}
# isi tegas untuk mark grafik (selalu berpasangan dengan label + angka)
WARNA_GRAFIK = {KRITIS: "#C62828", WASPADA: "#D98E04",
                NORMAL: "#2E7D32", TANPA_BON: "#9E9E9E",
                "Defisit": "#C62828", "Surplus": "#2A9D8F"}
URUT_STATUS = [KRITIS, WASPADA, NORMAL, TANPA_BON]


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
    return df, hist, daftar, rekap_qc, neraca, evaluasi, sifat_musim


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


df, hist, daftar, rekap_qc, neraca, evaluasi, sifat_musim = muat_data()

# ------------------------------------------------------------------ header
st.markdown(
    f"<h2 style='color:{WARNA['navy']};margin-bottom:0'>🌊 Prediksi TMA Musim Kering 2026</h2>"
    f"<p style='color:{WARNA['biru']};margin-top:2px'>Pusat Monitoring Bendungan — LSTM per periode "
    f"10/15-harian (format RTOW) + sifat musim · prediksi Agustus–Desember 2026 · "
    f"sandingan BON A / BON B / RTOW</p>",
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

    # -- baris 2: per balai + neraca air per bulan
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
        nrc_bulan = (neraca_dff[neraca_dff["status_neraca"] != "Tanpa Data"]
                     .groupby(["bulan", "status_neraca"]).size()
                     .unstack(fill_value=0)
                     .reindex(columns=["Defisit", "Surplus"], fill_value=0))
        fig_n = go.Figure()
        for s in ("Defisit", "Surplus"):
            fig_n.add_trace(go.Bar(
                x=[BULAN_ID[b] for b in nrc_bulan.index], y=nrc_bulan[s],
                name=s, marker_color=WARNA_GRAFIK[s],
                text=nrc_bulan[s], textposition="inside"))
        fig_n.update_layout(
            barmode="stack", height=max(380, 26 * len(per_balai) + 120),
            plot_bgcolor="white",
            title=dict(text="Neraca air: jumlah bendungan defisit vs surplus per bulan",
                       font=dict(size=14, color=WARNA["navy"])),
            yaxis_title="Jumlah bendungan", legend=dict(orientation="h", y=-0.12))
        fig_n.update_yaxes(showgrid=True, gridcolor="#EEE")
        st.plotly_chart(fig_n, use_container_width=True)

    # -- baris 3: peta panas bendungan kritis + top defisit
    c_kiri, c_kanan = st.columns(2)
    with c_kiri:
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
    with c_kanan:
        top_def = (per_bdg[per_bdg["defisit_terbesar_juta_m3"] > 0]
                   .nlargest(15, "defisit_terbesar_juta_m3")
                   .sort_values("defisit_terbesar_juta_m3"))
        if top_def.empty:
            st.success("✅ Tidak ada defisit air Agu–Des pada cakupan filter ini.")
        else:
            fig_d = go.Figure(go.Bar(
                y=top_def["nama_bendungan"] + " (" + top_def["kode_bendungan"] + ")",
                x=top_def["defisit_terbesar_juta_m3"], orientation="h",
                marker_color=WARNA_GRAFIK["Defisit"],
                text=top_def["defisit_terbesar_juta_m3"].round(2),
                textposition="outside"))
            fig_d.update_layout(
                height=max(380, 24 * len(top_def) + 120), plot_bgcolor="white",
                title=dict(text="15 defisit air bulanan terbesar Agu–Des (juta m³)",
                           font=dict(size=14, color=WARNA["navy"])),
                xaxis_title="Defisit bulan terparah (juta m³)",
                yaxis=dict(tickfont=dict(size=10)))
            fig_d.update_xaxes(showgrid=True, gridcolor="#EEE")
            st.plotly_chart(fig_d, use_container_width=True)

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
    from src.utils import daftar_periode
    urut = daftar_periode(fmt, 1, 12)                 # '01-01' ... '12-02'/'12-03'
    d = g.drop_duplicates("periode").set_index("periode").reindex(urut)

    st.caption(f"{g['nama_balai'].iloc[0]} · {g['nama_pulau'].iloc[0]} · "
               f"skala periode **{fmt}** ({len(urut)} periode/tahun, sesuai "
               f"format RTOW)")

    fig = go.Figure()
    if tampil_hist and not hist.empty:
        h = hist[(hist["kode_bendungan"] == pilihan)
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
    if "rtow" in d.columns and d["rtow"].notna().any():
        fig.add_trace(go.Scatter(x=urut, y=d["rtow"], mode="lines",
                                 line=dict(color=WARNA["biru"], width=2, dash="dot"),
                                 name="RTOW (rencana operasi)"))
    if d["bon_a"].notna().any():
        fig.add_trace(go.Scatter(x=urut, y=d["bon_a"], mode="lines",
                                 line=dict(color=WARNA["amber"], width=2, dash="dashdot"),
                                 name="BON A"))
    if d["bon_b"].notna().any():
        fig.add_trace(go.Scatter(x=urut, y=d["bon_b"], mode="lines",
                                 line=dict(color=WARNA["coral"], width=2, dash="dashdot"),
                                 name="BON B"))
    if d["bon_a"].isna().all() and d["bon_b"].isna().all():
        st.info("ℹ️ Bendungan ini belum memiliki data BON A/B pada file referensi — "
                "status prediksi tidak dapat dinilai.")
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

    # ---------- neraca air: kebutuhan vs ketersediaan ----------
    st.markdown("##### 💧 Neraca Air: Kebutuhan vs Ketersediaan")
    n_b = neraca[neraca["kode_bendungan"] == pilihan].sort_values("bulan")
    if n_b.empty or (n_b["status_neraca"] == "Tanpa Data").all():
        st.info("ℹ️ Bendungan ini belum memiliki data ketersediaan/kebutuhan "
                "air pada file referensi.")
    else:
        narasi_n = lap.narasi_neraca(n_b)
        if (n_b["status_neraca"] == "Defisit").any():
            st.warning(f"⚠️ {narasi_n}")
        else:
            st.success(f"✅ {narasi_n}")

        nb = n_b.copy()
        nb["nama_bulan"] = nb["bulan"].map(BULAN_ID)
        fig_n = go.Figure()
        fig_n.add_trace(go.Bar(
            x=nb["nama_bulan"], y=(nb["ketersediaan_m3"] / 1e6).round(3),
            name="Ketersediaan air", marker_color=WARNA["teal"]))
        fig_n.add_trace(go.Bar(
            x=nb["nama_bulan"], y=(nb["kebutuhan_m3"] / 1e6).round(3),
            name="Kebutuhan air", marker_color=WARNA["navy"]))
        # tanda defisit di atas pasangan batang
        def_b = nb[nb["status_neraca"] == "Defisit"]
        if not def_b.empty:
            fig_n.add_trace(go.Scatter(
                x=def_b["nama_bulan"],
                y=(def_b[["ketersediaan_m3", "kebutuhan_m3"]].max(axis=1) / 1e6) * 1.06,
                mode="text", text="▼ defisit", textfont=dict(color=WARNA["merah"], size=11),
                showlegend=False, hoverinfo="skip"))
        fig_n.update_layout(
            barmode="group", height=380, plot_bgcolor="white",
            yaxis_title="Volume (juta m³/bulan)", hovermode="x unified",
            legend=dict(orientation="h", y=-0.2),
            title=dict(text=f"Neraca Air Bulanan — Bendungan {nama}",
                       font=dict(size=15, color=WARNA["navy"])))
        fig_n.update_yaxes(showgrid=True, gridcolor="#EEE")
        st.plotly_chart(fig_n, use_container_width=True)

        st.dataframe(warnai_status(
            lap.tabel_neraca(n_b).drop(columns=["kode_bendungan",
                                                "nama_bendungan", "bulan"]),
            "status_neraca"), width="stretch", hide_index=True)

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
