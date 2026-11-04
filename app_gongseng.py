"""Dashboard Streamlit Gongseng: evaluasi akurasi running Juli dan Agustus terhadap realisasi September.

Jalankan:  streamlit run app_gongseng.py
Deploy  :  Streamlit Community Cloud, Main file path = app_gongseng.py
Data    :  output/prediksi/akurasi_realisasi.parquet dan prediksi_tma_2026.parquet (hasil pipeline, tidak diubah).
Created by Faradilah Ade, internal system of Pusat Monitoring Bendungan.
"""
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

ROOT = os.path.dirname(os.path.abspath(__file__))
KODE, NAMA = "BD034", "GONGSENG"
BLN = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
BLN_PENUH = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus",
             "September", "Oktober", "November", "Desember"]
RUN_COLOR = {"2026-07": "#fb923c", "2026-08": "#a3e635", "2026-09": "#c084fc"}
C_REAL, C_BONB, C_BONA = "#22d3ee", "#fb7185", "#fbbf24"

st.set_page_config(page_title="Gongseng, Akurasi Model", page_icon="🌊", layout="wide")
st.markdown("""<style>
.stApp{background:radial-gradient(1000px 500px at 85% -10%,#0f4c5c66,transparent),#06121c;color:#eaf6fb}
html,body,[class*="css"]{font-family:'Segoe UI',system-ui,sans-serif}
h1,h2,h3,p,li,label,span,div{color:#eaf6fb}
.block-container{padding-top:1.2rem;max-width:1400px}
.kp{background:linear-gradient(180deg,#11324a,#0d2638);border:1px solid #1f4560;border-radius:16px;padding:14px 18px;height:100%}
.kp .l{color:#8fb3c7;font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.8px}
.kp .v{font-size:32px;font-weight:800;margin:2px 0}
.kp .n{color:#8fb3c7;font-size:13px}
.ok{color:#34d399!important}.bad{color:#fb7185!important}
.an{background:#0d2638;border:1px solid #1f4560;border-left:5px solid #22d3ee;border-radius:12px;padding:14px 18px;margin:8px 0;font-size:15px;line-height:1.6}
.cr{color:#8fb3c7;font-size:13px;border-top:1px solid #1f4560;margin-top:24px;padding-top:10px}
</style>""", unsafe_allow_html=True)


@st.cache_data
def muat():
    a = pd.read_parquet(os.path.join(ROOT, "output", "prediksi", "akurasi_realisasi.parquet"))
    a = a[a.kode_bendungan == KODE].copy()
    p = pd.read_parquet(os.path.join(ROOT, "output", "prediksi", "prediksi_tma_2026.parquet"))
    p = p[p.kode_bendungan == KODE].copy()
    return a.sort_values(["run_id", "tanggal"]), p.sort_values("tanggal")


def pl(l):
    return f"{BLN[int(l[:2]) - 1]} {['I', 'II', 'III'][int(l[3:]) - 1]}"


def rn(r):
    return "Running " + BLN_PENUH[int(r[-2:]) - 1]


def f(v, d=2):
    return f"{v:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def kartu(col, l, v, n, cls=""):
    col.markdown(f'<div class="kp"><div class="l">{l}</div><div class="v {cls}">{v}</div><div class="n">{n}</div></div>',
                 unsafe_allow_html=True)


def layout(fig, h=380):
    fig.update_layout(height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0a1f2e",
                      font=dict(color="#eaf6fb", size=14), margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.12), hovermode="x unified")
    fig.update_xaxes(gridcolor="#1f4560")
    fig.update_yaxes(gridcolor="#1f4560")
    return fig


acc, pred = muat()
st.markdown(f"## 🌊 Bendungan {NAMA.title()}: Apakah model LSTM membaik?")
st.caption("Prediksi running terdahulu dibandingkan dengan realisasi TMA. Fokus: running Juli dan Agustus terhadap realisasi September.")

bulan_tersedia = sorted({int(p[:2]) for p in acc.periode} & {int(p[:2]) for p in acc[acc.tma_realisasi.notna()].periode})
c1, c2 = st.columns([1, 3])
bf = c1.selectbox("Bulan realisasi yang dievaluasi", bulan_tersedia, index=bulan_tersedia.index(9) if 9 in bulan_tersedia else 0,
                  format_func=lambda m: BLN_PENUH[m - 1])
d = acc[acc.periode.str.startswith(f"{bf:02d}-") & acc.tma_realisasi.notna()].copy()
runs = sorted(d.run_id.unique())
if len(runs) < 2:
    st.warning("Belum ada dua running atau lebih yang mencakup bulan ini.")
    st.stop()
d["ae"] = d.error_m.abs()
d["lbl"] = d.run_id.map(rn)
rk = d.groupby("run_id").agg(mae=("ae", "mean"), bias=("error_m", "mean"), mx=("ae", "max"),
                             sesuai=("status_sesuai", "mean"), n=("ae", "size"))
rk["rmse"] = d.groupby("run_id").error_m.apply(lambda s: float(np.sqrt((s ** 2).mean())))
a_run, b_run = runs[0], runs[-1]
perb = (1 - rk.loc[b_run, "mae"] / rk.loc[a_run, "mae"]) * 100
nm_bulan = BLN_PENUH[bf - 1]

k = st.columns(4)
kartu(k[0], f"MAE {rn(a_run)}", f"{f(rk.loc[a_run, 'mae'])} m", f"Bias {f(rk.loc[a_run, 'bias'])} m, galat maks {f(rk.loc[a_run, 'mx'])} m")
kartu(k[1], f"MAE {rn(b_run)}", f"{f(rk.loc[b_run, 'mae'])} m", f"Bias {f(rk.loc[b_run, 'bias'])} m, galat maks {f(rk.loc[b_run, 'mx'])} m")
kartu(k[2], "Perubahan galat", f"{abs(perb):.0f}% {'lebih kecil' if perb > 0 else 'lebih besar'}",
      f"{rn(b_run)} dibanding {rn(a_run)}", "ok" if perb > 0 else "bad")
kartu(k[3], "Status BON sesuai", f"{rk.loc[b_run, 'sesuai'] * 100:.0f}%", f"{rn(a_run)}: {rk.loc[a_run, 'sesuai'] * 100:.0f}% dari {int(rk.loc[a_run, 'n'])} periode")

st.markdown(f"### Prediksi dan realisasi {nm_bulan} 2026")
per = sorted(d.periode.unique())
x = [pl(p) for p in per]
real = d.drop_duplicates("periode").set_index("periode").reindex(per)
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=real.bon_a, name="BON A", line=dict(color=C_BONA, dash="dash", width=2)))
fig.add_trace(go.Scatter(x=x, y=real.bon_b, name="BON B", line=dict(color=C_BONB, dash="dash", width=2.5)))
fig.add_trace(go.Scatter(x=x, y=real.tma_realisasi, name="Realisasi", line=dict(color=C_REAL, width=4),
                         mode="lines+markers", marker=dict(size=10)))
for r in runs:
    q = d[d.run_id == r].set_index("periode").reindex(per)
    fig.add_trace(go.Scatter(x=x, y=q.tma_prediksi, name=rn(r), mode="lines+markers",
                             line=dict(color=RUN_COLOR.get(r, "#fff"), width=3), marker=dict(size=11, symbol="diamond")))
st.plotly_chart(layout(fig, 400), use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    st.markdown("#### Galat per periode (meter)")
    fb = go.Figure()
    for r in runs:
        q = d[d.run_id == r].set_index("periode").reindex(per)
        fb.add_trace(go.Bar(x=x, y=q.error_m, name=rn(r), marker_color=RUN_COLOR.get(r, "#fff"),
                            text=[f(v) if pd.notna(v) else "" for v in q.error_m], textposition="outside"))
    fb.add_hline(y=0, line_color="#8fb3c7")
    fb.update_layout(barmode="group")
    fb.update_yaxes(title="Prediksi dikurangi realisasi (m)")
    st.plotly_chart(layout(fb, 340), use_container_width=True)
with c2:
    st.markdown("#### Tabel rinci")
    t = d.pivot_table(index="periode", columns="run_id", values="error_m").reindex(per)
    t.columns = [rn(c) for c in t.columns]
    t.insert(0, "Realisasi (m)", real.tma_realisasi.values)
    t.insert(1, "BON B (m)", real.bon_b.values)
    t.index = [pl(p) for p in t.index]
    st.dataframe(t.round(3), use_container_width=True)
    st.caption("Nilai galat positif berarti model memprediksi lebih tinggi daripada realisasi.")

# ===== analisis BON B =====
st.markdown("### Analisis terhadap BON B")
real_min = real.tma_realisasi.min()
jarak = real.tma_realisasi - real.bon_b
rows = []
for r in runs:
    q = d[d.run_id == r].set_index("periode").reindex(per)
    rows.append({"Running": rn(r),
                 "Prediksi terendah (m)": q.tma_prediksi.min(),
                 "Jarak prediksi ke BON B terkecil (m)": (q.tma_prediksi - q.bon_b).min(),
                 "Jarak realisasi ke BON B terkecil (m)": (q.tma_realisasi - q.bon_b).min(),
                 "Selisih jarak ke BON B (m)": (q.tma_prediksi - q.bon_b).min() - (q.tma_realisasi - q.bon_b).min(),
                 "Status BON sesuai": f"{q.status_sesuai.mean() * 100:.0f}%",
                 "Prediksi di bawah BON B": int((q.tma_prediksi < q.bon_b).sum())})
st.dataframe(pd.DataFrame(rows).round(3).set_index("Running"), use_container_width=True)

lb = rk.loc[b_run]
melewati = (real.tma_realisasi < real.bon_b).any()
lebih_tinggi = {r: bool((d[d.run_id == r].error_m > 0).mean() > .5) for r in runs}
txt = [
    f"<b>Ringkasan.</b> Untuk {nm_bulan}, MAE {rn(a_run)} sebesar {f(rk.loc[a_run, 'mae'])} m dan {rn(b_run)} sebesar {f(lb.mae)} m, "
    f"jadi galat {'turun' if perb > 0 else 'naik'} {abs(perb):.0f}%. Bias {rn(a_run)} {f(rk.loc[a_run, 'bias'])} m dan {rn(b_run)} {f(lb.bias)} m "
    f"({'prediksi cenderung lebih tinggi dari realisasi, artinya terlalu optimis' if lb.bias > 0 else 'prediksi cenderung lebih rendah dari realisasi'}).",
    f"<b>BON B.</b> Realisasi terendah {nm_bulan} adalah {f(real_min)} m, sebesar {f(jarak.min())} m "
    f"{'di atas' if jarak.min() >= 0 else 'di bawah'} BON B. "
    + ("Realisasi tidak menembus BON B, sehingga seluruh periode tetap berstatus di antara BON A dan BON B. " if not melewati
       else "Realisasi sempat menembus BON B. ")
    + f"Karena model memprediksi lebih tinggi, jarak ke BON B pada prediksi terlihat lebih lebar dari kenyataan. "
    f"Pada {rn(a_run)} jarak itu selisih {f(rows[0]['Selisih jarak ke BON B (m)'])} m, pada {rn(b_run)} selisih {f(rows[-1]['Selisih jarak ke BON B (m)'])} m. "
    "Selisih yang kecil berarti peringatan dini ke BON B lebih dapat dipercaya.",
    "<b>Catatan keadilan.</b> Running Agustus memprediksi September dari horizon yang lebih pendek daripada running Juli, sehingga sebagian perbaikan "
    "berasal dari horizon yang lebih dekat, bukan hanya dari model yang lebih baik. Galat terbesar kedua running ada pada akhir September, saat TMA turun cepat "
    "dan model masih menahan nilai tinggi."]
for t_ in txt:
    st.markdown(f'<div class="an">{t_}</div>', unsafe_allow_html=True)

# ===== tren seluruh running =====
with st.expander("Tren perbaikan antar running pada periode yang sama (Oktober sampai data terakhir)"):
    ok = acc[acc.tma_realisasi.notna()].copy()
    ok["ae"] = ok.error_m.abs()
    cnt = ok.groupby("periode").run_id.nunique()
    com = [p for p in cnt[cnt >= 2].index if p.startswith("10-")]
    if com:
        z = ok[ok.periode.isin(com)]
        zz = z.groupby("run_id").agg(MAE=("ae", "mean"), Bias=("error_m", "mean")).round(3)
        zz.index = [rn(i) for i in zz.index]
        st.caption("Periode pembanding: " + ", ".join(pl(c) for c in sorted(com)) + ". Semua running dinilai pada periode yang sama.")
        st.dataframe(zz, use_container_width=True)
        fz = go.Figure(go.Bar(x=list(zz.index), y=zz.MAE, marker_color=[RUN_COLOR.get(r, "#fff") for r in sorted(z.run_id.unique())],
                              text=[f(v) for v in zz.MAE], textposition="outside"))
        fz.update_yaxes(title="MAE (m)")
        st.plotly_chart(layout(fz, 300), use_container_width=True)
    else:
        st.info("Belum ada periode yang tercakup oleh dua running atau lebih.")

st.markdown('<div class="cr">Created by <b>Faradilah Ade</b> · Internal system of Pusat Monitoring Bendungan · Sumber data SINBAD dan model LSTM Encoder Decoder</div>',
            unsafe_allow_html=True)
