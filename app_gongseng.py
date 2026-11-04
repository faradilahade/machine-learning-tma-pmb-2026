"""Dashboard Streamlit Bendungan Gongseng: pemantauan, prediksi LSTM, dan evaluasi akurasi running.

Jalankan:  streamlit run app_gongseng.py
Deploy  :  Streamlit Community Cloud, Main file path = app_gongseng.py
Data    :  output/prediksi/*.parquet, data/processed/tma_periode.parquet, data_bon/sifat_musim.csv,
           data_gongseng/ref_gongseng.json (dibuat oleh buat_data_gongseng.py). Tidak ada yang diubah.
Created by Faradilah Ade, internal system of Pusat Monitoring Bendungan.
"""
import os
import json
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

ROOT = os.path.dirname(os.path.abspath(__file__))
KODE, NAMA = "BD034", "GONGSENG"
BLN = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
BLN_L = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September",
         "Oktober", "November", "Desember"]
A_, B_, AB_ = "Di atas BON A", "Di bawah BON B", "Di antara BON A dan BON B"
C = dict(real="#22d3ee", pred="#a3e635", bona="#fbbf24", bonb="#fb7185", rtow="#a78bfa",
         kering="#ef4444", dasar="#a16207", thn="#94a3b8", ket="#14b8a6", keb="#fb923c")
RUN_C = {"2026-07": "#fb923c", "2026-08": "#a3e635", "2026-09": "#c084fc", "2026-10": "#38bdf8"}

st.set_page_config(page_title="Gongseng, Prediksi TMA", page_icon="🌊", layout="wide")
st.markdown("""<style>
.stApp{background:radial-gradient(1000px 500px at 85% -10%,#0f4c5c66,transparent),#06121c}
.stApp,.stApp p,.stApp li,.stApp label,.stApp span,.stApp h1,.stApp h2,.stApp h3,.stApp h4{color:#eaf6fb}
header[data-testid=stHeader]{background:#06121c}
[data-testid=stToolbar] *,[data-testid=stHeader] button,[data-testid=stHeader] svg{color:#8fb3c7!important;fill:#8fb3c7}
.block-container{padding-top:3.2rem;max-width:1400px}
[data-testid=stMetric]{background:linear-gradient(180deg,#11324a,#0d2638);border:1px solid #1f4560;border-radius:14px;padding:12px 16px}
[data-testid=stMetricLabel] p{color:#8fb3c7!important;font-weight:700}
[data-testid=stMetricValue],[data-testid=stMetricValue] *{font-weight:800;color:#ffffff!important}
[data-testid=stMetricDelta],[data-testid=stMetricDelta] *{color:#a7e3f2!important;font-weight:600}
.stDownloadButton button{background:#0f766e;border:0}.stDownloadButton button,.stDownloadButton button *{color:#fff!important;font-weight:700}
.an{background:#0d2638;border:1px solid #1f4560;border-left:5px solid #22d3ee;border-radius:12px;padding:12px 16px;margin:8px 0;line-height:1.6}
.bd{padding:3px 14px;border-radius:999px;font-weight:700;color:#06121c}
.cr{color:#8fb3c7;font-size:13px;border-top:1px solid #1f4560;margin-top:24px;padding-top:10px}
button[data-baseweb=tab]{font-weight:700}
.tw{max-height:380px;overflow:auto;border:1px solid #1f4560;border-radius:12px;margin:6px 0 18px}
.tw table{border-collapse:collapse;width:100%;font-size:14px}
.tw th{position:sticky;top:0;background:#11324a;color:#8fb3c7;text-align:left;padding:8px 12px;font-weight:700;white-space:nowrap}
.tw td{padding:7px 12px;border-top:1px solid #1f4560;color:#eaf6fb;white-space:nowrap}
.tw tr:hover td{background:#14405c}
.sp{height:10px}
</style>""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def muat():
    P = os.path.join(ROOT, "output", "prediksi")
    flt = [("kode_bendungan", "==", KODE)]
    pred = pd.read_parquet(os.path.join(P, "prediksi_tma_2026.parquet"), filters=flt).sort_values("tanggal")
    acc = pd.read_parquet(os.path.join(P, "akurasi_realisasi.parquet"), filters=flt).sort_values(["run_id", "tanggal"])
    ev = pd.read_parquet(os.path.join(P, "evaluasi_metode.parquet"), filters=flt)
    hist = pd.read_parquet(os.path.join(ROOT, "data", "processed", "tma_periode.parquet"), filters=flt)
    ref = json.load(open(os.path.join(ROOT, "data_gongseng", "ref_gongseng.json")))
    sm = pd.read_csv(os.path.join(ROOT, "data_bon", "sifat_musim.csv"), sep=";")
    sm = sm[sm.nama_bendungan == NAMA].iloc[0].drop(["id", "nama_bendungan"]).to_dict()
    return pred, acc, ev, hist, ref, sm


def pl(l):
    return f"{BLN[int(l[:2]) - 1]} {['I', 'II', 'III'][int(l[3:]) - 1]}"


def rn(r):
    return "Running " + BLN_L[int(r[-2:]) - 1]


def f(v, d=2):
    return f"{v:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def html_tbl(df, idx=True, fmt=3):
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f(v, fmt))
    st.markdown('<div class="tw">' + d.to_html(index=idx, border=0, escape=True) + '</div><div class="sp"></div>', unsafe_allow_html=True)


def tema(fig, h):
    fig.update_layout(height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0a1f2e", font=dict(color="#eaf6fb", size=14),
                      title_font=dict(color="#ffffff", size=18), hoverlabel=dict(bgcolor="#04101a", font=dict(color="#ffffff", size=14)),
                      margin=dict(l=10, r=10, t=50, b=10), hovermode="x unified", legend=dict(orientation="h", y=-0.25, font=dict(color="#eaf6fb", size=14), bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(gridcolor="#1f4560", tickfont=dict(color="#cfe6f2"), title_font=dict(color="#eaf6fb"))
    fig.update_yaxes(automargin=True, gridcolor="#1f4560", tickfont=dict(color="#cfe6f2"), title_font=dict(color="#eaf6fb"))
    return fig


pred, acc, ev, hist, ref, sm = muat()
# BON A, BON B, dan RTOW per periode dari workbook RTOW periodik (sama dengan dashboard DDEWOMS)
for k_ in ("bon_a", "bon_b", "rtow"):
    pred[k_] = pred.periode.map(ref[k_])
pred["status_bon"] = np.where(pred.tma >= pred.bon_a, A_, np.where(pred.tma >= pred.bon_b, AB_, B_))
urut = list(pred.periode)
xl = [pl(p) for p in urut]
real, fc = pred[pred.jenis == "realisasi"], pred[pred.jenis == "prediksi"]
tma_now = float(real.tma.iloc[-1])
lm = ev[ev.metode.str.startswith("LSTM")]
mae = float(lm.error.abs().mean())
hari = pd.Timestamp(dt.date.today())
cur = pred[pred.tanggal <= hari].iloc[-1]

st.markdown(f"## 🌊 Bendungan {NAMA.title()}: Prediksi Adaptif TMA dengan LSTM Encoder Decoder")
st.caption("Mitigasi kekeringan El Nino · BBWS Bengawan Solo · Pusat Monitoring Bendungan")
tab1, tab2 = st.tabs(["📈 Pemantauan dan Prediksi", "🎯 Akurasi Model Juli dan Agustus"])

# ======================================================== TAB 1
with tab1:
    o1, o2, o3 = st.columns([1.3, 1.3, 2.4])
    show25 = o1.toggle("Bandingkan dengan TMA 2025", help="Data sensor 2025 valid, sebelum pertengahan 2024 konstan sehingga tidak dipakai")
    showband = o2.toggle("Pita galat model", value=True, help="Rata rata galat absolut LSTM pada backtest")
    r0, r1 = o3.select_slider("Fokus periode", options=xl, value=(xl[0], xl[-1]))
    fig = go.Figure()
    if show25:
        h = hist[hist.tanggal.dt.year == 2025].set_index("periode").tma.reindex(urut)
        fig.add_trace(go.Scatter(x=xl, y=h.values, name="TMA 2025", line=dict(color=C["thn"], width=2, shape="spline", smoothing=0.6), opacity=.8))
    fig.add_trace(go.Scatter(x=xl, y=pred.rtow, name="RTOW rencana operasi", line=dict(color=C["rtow"], width=2.5, dash="dot", shape="spline", smoothing=0.6)))
    fig.add_trace(go.Scatter(x=xl, y=pred.bon_a, name="BON A", line=dict(color=C["bona"], width=2.5, dash="dashdot", shape="spline", smoothing=0.6)))
    fig.add_trace(go.Scatter(x=xl, y=pred.bon_b, name="BON B", line=dict(color=C["bonb"], width=2.5, dash="dashdot", shape="spline", smoothing=0.6)))
    fig.add_trace(go.Scatter(x=xl, y=[ref["kekeringan"]] * len(xl), name="Elevasi kekeringan RTOW",
                             line=dict(color=C["kering"], width=2, dash="longdash")))
    fig.add_trace(go.Scatter(x=xl, y=[ref["dasar"]] * len(xl), name="Elevasi dasar waduk",
                             line=dict(color=C["dasar"], width=1.5, dash="longdashdot")))
    xr = [pl(p) for p in real.periode]
    fig.add_trace(go.Scatter(x=xr, y=real.tma, name="TMA realisasi 2026", mode="lines+markers",
                             line=dict(color=C["real"], width=4, shape="spline", smoothing=0.6), marker=dict(size=7)))
    xp = [xr[-1]] + [pl(p) for p in fc.periode]
    yp = [tma_now] + list(fc.tma)
    if showband:
      fig.add_trace(go.Scatter(x=xp, y=np.array(yp) + mae, line=dict(width=0), showlegend=False, hoverinfo="skip"))
      fig.add_trace(go.Scatter(x=xp, y=np.array(yp) - mae, line=dict(width=0), fill="tonexty", fillcolor="rgba(163,230,53,.18)",
                             name=f"Pita galat historis ± {f(mae)} m", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xp, y=yp, name="TMA prediksi LSTM", mode="lines+markers",
                             line=dict(color=C["pred"], width=4, dash="dash", shape="spline", smoothing=0.6),
                             marker=dict(size=11, symbol="circle", line=dict(color="#06121c", width=2))))
    krit = fc[fc.status_bon == B_]
    if len(krit):
        fig.add_trace(go.Scatter(x=[pl(p) for p in krit.periode], y=krit.tma, mode="markers", name="Di bawah BON B",
                                 marker=dict(size=18, symbol="circle-open", line=dict(color=C["bonb"], width=3)), hoverinfo="skip"))
    fig.add_vline(x=xr[-1], line=dict(color="#94a3b8", dash="dot"))
    fig.add_annotation(x=xr[-1], yref="paper", y=1.06, text="realisasi | prediksi", showarrow=False, font=dict(size=12, color="#8fb3c7"))
    fig.update_layout(title=dict(text="Sandingan TMA per Periode 10 Harian 2026, Bendungan Gongseng"),
                      yaxis_title="TMA (mdpl)", xaxis_title="Periode (bulan dan dasarian)")
    fig.update_xaxes(type="category", categoryorder="array", categoryarray=xl, tickangle=-90, range=[xl.index(r0) - .5, xl.index(r1) + .5])
    st.plotly_chart(tema(fig, 560), use_container_width=True, theme=None)

    k = st.columns(4)
    k[0].metric(f"TMA realisasi terakhir ({pl(real.periode.iloc[-1])})", f"{f(tma_now)} mdpl")
    k[1].metric(f"TMA awal prediksi ({pl(fc.periode.iloc[0])})", f"{f(fc.tma.iloc[0])} mdpl", f"{fc.tma.iloc[0] - tma_now:+.2f} m".replace(".", ","))
    k[2].metric(f"TMA akhir Desember ({pl(fc.periode.iloc[-1])})", f"{f(fc.tma.iloc[-1])} mdpl", f"{fc.tma.iloc[-1] - tma_now:+.2f} m".replace(".", ","))
    nb = int((fc.tma < fc.bon_a).sum())
    k[3].metric("Periode prediksi di bawah BON A", f"{nb} dari {len(fc)}")

    z = A_ if tma_now >= cur.bon_a else (AB_ if tma_now >= cur.bon_b else B_)
    wz = {A_: "#34d399", AB_: "#fbbf24", B_: "#fb7185"}[z]
    ik = {A_: "✅", AB_: "⚠️", B_: "🚨"}[z]
    st.markdown(f"**Rekomendasi 3 Zona Operasi Waduk, periode berjalan {pl(cur.periode)}:** <span class='bd' style='background:{wz}'>{ik} {z}</span> "
                f"&nbsp;<small>TMA terkini {f(tma_now)} mdpl · BON A {f(cur.bon_a)} mdpl · BON B {f(cur.bon_b)} mdpl</small>", unsafe_allow_html=True)

    REK = {A_: "Pertahankan pola operasi sesuai RTOW dan lakukan pemantauan rutin.",
           AB_: "Hemat pelepasan, prioritaskan kebutuhan pokok, laporkan pembacaan setiap hari.",
           B_: "Batasi pelepasan air, lapor ke koordinator, dan terapkan pola operasi darurat kekeringan."}
    st.info(f"**Tindakan:** {REK[z]}")
    g1, g2 = st.columns(2)
    with g1:
        lo_, hi_ = ref["kekeringan"] - .5, max(cur.bon_a, tma_now) + .8
        gg = go.Figure(go.Indicator(mode="gauge+number+delta", value=tma_now, number=dict(suffix=" mdpl", font=dict(color="#fff")),
                                    delta=dict(reference=cur.bon_b, suffix=" m dari BON B", valueformat=".2f"),
                                    title=dict(text="Posisi TMA terhadap BON", font=dict(color="#fff", size=16)),
                                    gauge=dict(axis=dict(range=[lo_, hi_], tickcolor="#cfe6f2"), bar=dict(color=C["real"], thickness=.28),
                                               steps=[dict(range=[lo_, cur.bon_b], color="#7f1d1d"), dict(range=[cur.bon_b, cur.bon_a], color="#78350f"),
                                                      dict(range=[cur.bon_a, hi_], color="#064e3b")],
                                               threshold=dict(line=dict(color="#fff", width=4), value=tma_now))))
        gg.update_layout(height=260, paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#eaf6fb"), margin=dict(l=20, r=20, t=50, b=10))
        st.plotly_chart(gg, use_container_width=True, theme=None)
    with g2:
        st.markdown("**Simulasi sensitivitas**")
        dlt = st.slider("Jika TMA aktual lebih rendah dari prediksi (m)", 0.0, 3.0, 0.0, 0.1, help="Uji seberapa cepat status turun bila kemarau lebih ekstrem dari prediksi")
        s_ = fc.tma - dlt
        m1, m2 = st.columns(2)
        m1.metric("Periode di bawah BON A", f"{int((s_ < fc.bon_a).sum())} dari {len(fc)}")
        m2.metric("Periode di bawah BON B", f"{int((s_ < fc.bon_b).sum())} dari {len(fc)}")
        st.caption(f"TMA terendah skenario {f(s_.min())} mdpl, {f(s_.min() - ref['kekeringan'])} m dari elevasi kekeringan {f(ref['kekeringan'], 1)} mdpl.")
        st.download_button("⬇️ Unduh prediksi (CSV)", pred[["periode", "tanggal", "tma", "jenis", "bon_a", "bon_b", "rtow", "status_bon"]].to_csv(index=False).encode("utf8"),
                           "prediksi_gongseng.csv", "text/csv")

    din = pred.dropna(subset=["rtow", "bon_a", "bon_b"])
    luar = din[(din.rtow > din.bon_a + 1e-6) | (din.rtow < din.bon_b - 1e-6)]
    if luar.empty:
        st.success(f"📝 Catatan RTOW: elevasi rencana RTOW berada di antara BON A dan BON B pada seluruh {len(din)} periode.")
    else:
        at = [pl(p) for p in luar[luar.rtow > luar.bon_a + 1e-6].periode]
        bw = [pl(p) for p in luar[luar.rtow < luar.bon_b - 1e-6].periode]
        st.warning(f"📝 Catatan RTOW: pada {len(luar)} dari {len(din)} periode, RTOW berada di luar rentang BON A dan BON B"
                   + (f"; {len(at)} periode di atas BON A ({', '.join(at[:8])}{' ...' if len(at) > 8 else ''})" if at else "")
                   + (f"; {len(bw)} periode di bawah BON B ({', '.join(bw[:8])}{' ...' if len(bw) > 8 else ''})" if bw else "")
                   + ". Nilai perlu diverifikasi ke pengelola bendungan.")

    # ---- hasil analisis
    pm = fc.loc[fc.tma.idxmin()]
    tren = fc.tma.iloc[-1] - fc.tma.iloc[0]
    bb = fc[fc.tma < fc.bon_b]
    ba = fc[fc.tma < fc.bon_a]
    st.markdown("#### 🔎 Hasil Analisis")
    tx = (f"TMA realisasi terakhir ({pl(real.periode.iloc[-1])}) {f(tma_now)} mdpl; prediksi {pl(fc.periode.iloc[0])} {f(fc.tma.iloc[0])} mdpl "
          f"menjadi {pl(fc.periode.iloc[-1])} {f(fc.tma.iloc[-1])} mdpl ({'naik' if tren > 0 else 'turun'} {f(abs(tren))} m). "
          f"TMA minimum prediksi {f(pm.tma)} mdpl pada {pl(pm.periode)}, berjarak {f(pm.tma - ref['kekeringan'])} m dari elevasi kekeringan {f(ref['kekeringan'], 1)} mdpl dan "
          f"{f(pm.tma - pm.bon_b)} m dari BON B. ")
    tx += (f"Prediksi menembus BON B pada {len(bb)} periode (mulai {pl(bb.periode.iloc[0])}), perlu penyesuaian pola operasi." if len(bb)
           else f"Prediksi tetap di atas BON B pada seluruh periode, tetapi di bawah BON A pada {len(ba)} dari {len(fc)} periode, sehingga status waspada perlu dipertahankan." if len(ba)
           else "Prediksi berada di atas BON A sepanjang horizon, pasokan air diperkirakan aman.")
    st.markdown(f'<div class="an">{tx}</div>', unsafe_allow_html=True)
    nA, nAB, nB = int((fc.status_bon == A_).sum()), int((fc.status_bon == AB_).sum()), int((fc.status_bon == B_).sum())
    st.markdown(f"**Rekap periode prediksi:** {nA} di atas BON A · {nAB} di antara BON A dan BON B · {nB} di bawah BON B (dari {len(fc)} periode)")
    st.markdown("**Sifat musim** (fitur pembelajaran LSTM): " + " → ".join(f"{k_}: {v}" for k_, v in sm.items()))
    st.markdown(f"**Akurasi backtest bendungan ini:** MAE LSTM {f(mae)} m rata rata dua periode uji (Agu sampai Des 2025 dan Mar sampai Jul 2026).")

    t = pred[["periode", "tanggal", "tma", "jenis", "bon_a", "bon_b", "rtow", "status_bon"]].copy()
    t["tanggal"] = t.tanggal.dt.strftime("%d %b %Y")
    html_tbl(t.rename(columns={"periode": "Periode", "tanggal": "Tanggal", "tma": "TMA", "jenis": "Jenis", "bon_a": "BON A", "bon_b": "BON B", "rtow": "RTOW", "status_bon": "Status BON"}), idx=False)

    # ---- neraca air
    st.markdown("#### 💧 Neraca air per periode (m³/detik)")
    kt = [ref["ketersediaan"].get(p) for p in urut]
    kb = [ref["kebutuhan"].get(p) for p in urut]
    fn = go.Figure()
    fn.add_trace(go.Bar(x=xl, y=kt, name="Ketersediaan air", marker_color=C["ket"]))
    fn.add_trace(go.Bar(x=xl, y=kb, name="Kebutuhan air", marker_color=C["keb"]))
    fn.update_layout(barmode="group", yaxis_title="m³/detik")
    fn.update_xaxes(type="category", categoryorder="array", categoryarray=xl, tickangle=-90, tickfont=dict(size=10))
    st.plotly_chart(tema(fn, 340), use_container_width=True, theme=None)
    i = urut.index(cur.periode)
    if kt[i] is not None and kb[i] is not None:
        d_ = kt[i] - kb[i]
        st.markdown(f"Periode berjalan {pl(cur.periode)}: ketersediaan {f(kt[i])} m³/detik, kebutuhan {f(kb[i])} m³/detik, "
                    f"{'cukup, surplus' if d_ >= 0 else 'belum cukup, defisit'} {f(abs(d_))} m³/detik.")

# ======================================================== TAB 2
with tab2:
    st.caption("Prediksi running terdahulu dibandingkan realisasi TMA. Positif berarti prediksi lebih tinggi dari realisasi.")
    ok = acc[acc.tma_realisasi.notna()]
    bt = sorted({int(p[:2]) for p in ok.periode})
    bf = st.selectbox("Bulan realisasi yang dievaluasi", bt, index=bt.index(9) if 9 in bt else 0, format_func=lambda m: BLN_L[m - 1])
    d = ok[ok.periode.str.startswith(f"{bf:02d}-")].copy()
    runs = sorted(d.run_id.unique())
    if len(runs) < 2:
        st.warning("Belum ada dua running atau lebih yang mencakup bulan ini.")
        st.stop()
    d["ae"] = d.error_m.abs()
    rk = d.groupby("run_id").agg(mae=("ae", "mean"), bias=("error_m", "mean"), mx=("ae", "max"), ses=("status_sesuai", "mean"), n=("ae", "size"))
    a_r, b_r = runs[0], runs[-1]
    perb = (1 - rk.loc[b_r, "mae"] / rk.loc[a_r, "mae"]) * 100
    nm = BLN_L[bf - 1]
    k = st.columns(4)
    k[0].metric(f"MAE {rn(a_r)}", f"{f(rk.loc[a_r, 'mae'])} m", f"bias {f(rk.loc[a_r, 'bias'])} m", delta_color="off")
    k[1].metric(f"MAE {rn(b_r)}", f"{f(rk.loc[b_r, 'mae'])} m", f"bias {f(rk.loc[b_r, 'bias'])} m", delta_color="off")
    k[2].metric("Perubahan galat", f"{abs(perb):.0f}% {'lebih kecil' if perb > 0 else 'lebih besar'}", f"{rn(b_r)} dibanding {rn(a_r)}",
                delta_color="off")
    k[3].metric("Status BON sesuai", f"{rk.loc[b_r, 'ses'] * 100:.0f}%", f"{rn(a_r)}: {rk.loc[a_r, 'ses'] * 100:.0f}%", delta_color="off")

    per = sorted(d.periode.unique())
    x = [pl(p) for p in per]
    rl = d.drop_duplicates("periode").set_index("periode").reindex(per)
    g = go.Figure()
    g.add_trace(go.Scatter(x=x, y=rl.bon_a, name="BON A", line=dict(color=C["bona"], dash="dashdot", width=2.5)))
    g.add_trace(go.Scatter(x=x, y=rl.bon_b, name="BON B", line=dict(color=C["bonb"], dash="dashdot", width=2.5)))
    g.add_trace(go.Scatter(x=x, y=rl.tma_realisasi, name="Realisasi", mode="lines+markers", line=dict(color=C["real"], width=4), marker=dict(size=10)))
    for r in runs:
        q = d[d.run_id == r].set_index("periode").reindex(per)
        g.add_trace(go.Scatter(x=x, y=q.tma_prediksi, name=rn(r), mode="lines+markers", line=dict(color=RUN_C.get(r, "#fff"), width=3, dash="dash"),
                               marker=dict(size=11, symbol="diamond")))
    g.update_layout(title=f"Prediksi dan realisasi {nm} 2026", yaxis_title="TMA (mdpl)")
    st.plotly_chart(tema(g, 400), use_container_width=True, theme=None)

    c1, c2 = st.columns(2)
    fb = go.Figure()
    for r in runs:
        q = d[d.run_id == r].set_index("periode").reindex(per)
        fb.add_trace(go.Bar(x=x, y=q.error_m, name=rn(r), marker_color=RUN_C.get(r, "#fff"),
                            text=[f(v) if pd.notna(v) else "" for v in q.error_m], textposition="outside", cliponaxis=False))
    fb.update_layout(barmode="group", title="Galat per periode (m)")
    fb.update_yaxes(range=[min(-.6, d.error_m.min() - .5), d.error_m.max() + .8])
    c1.plotly_chart(tema(fb, 340), use_container_width=True, theme=None)
    tb = d.pivot_table(index="periode", columns="run_id", values="error_m").reindex(per)
    tb.columns = [rn(c) for c in tb.columns]
    tb.insert(0, "Realisasi (m)", rl.tma_realisasi.values)
    tb.insert(1, "BON B (m)", rl.bon_b.values)
    tb.index = x
    c2.markdown("**Tabel rinci (galat dalam meter)**")
    with c2:
        html_tbl(tb)

    st.markdown("#### Analisis terhadap BON B")
    rows = []
    for r in runs:
        q = d[d.run_id == r].set_index("periode").reindex(per)
        rows.append({"Running": rn(r), "Prediksi terendah (m)": q.tma_prediksi.min(),
                     "Jarak prediksi ke BON B terkecil (m)": (q.tma_prediksi - q.bon_b).min(),
                     "Jarak realisasi ke BON B terkecil (m)": (q.tma_realisasi - q.bon_b).min(),
                     "Prediksi di bawah BON B": int((q.tma_prediksi < q.bon_b).sum())})
    rt = pd.DataFrame(rows).set_index("Running")
    rt["Selisih jarak (m)"] = rt.iloc[:, 1] - rt.iloc[:, 2]
    html_tbl(rt)
    jr = (rl.tma_realisasi - rl.bon_b).min()
    for t_ in [
        f"<b>Ringkasan.</b> Untuk {nm}, MAE {rn(a_r)} {f(rk.loc[a_r, 'mae'])} m dan {rn(b_r)} {f(rk.loc[b_r, 'mae'])} m, galat "
        f"{'turun' if perb > 0 else 'naik'} {abs(perb):.0f}%. Bias {rn(b_r)} {f(rk.loc[b_r, 'bias'])} m "
        f"({'prediksi cenderung terlalu tinggi' if rk.loc[b_r, 'bias'] > 0 else 'prediksi cenderung terlalu rendah'}).",
        f"<b>BON B.</b> Realisasi terendah {nm} {f(rl.tma_realisasi.min())} m, {f(abs(jr))} m {'di atas' if jr >= 0 else 'di bawah'} BON B. "
        f"{'Tidak ada periode yang menembus BON B, status tetap di antara BON A dan BON B. ' if jr >= 0 else 'Realisasi sempat menembus BON B. '}"
        f"Selisih jarak ke BON B antara prediksi dan kenyataan {f(rt.iloc[0, -1])} m pada {rn(a_r)} dan {f(rt.iloc[-1, -1])} m pada {rn(b_r)}; "
        "makin kecil selisih, makin dapat dipercaya peringatan dini ke BON B.",
        "<b>Catatan keadilan.</b> Running yang lebih baru memprediksi dari horizon yang lebih pendek, sehingga sebagian perbaikan berasal dari horizon, "
        "bukan hanya dari model."]:
        st.markdown(f'<div class="an">{t_}</div>', unsafe_allow_html=True)

    with st.expander("Tren perbaikan antar running pada periode yang sama"):
        cnt = ok.groupby("periode").run_id.nunique()
        com = sorted(p for p in cnt[cnt >= 2].index if p.startswith("10-"))
        if com:
            z = ok[ok.periode.isin(com)]
            zz = z.groupby("run_id").agg(MAE=("error_m", lambda s: s.abs().mean()), Bias=("error_m", "mean")).round(3)
            zz.index = [rn(i) for i in zz.index]
            st.caption("Periode pembanding: " + ", ".join(pl(c) for c in com))
            html_tbl(zz)
        else:
            st.info("Belum ada periode yang tercakup oleh dua running atau lebih.")

st.markdown('<div class="cr">Created by <b>Faradilah Ade</b> · Internal system of Pusat Monitoring Bendungan · Sumber data SINBAD dan model LSTM Encoder Decoder</div>',
            unsafe_allow_html=True)
