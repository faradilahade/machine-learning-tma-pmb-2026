"""Membuat template data_bon/data_bon_a_bon_b.xlsx dan daftar_bendungan.csv.

Daftar bendungan yang dimodelkan HANYA yang tercantum di file BON ini.
Nilai contoh diisi 5 bendungan dummy -- ganti dengan data BON resmi per bendungan.
"""
import os
import sys
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import path_root, load_config

BULAN = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun",
         "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]

# Contoh 5 bendungan (GANTI dengan data resmi; kode = kode integrasi SINBAD)
CONTOH = [
    # kode, nama, elevasi dasar BON A Jan (mdpl), amplitudo musiman
    ("BD001", "Jatigede",    258.0, 4.0),
    ("BD002", "Jatiluhur",   105.0, 3.5),
    ("BD003", "Kedung Ombo",  88.0, 3.0),
    ("BD004", "Sermo",       134.0, 2.5),
    ("BD005", "Batutegi",    268.0, 4.5),
]

HEADER_FILL = PatternFill("solid", fgColor="1C5D8C")   # biru Ocean Gradient
HEADER_FONT = Font(name="Arial", bold=True, color="FFFFFF", size=10)
BODY_FONT = Font(name="Arial", size=10)
NOTE_FONT = Font(name="Arial", size=9, italic=True, color="808080")
INPUT_FILL = PatternFill("solid", fgColor="FFF2CC")    # kuning muda = sel isian
BORDER = Border(*[Side(style="thin", color="D9D9D9")] * 4)


def profil_bon(dasar: float, amp: float, offset: float):
    """Profil BON musiman sederhana: tinggi di musim hujan, turun di kemarau."""
    import math
    return [round(dasar + offset + amp * math.cos((b - 3) * 2 * math.pi / 12), 2)
            for b in range(1, 13)]


def tulis_sheet(ws, judul, offset):
    ws.append(["kode_bendungan", "nama_bendungan"] + BULAN)
    for c in ws[1]:
        c.fill, c.font, c.border = HEADER_FILL, HEADER_FONT, BORDER
        c.alignment = Alignment(horizontal="center")
    for kode, nama, dasar, amp in CONTOH:
        ws.append([kode, nama] + profil_bon(dasar, amp, offset))
    for row in ws.iter_rows(min_row=2, max_row=1 + len(CONTOH)):
        for c in row:
            c.font, c.border = BODY_FONT, BORDER
            if c.column >= 3:
                c.fill = INPUT_FILL
                c.number_format = "0.00"
    ws.freeze_panes = "C2"
    ws.column_dimensions["A"].width = 16
    ws.column_dimensions["B"].width = 22
    for col in "CDEFGHIJKLMN":
        ws.column_dimensions[col].width = 9
    ws.auto_filter.ref = f"A1:N{1 + len(CONTOH)}"


def main():
    cfg = load_config()
    wb = Workbook()

    # Sheet petunjuk
    ws0 = wb.active
    ws0.title = "PETUNJUK"
    petunjuk = [
        "TEMPLATE DATA BON A / BON B (Batas Operasi Normal) - satuan mdpl",
        "",
        "1. Sheet BON_A dan BON_B: satu baris per bendungan, 12 kolom bulan (Jan-Des).",
        "2. Sel berlatar KUNING adalah sel isian - ganti nilai contoh dengan data BON resmi.",
        "3. kode_bendungan HARUS sama dengan kode integrasi di database SINBAD.",
        "4. Daftar bendungan yang dimodelkan = HANYA bendungan yang tercantum di file ini.",
        "5. Baris contoh (Jatigede dst.) hanya ilustrasi format - hapus/ganti dengan data resmi.",
        "6. Setelah mengubah file ini, jalankan: python src/buat_daftar_dari_bon.py",
        "   untuk memperbarui daftar_bendungan.csv secara otomatis.",
    ]
    for i, t in enumerate(petunjuk, 1):
        ws0.cell(row=i, column=1, value=t).font = (
            Font(name="Arial", bold=True, size=11, color="0F2A47") if i == 1 else BODY_FONT
        )
    ws0.column_dimensions["A"].width = 100

    tulis_sheet(wb.create_sheet("BON_A"), "BON A", offset=0.0)
    tulis_sheet(wb.create_sheet("BON_B"), "BON B", offset=-2.5)  # BON B di bawah BON A

    out_xlsx = path_root(cfg["bon"]["file_bon"])
    os.makedirs(os.path.dirname(out_xlsx), exist_ok=True)
    wb.save(out_xlsx)

    # daftar_bendungan.csv diturunkan dari file BON (fokus nama bendungan di BON)
    df = pd.DataFrame([(k, n) for k, n, _, _ in CONTOH],
                      columns=["kode_bendungan", "nama_bendungan"])
    df.to_csv(path_root(cfg["bon"]["file_daftar"]), index=False)
    print(f"OK  {out_xlsx}")
    print(f"OK  {path_root(cfg['bon']['file_daftar'])}")


if __name__ == "__main__":
    main()
