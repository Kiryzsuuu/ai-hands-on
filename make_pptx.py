"""Generate a beginner-friendly AI learning module as a PPTX deck.

Run (Windows PowerShell):
  & "./.venv/Scripts/python.exe" make_pptx.py

Output:
  AI_Learning_Modul.pptx
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "AI_Learning_Modul.pptx"


# ══════════════════════════════════════════════════════════════════════
# Layout grid  (symmetric on 13.33 × 7.5 widescreen)
# ══════════════════════════════════════════════════════════════════════
SLIDE_W = 13.333
SLIDE_H = 7.5

LM   = 0.67            # left (and right) margin
BW   = 12.0            # body width  →  right edge = 12.67 ≈ symmetric
TT   = 0.35            # title top
BT   = 1.65            # body top (below title)
NT   = 6.45            # note-bar top
NH   = 0.72            # note-bar height
CG   = 0.50            # column gap
CW   = (BW - CG) / 2  # column width  = 5.75
C2   = LM + CW + CG   # 2nd column left = 6.92


# ══════════════════════════════════════════════════════════════════════
# Professional modern theme
# ══════════════════════════════════════════════════════════════════════
FONT      = "Segoe UI"          # clean, modern, ships with Windows & Office
FONT_HDR  = "Segoe UI Semibold" # bolder weight for titles

BG      = RGBColor(255, 255, 255)   # clean white
INK     = RGBColor( 33,  37,  41)   # dark charcoal text
MUTED   = RGBColor(108, 117, 125)   # secondary gray
ACCENT  = RGBColor(  0, 102, 204)   # corporate blue (titles)
NOTE_C  = RGBColor(232, 245, 233)   # subtle green tint
FLOW_C  = RGBColor(227, 242, 253)   # subtle blue tint
CODE_C  = RGBColor(248, 249, 250)   # near-white gray
BORDER  = RGBColor(206, 212, 218)   # soft gray border
ACCBAR  = RGBColor(  0, 102, 204)   # accent bar color


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

_KEEP = {"Consolas", "Courier New", "Menlo", "Monaco", FONT, FONT_HDR}


def _no_line(shape):
    """Remove any visible border from a shape."""
    shape.line.fill.background()


def _bg(slide):
    f = slide.background.fill; f.solid(); f.fore_color.rgb = BG


def _accent_bar(slide):
    """Add a thin accent bar at the top of the slide for modern look."""
    bar = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE,
                                 Inches(0), Inches(0), Inches(SLIDE_W), Inches(0.06))
    bar.fill.solid(); bar.fill.fore_color.rgb = ACCBAR
    bar.line.fill.background()
    bar.name = "__accent_bar__"


def _tf(tf):
    """Normalise font family only; colours stay as set by each helper."""
    for p in tf.paragraphs:
        if p.font.name not in _KEEP:
            p.font.name = FONT
        if p.font.size is None:
            p.font.size = Pt(18)
        for r in p.runs:
            if r.font.name is None and p.font.name in _KEEP:
                continue
            if r.font.name not in _KEEP:
                r.font.name = FONT


def _title_style(sh):
    if sh is None:
        return
    _tf(sh.text_frame)
    for p in sh.text_frame.paragraphs:
        p.font.size = Pt(30); p.font.bold = True
        p.font.name = FONT_HDR; p.font.color.rgb = ACCENT


def _title(slide, text):
    if slide.shapes.title is not None:
        slide.shapes.title.text = text
        _title_style(slide.shapes.title)
        return
    bx = slide.shapes.add_textbox(Inches(LM), Inches(TT), Inches(BW), Inches(0.7))
    _no_line(bx)
    bx.text_frame.clear()
    r = bx.text_frame.paragraphs[0].add_run()
    r.text = text; r.font.size = Pt(30); r.font.bold = True
    r.font.name = FONT_HDR; r.font.color.rgb = ACCENT


def _bul(slide, items, *, l=LM, t=BT, w=BW, h=5.0, fs=18, sp=5):
    bx = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    _no_line(bx)
    tf = bx.text_frame; tf.word_wrap = True; tf.clear()
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = item; p.level = 0
        p.font.size = Pt(fs); p.font.name = FONT
        p.font.color.rgb = INK; p.space_after = Pt(sp)


def _code(slide, lines, *, l=LM, t=4.0, w=BW, h=1.4):
    bx = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE,
                                Inches(l), Inches(t), Inches(w), Inches(h))
    bx.fill.solid(); bx.fill.fore_color.rgb = CODE_C
    bx.line.color.rgb = BORDER; bx.line.width = Pt(1)
    tf = bx.text_frame; tf.clear(); tf.word_wrap = True
    tf.margin_left = Inches(0.25); tf.margin_right = Inches(0.15)
    tf.margin_top = Inches(0.10); tf.margin_bottom = Inches(0.10)
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run(); r.text = ln
        r.font.name = "Consolas"; r.font.size = Pt(14); r.font.color.rgb = INK


def _note(slide, text):
    bx = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE,
                                Inches(LM), Inches(NT), Inches(BW), Inches(NH))
    bx.fill.solid(); bx.fill.fore_color.rgb = NOTE_C
    bx.line.color.rgb = BORDER; bx.line.width = Pt(0.75)
    tf = bx.text_frame; tf.clear(); tf.word_wrap = True
    tf.margin_left = Inches(0.20); tf.margin_right = Inches(0.15)
    tf.margin_top = Inches(0.08); tf.margin_bottom = Inches(0.08)
    p = tf.paragraphs[0]; p.text = text
    p.font.size = Pt(13); p.font.name = FONT; p.font.color.rgb = MUTED
    p.alignment = PP_ALIGN.CENTER


def _section(prs, title, sub=None):
    sl = prs.slides.add_slide(prs.slide_layouts[5]); _bg(sl); _title(sl, title)
    if sub:
        bx = sl.shapes.add_textbox(Inches(LM), Inches(BT + 0.6), Inches(BW), Inches(1.6))
        _no_line(bx)
        bx.text_frame.clear()
        p = bx.text_frame.paragraphs[0]; p.text = sub
        p.font.size = Pt(20); p.font.name = FONT; p.font.color.rgb = MUTED; p.font.italic = True


def _cols(slide, lh, li, rh, ri, *, th=BT, fs=16):
    tb = th + 0.45
    for x, hdr in ((LM, lh), (C2, rh)):
        bx = slide.shapes.add_textbox(Inches(x), Inches(th), Inches(CW), Inches(0.4))
        _no_line(bx)
        bx.text_frame.text = hdr
        p = bx.text_frame.paragraphs[0]
        p.font.size = Pt(17); p.font.bold = True; p.font.name = FONT_HDR; p.font.color.rgb = ACCENT
    bh = NT - tb - 0.15
    _bul(slide, li, l=LM, t=tb, w=CW, h=bh, fs=fs)
    _bul(slide, ri, l=C2,  t=tb, w=CW, h=bh, fs=fs)


def _flow(slide, steps):
    n = len(steps); bw_ = 2.0; bh_ = 0.85; gap = 0.30
    total = n * bw_ + (n - 1) * gap
    sx = LM + (BW - total) / 2; y = BT + 0.1
    for i, step in enumerate(steps):
        bx = sx + i * (bw_ + gap)
        sh = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE,
                                    Inches(bx), Inches(y), Inches(bw_), Inches(bh_))
        sh.fill.solid(); sh.fill.fore_color.rgb = FLOW_C
        sh.line.color.rgb = BORDER; sh.line.width = Pt(1)
        tf = sh.text_frame; tf.clear(); tf.word_wrap = True
        p = tf.paragraphs[0]; p.text = step; p.alignment = PP_ALIGN.CENTER
        p.font.size = Pt(14); p.font.bold = True; p.font.name = FONT_HDR; p.font.color.rgb = ACCENT
        if i < n - 1:
            ar = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RIGHT_ARROW,
                                        Inches(bx + bw_), Inches(y + 0.25),
                                        Inches(gap), Inches(bh_ - 0.50))
            ar.fill.solid(); ar.fill.fore_color.rgb = ACCENT; ar.line.color.rgb = ACCENT


# ══════════════════════════════════════════════════════════════════════
# Deck builder
# ══════════════════════════════════════════════════════════════════════

def build_deck() -> Presentation:
    prs = Presentation()
    prs.slide_width  = Inches(SLIDE_W)
    prs.slide_height = Inches(SLIDE_H)

    # ── Cover ────────────────────────────────────────────────────────
    sl = prs.slides.add_slide(prs.slide_layouts[0]); _bg(sl)
    sl.shapes.title.text = "Belajar AI untuk Pemula"
    _title_style(sl.shapes.title)
    sub = sl.placeholders[1]
    sub.text = (
        "Modul latihan (Regression, Classification, NN, CNN)\n"
        "+ alur belajar + latihan + pembahasan + troubleshooting\n"
        f"Generated: {date.today().isoformat()}  |  Repo: Ai Learn"
    )
    for p in sub.text_frame.paragraphs:
        p.font.name = FONT; p.font.size = Pt(18); p.font.color.rgb = MUTED

    # ── Agenda ───────────────────────────────────────────────────────
    sl = prs.slides.add_slide(prs.slide_layouts[1]); _bg(sl)
    _title(sl, "Agenda")
    _bul(sl, [
        "0) Cara pakai repo + cara run yang benar",
        "1) Alur ML: Data \u2192 Train \u2192 Evaluate \u2192 Improve",
        "2) Konsep penting: fitur/target, split, overfitting",
        "3) Metrik regression: MAE / RMSE / R\u00b2 (cara baca)",
        "4) Metrik classification: accuracy / precision / recall / F1",
        "5) Modul 1\u20134 + latihan + pembahasan",
        "6) Troubleshooting + checklist project",
    ])

    # ── Isi Repo ─────────────────────────────────────────────────────
    sl = prs.slides.add_slide(prs.slide_layouts[1]); _bg(sl)
    _title(sl, "Isi Repo (Yang Akan Anda Pakai)")
    _cols(sl,
        "Program latihan", [
            "01_linear_regression.py  (Regression)",
            "02_classification.py  (Classification)",
            "03_neural_network.py  (MNIST NN)",
            "04_computer_vision.py  (CNN CIFAR-10)",
        ],
        "Dokumentasi", [
            "BUKU_PANDUAN_AI.pdf  (buku modul)",
            "AI_Learning_Modul.pptx  (slide ini)",
            "TUTORIAL.md  (cara run + tips)",
            "PANDUAN.md  (perilaku normal vs error)",
            "GLOSSARY.md  (istilah)",
        ],
    )
    _note(sl, "Prinsip: baca 1\u20132 halaman \u2192 jalankan 1 program \u2192 ubah 1 parameter \u2192 catat hasil.")

    # ── Flow ─────────────────────────────────────────────────────────
    sl = prs.slides.add_slide(prs.slide_layouts[5]); _bg(sl)
    _title(sl, "Alur Belajar (Flow)")
    _flow(sl, ["Setup", "Data", "Train", "Evaluate", "Improve", "Project"])
    _bul(sl, [
        "Kalau Anda bingung: kembali ke flow ini. Semua modul mengikuti pola yang sama.",
        "Target akhir: Anda bisa menjelaskan 2 hal \u2014 (1) hasil/metric, (2) mengapa hasilnya begitu.",
    ], t=3.4, h=2.5)

    # ── 0) Setup ─────────────────────────────────────────────────────
    _section(prs, "0) Setup", "Pastikan cara run benar (biar tidak error palsu)")

    sl = prs.slides.add_slide(prs.slide_layouts[1]); _bg(sl)
    _title(sl, "Cara Menjalankan (Windows PowerShell)")
    _bul(sl, [
        "Selalu jalankan Python dari .venv project (bukan Python global).",
        'Format perintah: & "./.venv/Scripts/python.exe" <nama_file.py>',
    ], h=2.0)
    _code(sl, [
        '& "./.venv/Scripts/python.exe" 01_linear_regression.py',
        '& "./.venv/Scripts/python.exe" 02_classification.py',
        '& "./.venv/Scripts/python.exe" 03_neural_network.py',
        '& "./.venv/Scripts/python.exe" 04_computer_vision.py',
    ], t=3.5, h=2.2)
    _note(sl, "Jika Anda pakai VS Code: pastikan interpreter VS Code juga menunjuk ke .venv.")

    sl = prs.slides.add_slide(prs.slide_layouts[1]); _bg(sl)
    _title(sl, "Matplotlib: Grafik Muncul, Lalu Apa?")
    _cols(sl,
        "Perilaku normal", [
            "Program berhenti sejenak di plt.show()",
            "Anda melihat grafik",
            "Tutup window grafik \u2192 program lanjut",
        ],
        "Yang sering disangka error", [
            "Tutup window lalu muncul KeyboardInterrupt",
            "Atau Anda tekan Ctrl+C",
            "Artinya program dihentikan manual",
        ],
    )
    _note(sl, "Rule of thumb: kalau hasil sudah keluar dan Anda yang menghentikan, itu bukan bug.")

    # ── 1) Fondasi ───────────────────────────────────────────────────
    _section(prs, "1) Fondasi", "Konsep wajib sebelum masuk ke model")

    sl = prs.slides.add_slide(prs.slide_layouts[1]); _bg(sl)
    _title(sl, "Fitur (X) dan Target (y)")
    _bul(sl, [
        "Fitur (X): data input yang dipakai model untuk membuat prediksi.",
        "Target (y): jawaban yang ingin dipelajari model.",
        "Regression: y berupa angka (mis. harga).",
        "Classification: y berupa kelas (mis. spam / tidak).",
        "Kesalahan umum: fitur bocor (data leakage) \u2014 memasukkan info masa depan ke fitur.",
    ])

    sl = prs.slides.add_slide(prs.slide_layouts[1]); _bg(sl)
    _title(sl, "Train / Validation / Test")
    _bul(sl, [
        "Train: data untuk belajar (fit).",
        "Validation: data untuk memilih setting (hyperparameter).",
        "Test: data untuk evaluasi final (simulasi dunia nyata).",
        "Kalau pemula: minimal train/test dulu, lalu tambah validation saat tuning.",
    ])
    _note(sl, "Prinsip: test hanya dipakai sekali untuk hasil final.")

    sl = prs.slides.add_slide(prs.slide_layouts[1]); _bg(sl)
    _title(sl, "Overfitting vs Underfitting")
    _cols(sl,
        "Underfitting", [
            "Model terlalu sederhana",
            "Train score rendah",
            "Test score rendah",
            "Solusi: model lebih kuat, fitur lebih informatif",
        ],
        "Overfitting", [
            "Model terlalu menghafal",
            "Train score tinggi",
            "Test score turun",
            "Solusi: regularisasi, data lebih banyak, early stopping",
        ],
    )

    # ── 2) Metrik ────────────────────────────────────────────────────
    _section(prs, "2) Metrik", "Cara membaca hasil agar tidak salah interpretasi")

    sl = prs.slides.add_slide(prs.slide_layouts[1]); _bg(sl)
    _title(sl, "Regression Metrics: MAE / RMSE / R\u00b2")
    _bul(sl, [
        "MAE: rata-rata selisih |aktual \u2212 prediksi| (lebih mudah dipahami).",
        "RMSE: seperti MAE tapi menghukum error besar (lebih sensitif outlier).",
        "R\u00b2: seberapa besar variasi y yang bisa dijelaskan model (0.0\u20131.0).",
        "Interpretasi cepat: R\u00b2 ~0.8+ bagus untuk data yang cukup linear.",
    ])

    sl = prs.slides.add_slide(prs.slide_layouts[1]); _bg(sl)
    _title(sl, "Cara Membaca R\u00b2 (Contoh)")
    _bul(sl, [
        "R\u00b2 = 0.90 \u2192 model menjelaskan 90% variasi harga (pola sangat tertangkap).",
        "R\u00b2 = 0.50 \u2192 masih banyak variasi yang belum terjelaskan (fitur kurang / noise tinggi).",
        "R\u00b2 < 0 \u2192 model lebih buruk dari prediksi rata-rata (indikasi masalah serius).",
    ], h=3.0)
    _note(sl, "Jangan lihat 1 metric saja: cek juga plot residual / pred vs actual jika tersedia.")

    sl = prs.slides.add_slide(prs.slide_layouts[1]); _bg(sl)
    _title(sl, "Classification Metrics")
    _bul(sl, [
        "Accuracy: proporsi prediksi yang benar (bagus jika kelas seimbang).",
        "Precision: dari yang diprediksi positif, berapa yang benar positif.",
        "Recall: dari yang sebenarnya positif, berapa yang berhasil ditangkap.",
        "F1: rata-rata harmonik precision & recall (bagus untuk data tidak seimbang).",
    ])

    sl = prs.slides.add_slide(prs.slide_layouts[1]); _bg(sl)
    _title(sl, "Confusion Matrix (Cara Baca)")
    _cols(sl,
        "Makna kotak", [
            "TP: benar positif",
            "TN: benar negatif",
            "FP: salah positif (false alarm)",
            "FN: salah negatif (miss)",
        ],
        "Kapan fokus apa?", [
            "Jika false alarm mahal \u2192 perhatikan precision",
            "Jika miss mahal \u2192 perhatikan recall",
            "Jika seimbang \u2192 lihat F1",
        ],
    )
    _note(sl, "Iris biasanya seimbang (accuracy cukup). Pada fraud/spam, F1 lebih aman.")

    # ── Modul 1: Linear Regression ───────────────────────────────────
    _section(prs, "Modul 1", "Linear Regression \u2014 prediksi angka")

    sl = prs.slides.add_slide(prs.slide_layouts[1]); _bg(sl)
    _title(sl, "Modul 1: Tujuan & Output")
    _bul(sl, [
        "Tujuan: memprediksi harga rumah dari ukuran.",
        "Output: rumus garis (slope & intercept), R\u00b2 / MAE / RMSE, dan plot.",
        "File: 01_linear_regression.py",
    ], h=3.0)
    _code(sl, ['& "./.venv/Scripts/python.exe" 01_linear_regression.py'], t=4.8, h=1.0)

    sl = prs.slides.add_slide(prs.slide_layouts[1]); _bg(sl)
    _title(sl, "Modul 1: Langkah di Kode (Flow)")
    _flow(sl, ["Buat data", "Plot", "Split", "Fit", "Predict", "Evaluate"])
    _bul(sl, [
        "Perhatikan: bentuk data X harus 2D untuk scikit-learn.",
        "Jangan lupa: evaluasi pakai data test, bukan training.",
    ], t=3.4, h=2.5)

    sl = prs.slides.add_slide(prs.slide_layouts[1]); _bg(sl)
    _title(sl, "Modul 1: Mengapa Jawabannya Seperti Itu?")
    _bul(sl, [
        "Linear regression mencari garis yang meminimalkan total error kuadrat (least squares).",
        "Slope = dampak perubahan ukuran terhadap harga (per 1 m\u00b2).",
        "Intercept = harga dasar ketika ukuran mendekati 0 (makna tergantung data).",
        "Jika noise besar \u2192 R\u00b2 turun karena pola linear tertutup variasi acak.",
    ])

    sl = prs.slides.add_slide(prs.slide_layouts[1]); _bg(sl)
    _title(sl, "Modul 1: Latihan")
    _bul(sl, [
        "Latihan A: Naikkan noise \u2192 apa yang terjadi pada RMSE dan R\u00b2?",
        "Latihan B: Tambah jumlah sampel \u2192 apakah hasil makin stabil?",
        "Latihan C: Buat outlier ekstrem \u2192 metric mana yang paling terpengaruh (MAE vs RMSE)?",
    ])
    _note(sl, "Pembahasan singkat: noise/outlier biasanya membuat RMSE naik lebih tajam daripada MAE.")

    sl = prs.slides.add_slide(prs.slide_layouts[1]); _bg(sl)
    _title(sl, "Modul 1: Pembahasan Latihan (Step-by-step)")
    _bul(sl, [
        "A) Noise naik: titik data makin menyebar \u2192 error rata-rata naik.",
        "   Dampak: RMSE naik, R\u00b2 turun (variasi y lebih banyak dari noise).",
        "B) Sample size naik: estimasi slope/intercept makin stabil.",
        "   Dampak: R\u00b2 biasanya lebih konsisten antar run.",
        "C) Outlier ekstrem: error kuadrat membuat RMSE sangat sensitif.",
        "   Dampak: RMSE melonjak, MAE naik tapi tidak setajam RMSE.",
    ], fs=16)

    sl = prs.slides.add_slide(prs.slide_layouts[1]); _bg(sl)
    _title(sl, "Modul 1: Contoh Kasus Nyata (Harga Rumah)")
    _cols(sl,
        "Fitur (X)", [
            "Luas bangunan, luas tanah",
            "Lokasi (jarak ke pusat kota)",
            "Jumlah kamar, usia bangunan",
            "Akses jalan / transportasi",
        ],
        "Target (y) + Evaluasi", [
            "Target: harga jual (angka)",
            "Metric: MAE / RMSE (rupiah) + R\u00b2",
            "Waspada: data leakage (harga appraisal)",
            "Output: interval harga & fitur penting",
        ],
    )
    _note(sl, "Checklist: baseline linear \u2192 tambah fitur \u2192 bandingkan MAE di test.")

    # ── Modul 2: Classification ──────────────────────────────────────
    _section(prs, "Modul 2", "Classification \u2014 Decision Tree")

    sl = prs.slides.add_slide(prs.slide_layouts[1]); _bg(sl)
    _title(sl, "Modul 2: Tujuan & Output")
    _bul(sl, [
        "Tujuan: klasifikasi bunga iris ke 3 spesies.",
        "Output: accuracy + classification report + confusion matrix + visualisasi tree.",
        "File: 02_classification.py",
    ], h=3.0)
    _code(sl, ['& "./.venv/Scripts/python.exe" 02_classification.py'], t=4.8, h=1.0)

    sl = prs.slides.add_slide(prs.slide_layouts[1]); _bg(sl)
    _title(sl, "Modul 2: Mengapa Decision Tree Bisa Benar?")
    _bul(sl, [
        "Tree membuat aturan IF-THEN yang memecah data agar tiap bagian makin 'murni'.",
        "Split dipilih yang paling meningkatkan pemisahan kelas (gini / information gain).",
        "Iris: petal length/width sering jadi fitur paling penting karena separasi jelas.",
    ])

    sl = prs.slides.add_slide(prs.slide_layouts[1]); _bg(sl)
    _title(sl, "Modul 2: Cara Baca Confusion Matrix")
    _bul(sl, [
        "Diagonal besar = banyak prediksi benar.",
        "Jika versicolor sering diprediksi virginica: fitur overlap di area tersebut.",
        "Fokus improvement: cek fitur tambahan / model ensemble (next level).",
    ])

    sl = prs.slides.add_slide(prs.slide_layouts[1]); _bg(sl)
    _title(sl, "Modul 2: Latihan")
    _bul(sl, [
        "Latihan A: Ubah max_depth (1, 2, 3, 5, None). Catat train vs test accuracy.",
        "Latihan B: Cari fitur paling penting. Apakah selalu sama?",
        "Latihan C: Coba cross-validation dan bandingkan dengan 1 kali split.",
    ])
    _note(sl, "Pembahasan singkat: depth besar \u2192 train naik, test bisa turun (overfitting).")

    sl = prs.slides.add_slide(prs.slide_layouts[1]); _bg(sl)
    _title(sl, "Modul 2: Pembahasan Latihan (Step-by-step)")
    _bul(sl, [
        "A) max_depth kecil: aturan sedikit \u2192 underfitting (train & test rendah).",
        "   max_depth besar: train tinggi, test bisa turun (overfitting).",
        "B) Fitur penting bisa berubah jika data split berubah sedikit (tree sensitif).",
        "   Solusi: lihat rata-rata feature importance dari beberapa run.",
        "C) Cross-validation memberi estimasi performa lebih stabil dibanding 1 kali split.",
    ], fs=16)

    sl = prs.slides.add_slide(prs.slide_layouts[1]); _bg(sl)
    _title(sl, "Modul 2: Contoh Kasus Nyata (Spam / Fraud)")
    _cols(sl,
        "Yang sering terjadi", [
            "Data tidak seimbang (fraud sangat sedikit)",
            "Accuracy bisa menipu (99% 'normal')",
            "Biaya salah beda: FN lebih mahal",
        ],
        "Solusi evaluasi", [
            "Fokus: precision / recall / F1",
            "Confusion matrix: lihat FP vs FN",
            "Pertimbangkan threshold tuning",
            "Stratified split + cross-validation",
        ],
    )
    _note(sl, "Rule: pilih metric sesuai biaya bisnis, bukan sekadar accuracy.")

    # ── Modul 3: Neural Network ──────────────────────────────────────
    _section(prs, "Modul 3", "Neural Network \u2014 MNIST")

    sl = prs.slides.add_slide(prs.slide_layouts[1]); _bg(sl)
    _title(sl, "Modul 3: Tujuan & Output")
    _bul(sl, [
        "Tujuan: mengenali digit 0\u20139 dari gambar.",
        "Output: loss/accuracy per epoch + evaluasi test + prediksi contoh.",
        "File: 03_neural_network.py",
    ], h=3.0)
    _code(sl, ['& "./.venv/Scripts/python.exe" 03_neural_network.py'], t=4.8, h=1.0)

    sl = prs.slides.add_slide(prs.slide_layouts[1]); _bg(sl)
    _title(sl, "Modul 3: Komponen NN (Yang Harus Dihafal)")
    _cols(sl,
        "Komponen", [
            "Layer: Dense / Conv",
            "Activation: ReLU / Softmax",
            "Loss: cross-entropy",
            "Optimizer: Adam",
        ],
        "Makna", [
            "Layer = transformasi fitur",
            "Activation = non-linearitas",
            "Loss = ukuran salah",
            "Optimizer = cara update weights",
        ],
    )

    sl = prs.slides.add_slide(prs.slide_layouts[1]); _bg(sl)
    _title(sl, "Modul 3: Mengapa Prediksi Bisa Benar?")
    _bul(sl, [
        "Output softmax = probabilitas 10 kelas.",
        "Backpropagation menghitung gradien: bagaimana weights harus berubah agar loss turun.",
        "Gradient descent / Adam melakukan update kecil berulang sampai model membaik.",
        "Overfitting muncul jika model terlalu kompleks / epoch terlalu banyak.",
    ])

    sl = prs.slides.add_slide(prs.slide_layouts[1]); _bg(sl)
    _title(sl, "Modul 3: Latihan")
    _bul(sl, [
        "Latihan A: Ubah epochs (1, 3, 10). Catat train/test accuracy.",
        "Latihan B: Tambah Dropout. Apa dampaknya pada gap train vs val?",
        "Latihan C: Digit mana yang sering salah (mis. 4 vs 9)? Kenapa?",
    ])

    sl = prs.slides.add_slide(prs.slide_layouts[1]); _bg(sl)
    _title(sl, "Modul 3: Pembahasan Latihan (Step-by-step)")
    _bul(sl, [
        "A) Epoch naik: train accuracy naik; kalau val tidak naik \u2192 overfitting.",
        "B) Dropout: train accuracy turun sedikit, tapi val/test bisa lebih stabil.",
        "C) Digit mirip (4 vs 9): gaya tulisan bervariasi; noise bikin overlap.",
        "   Cara cek: tampilkan contoh yang salah \u2192 amati apakah memang ambigu.",
    ], fs=16)

    sl = prs.slides.add_slide(prs.slide_layouts[1]); _bg(sl)
    _title(sl, "Modul 3: Contoh Kasus Nyata (OCR / Dokumen)")
    _cols(sl,
        "Masalah dunia nyata", [
            "Foto blur, miring, cahaya buruk",
            "Font beragam, tulisan tangan",
            "Background mengganggu",
        ],
        "Strategi", [
            "Preprocessing (crop, threshold, denoise)",
            "Data augmentation (rotasi, noise)",
            "Model CNN / CRNN untuk teks panjang",
            "Evaluasi: word error rate",
        ],
    )
    _note(sl, "MNIST adalah latihan dasar. OCR nyata butuh data & preprocessing lebih serius.")

    # ── Modul 4: CNN / Computer Vision ───────────────────────────────
    _section(prs, "Modul 4", "Computer Vision \u2014 CNN CIFAR-10")

    sl = prs.slides.add_slide(prs.slide_layouts[1]); _bg(sl)
    _title(sl, "Modul 4: Tujuan & Output")
    _bul(sl, [
        "Tujuan: klasifikasi gambar warna 10 kelas.",
        "Output: loss/accuracy + evaluasi + contoh prediksi.",
        "File: 04_computer_vision.py",
    ], h=3.0)
    _code(sl, ['& "./.venv/Scripts/python.exe" 04_computer_vision.py'], t=4.8, h=1.0)

    sl = prs.slides.add_slide(prs.slide_layouts[1]); _bg(sl)
    _title(sl, "Modul 4: Mengapa CNN Lebih Cocok untuk Gambar?")
    _bul(sl, [
        "Convolution menangkap pola lokal (edge/texture) tanpa 'melihat' seluruh gambar.",
        "Pooling mereduksi ukuran dan membuat model tahan terhadap pergeseran kecil.",
        "Layer awal belajar fitur sederhana; layer akhir belajar konsep objek.",
    ])

    sl = prs.slides.add_slide(prs.slide_layouts[1]); _bg(sl)
    _title(sl, "Modul 4: Latihan")
    _bul(sl, [
        "Latihan A: Tambah filter (16\u219232) dan lihat akurasi + waktu training.",
        "Latihan B: Tambah 1 layer conv. Akurasi naik atau overfitting?",
        "Latihan C: Kurangi epoch untuk eksperimen cepat. Trade-off waktu vs performa?",
    ])

    sl = prs.slides.add_slide(prs.slide_layouts[1]); _bg(sl)
    _title(sl, "Modul 4: Pembahasan Latihan (Step-by-step)")
    _bul(sl, [
        "A) Filter naik: kapasitas naik \u2192 akurasi bisa naik, tapi training lebih lama.",
        "B) Layer bertambah: pola lebih kompleks, tapi risiko overfitting naik.",
        "C) Epoch turun: cepat untuk eksperimen, tapi mungkin belum konvergen.",
        "   Praktik: baseline cepat dulu \u2192 training lama saat arsitektur oke.",
    ], fs=16)

    sl = prs.slides.add_slide(prs.slide_layouts[1]); _bg(sl)
    _title(sl, "Modul 4: Kasus Nyata (Quality Check / Klasifikasi Produk)")
    _cols(sl,
        "Contoh", [
            "Deteksi cacat produk (retak/lecet)",
            "Klasifikasi jenis barang di gudang",
            "Klasifikasi makanan (fresh / tidak)",
        ],
        "Catatan penting", [
            "Data harus representatif (cahaya, sudut)",
            "Evaluasi per kelas (minoritas sering gagal)",
            "Waspada: domain shift",
            "Solusi: transfer learning + monitoring",
        ],
    )

    # ── Troubleshooting ──────────────────────────────────────────────
    _section(prs, "Troubleshooting", "Jika ada error, cek ini dulu (urut)!")

    sl = prs.slides.add_slide(prs.slide_layouts[1]); _bg(sl)
    _title(sl, "1) Pastikan Interpreter Benar")
    _bul(sl, [
        "Gejala: ModuleNotFoundError / versi package tidak sesuai.",
        "Solusi: jalankan dengan .venv dan cek sys.executable.",
    ], h=2.0)
    _code(sl, ['& "./.venv/Scripts/python.exe" -c "import sys; print(sys.executable)"'], t=3.8, h=1.0)

    sl = prs.slides.add_slide(prs.slide_layouts[1]); _bg(sl)
    _title(sl, "2) Install Dependency")
    _bul(sl, [
        "Jika pip install gagal: update pip dulu.",
        "Install paket satu per satu untuk menemukan penyebab.",
    ], h=2.0)
    _code(sl, [
        '& "./.venv/Scripts/python.exe" -m pip install --upgrade pip',
        '& "./.venv/Scripts/python.exe" -m pip install -r requirements.txt',
    ], t=3.8, h=1.4)

    sl = prs.slides.add_slide(prs.slide_layouts[1]); _bg(sl)
    _title(sl, "3) Program Terasa Lama")
    _bul(sl, [
        "Kemungkinan: download dataset (MNIST/CIFAR) pertama kali.",
        "Kemungkinan: training berjalan di CPU (wajar lebih lambat).",
        "Solusi cepat: kurangi epoch dulu untuk eksperimen.",
        "Solusi: pastikan storage cukup dan internet stabil.",
    ])

    # ── Tips & Best Practices ────────────────────────────────────────
    sl = prs.slides.add_slide(prs.slide_layouts[1]); _bg(sl)
    _title(sl, "Cara Mencatat Eksperimen (Wajib)")
    _bul(sl, [
        "Buat tabel sederhana: (Tanggal, Model, Setting, Metric, Catatan).",
        "Ubah 1 hal saja per eksperimen (epoch, depth, filter, dll.).",
        "Simpan hasil: screenshot plot / log training / confusion matrix.",
        "Tujuan: menjawab 'perubahan apa yang membuat model membaik?'.",
    ])

    sl = prs.slides.add_slide(prs.slide_layouts[1]); _bg(sl)
    _title(sl, "Checklist Evaluasi (Agar Tidak Salah Klaim)")
    _bul(sl, [
        "Apakah test set tidak pernah dipakai saat tuning?",
        "Apakah metric sesuai problem (dan biaya bisnis)?",
        "Apakah ada data leakage?",
        "Apakah Anda cek error cases (contoh yang salah)?",
        "Apakah performa stabil (cross-validation / beberapa seed)?",
    ])

    sl = prs.slides.add_slide(prs.slide_layouts[1]); _bg(sl)
    _title(sl, "Next Step: Deploy Sederhana (Opsional)")
    _cols(sl,
        "Pilihan mudah", [
            "Streamlit: UI cepat untuk demo",
            "FastAPI: API endpoint untuk model",
            "Notebook: laporan eksperimen",
        ],
        "Yang perlu dipikirkan", [
            "Simpan model (joblib / keras .h5)",
            "Input validation (hindari input aneh)",
            "Monitoring: data baru beda dari training",
        ],
    )
    _note(sl, "Saya bisa bantu buat 1 demo Streamlit untuk Modul 1 atau 2.")

    sl = prs.slides.add_slide(prs.slide_layouts[1]); _bg(sl)
    _title(sl, "Portofolio (Agar Terlihat Progress)")
    _bul(sl, [
        "Buat 2\u20133 repo kecil: 1 regression, 1 classification, 1 vision.",
        "Setiap repo wajib ada: README, cara run, metric, contoh output.",
        "Tulis 'mengapa' Anda memilih metric/model tersebut.",
        "Ini yang membedakan 'sekadar run code' vs 'paham alur AI'.",
    ])

    # ── Next Level ───────────────────────────────────────────────────
    _section(prs, "Next Level", "Kalau sudah selesai modul, buat 1 project nyata")

    sl = prs.slides.add_slide(prs.slide_layouts[1]); _bg(sl)
    _title(sl, "Template Project (Wajib Ada)")
    _bul(sl, [
        "1) Problem statement (apa yang diprediksi, untuk siapa)",
        "2) Data: sumber, jumlah, fitur/target, pembersihan",
        "3) Baseline sederhana (model paling simple dulu)",
        "4) Evaluasi: metric yang tepat + interpretasi",
        "5) Improvement: 2\u20133 eksperimen dan catat hasil",
        "6) Kesimpulan: kapan model bisa dipakai, kapan tidak",
    ])

    sl = prs.slides.add_slide(prs.slide_layouts[5]); _bg(sl)
    _title(sl, "Roadmap Latihan 2 Minggu")
    _flow(sl, ["Hari 1-2", "Hari 3-4", "Hari 5-7", "Minggu 2", "Project"])
    _bul(sl, [
        "Hari 1\u20132: Regression + R\u00b2 + visualisasi",
        "Hari 3\u20134: Classification + confusion matrix + depth tuning",
        "Hari 5\u20137: NN + epoch/batch + overfitting",
        "Minggu 2: CNN + eksperimen arsitektur sederhana",
        "Project: 1 kasus nyata + README hasil",
    ], t=3.4, h=3.0)

    # ── Closing ──────────────────────────────────────────────────────
    sl = prs.slides.add_slide(prs.slide_layouts[1]); _bg(sl)
    _title(sl, "Penutup")
    _bul(sl, [
        "Skill inti AI developer pemula: mengukur hasil (metric) dan menjelaskan 'mengapa'.",
        "Saran: ulangi tiap modul 2x sambil ubah parameter dan catat.",
        "Referensi: BUKU_PANDUAN_AI.pdf (buku) untuk penjelasan lebih panjang.",
    ])

    # ── Final pass: cleanup + accent bar + font normalisation ────────
    for s in prs.slides:
        _bg(s)
        # Remove empty content placeholders (avoid dashed-border artefacts)
        for ph in list(s.placeholders):
            if ph.placeholder_format.idx > 0 and not ph.text.strip():
                ph._element.getparent().remove(ph._element)
        _accent_bar(s)
        for sh in s.shapes:
            if getattr(sh, 'name', '') == '__accent_bar__':
                continue
            if getattr(sh, "has_text_frame", False) and sh.has_text_frame:
                _tf(sh.text_frame)

    return prs


def main() -> None:
    prs = build_deck()
    prs.save(OUT)
    print(f"\u2705 PPTX created: {OUT}")


if __name__ == "__main__":
    main()
