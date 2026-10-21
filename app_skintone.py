import io
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

# ----------------------------------------------------------------------------- konstanta
KATEGORI = ["Cool", "Neutral", "Warm"]
COLORS = {"Cool": "#3d5a80", "Neutral": "#8d8d8d", "Warm": "#e07a5f"}
PETA_LEVEL = {2.5: 2.0,     # 2.5 digabung ke level 2 (semua undertone)
              3.5: 3.0}     # 3.5 -> 3
TOLERANSI = 1.0             # akurasi "dekat" bila selisih <= TOLERANSI level
LANGKAH_THETA = 0.5
FILE_LATIH_DEFAULT = "ColorMapping.xlsx"   # dipakai bila user tidak upload data latih

KOLOM_ALIAS = {
    "L": ["L", "L*", "l", "l*"],
    "a*": ["a*", "a", "A*"],
    "b*": ["b*", "b", "B*"],
    "label": ["Undertone - Skintone", "Undertone_Skintone", "Undertone-Skintone"],
    "level": ["Overtone Level", "Overtone_Level", "Level"],
    "id": ["Kode Subjek", "No", "ID", "Subject"],
}


# ============================================================================ 1. data
def split_undertone_skintone(df, column="Undertone - Skintone",
                             new_undertone_col="Undertone", new_skintone_col="Skintone"):
    def parse_one(text):
        if pd.isna(text):
            return pd.Series([None, None])
        text = str(text).strip()
        match = re.match(r"^(.*?)\s*[-\u2013]\s*([\d.,]+)\s*$", text)
        if not match:
            return pd.Series([text, None])
        name_part, num_part = match.group(1).strip(), match.group(2).strip().replace(",", ".")
        try:
            level = float(num_part)
        except ValueError:
            level = None
        return pd.Series([name_part, level])
    df = df.copy()
    df[[new_undertone_col, new_skintone_col]] = df[column].apply(parse_one)
    return df


def tebak_kolom(kolom, jenis):
    for k in KOLOM_ALIAS[jenis]:
        if k in kolom:
            return k
    return None


def siapkan_data_latih(df_raw, kol_L, kol_a, kol_b, kol_label, kol_level):
    df = df_raw.rename(columns={kol_L: "L", kol_a: "a*", kol_b: "b*"}).copy()
    df = df.loc[:, ~df.columns.duplicated()]
    df = split_undertone_skintone(df, column=kol_label)
    df = df.dropna(subset=[kol_label]).copy()
    df["Undertone"] = df["Undertone"].str.strip().str.title()
    df_cnw = df[df["Undertone"].isin(KATEGORI)].copy()
    df_cnw["Overtone Level"] = pd.to_numeric(df_cnw[kol_level], errors="coerce")
    df_cnw = df_cnw.dropna(subset=["L", "a*", "b*"])
    return df_cnw


# ============================================================================ 2. undertone
def hitung_hue(a, b):
    """Hue dalam derajat (0–360) dari a* dan b*."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.mod(np.degrees(np.arctan2(b, a)), 360)


def cari_batas(x, y):
    """Decision tree 1 fitur: jumlah daun = jumlah kelas -> batas-batas nilai x."""
    tree = DecisionTreeClassifier(max_leaf_nodes=len(np.unique(y)), random_state=42)
    tree.fit(x.reshape(-1, 1), y)
    batas = sorted(float(t) for t in tree.tree_.threshold if t != -2)
    return tree, batas, accuracy_score(y, tree.predict(x.reshape(-1, 1)))


def latih_hue(a, b, y):
    hue = hitung_hue(a, b)
    tree, batas, acc = cari_batas(hue, np.asarray(y))
    titik = np.array([batas[0] - 10, np.mean(batas), batas[-1] + 10]).reshape(-1, 1)
    return {"metode": "Hue", "tree": tree, "batas": batas, "acc": acc,
            "urutan": list(tree.predict(titik))}


def latih_linear(a, b, y, langkah=LANGKAH_THETA):
    a, b, y = np.asarray(a, float), np.asarray(b, float), np.asarray(y)
    terbaik = {"acc": -1}
    riwayat = []
    for theta in np.arange(0, 360, langkah):
        skor = np.cos(np.radians(theta)) * a + np.sin(np.radians(theta)) * b
        tree, batas, acc = cari_batas(skor, y)
        riwayat.append((theta, acc))
        if acc > terbaik["acc"]:
            terbaik = {"theta": theta, "batas": batas, "acc": acc, "tree": tree}
    terbaik["riwayat"] = pd.DataFrame(riwayat, columns=["theta", "acc"])
    terbaik["metode"] = "Linear"
    terbaik["c"] = float(np.cos(np.radians(terbaik["theta"])))
    terbaik["s"] = float(np.sin(np.radians(terbaik["theta"])))
    terbaik["urutan"] = list(terbaik["tree"].predict(
        np.array([-1e9, np.mean(terbaik["batas"]), 1e9]).reshape(-1, 1)))
    return terbaik


def skor_garis(a, b, garis):
    """Hue (derajat) untuk metode Hue, cos(θ)·a* + sin(θ)·b* untuk metode Linear."""
    if garis["metode"] == "Hue":
        return hitung_hue(a, b)
    return garis["c"] * np.asarray(a, float) + garis["s"] * np.asarray(b, float)


def prediksi_undertone(a, b, garis):
    skor = np.asarray(skor_garis(a, b, garis), dtype=float).reshape(-1, 1)
    return garis["tree"].predict(skor)


# ============================================================================ 3. level
def gabung(level):
    return np.array([PETA_LEVEL.get(float(lv), float(lv)) for lv in level])


def fit_median(L, y):
    levels = np.sort(np.unique(y))
    med = np.array([np.median(L[y == lv]) for lv in levels])
    batas = (med[:-1] + med[1:]) / 2
    urut = bool(np.all(np.diff(med) < 0))
    return levels, med, batas, urut


def predict_median(L, levels, batas):
    return levels[(np.asarray(L, float)[:, None] < batas[None, :]).sum(axis=1)]


def latih_level(df_cnw):
    df_lv = df_cnw.dropna(subset=["L", "Overtone Level"]).copy()
    df_lv["Level asli"] = df_lv["Overtone Level"].astype(float)
    df_lv["Level"] = gabung(df_lv["Level asli"])
    hasil = {}
    for ut in KATEGORI:
        sub = df_lv[df_lv["Undertone"] == ut]
        if len(sub) == 0:
            continue
        L, y = sub["L"].values.astype(float), sub["Level"].values
        levels, med, batas, urut = fit_median(L, y)
        y_pred = predict_median(L, levels, batas)
        hasil[ut] = {"levels": levels, "med": med, "batas": batas, "urut": urut,
                     "tepat": (y_pred == y).mean() * 100,
                     "dekat": (np.abs(y_pred - y) <= TOLERANSI).mean() * 100,
                     "n": len(y), "rentang_L": (L.min(), L.max()),
                     "akurasi_level": {lv: (y_pred[y == lv] == lv).mean() * 100 for lv in levels}}
    return df_lv, hasil


def prediksi_level(L, undertone, hasil):
    L = np.asarray(L, float)
    undertone = np.asarray(undertone)
    out = np.full(len(L), np.nan)
    for ut in KATEGORI:
        m = undertone == ut
        if m.any() and ut in hasil:
            out[m] = predict_median(L[m], hasil[ut]["levels"], hasil[ut]["batas"])
    return out


def tabel_aturan_level(hasil):
    baris = []
    for ut in KATEGORI:
        if ut not in hasil:
            continue
        lv, bt = hasil[ut]["levels"], hasil[ut]["batas"]
        tepi = np.r_[np.inf, bt, -np.inf]
        for k, l in enumerate(lv):
            baris.append({"Undertone": ut, "Level": l,
                          "L lebih dari": None if np.isinf(tepi[k + 1]) else round(float(tepi[k + 1]), 2),
                          "L sampai dengan": None if np.isinf(tepi[k]) else round(float(tepi[k]), 2),
                          "Median L latih": round(float(hasil[ut]["med"][k]), 2)})
    return pd.DataFrame(baris)


# ============================================================================ 4. plot
def gambar_batas_hue(ax, garis, r_max, label=True):
    """Gambar batas Hue sebagai garis dari titik (0, 0) di bidang a*-b*."""
    for i, bt in enumerate(garis["batas"]):
        ax.plot([0, r_max * np.cos(np.radians(bt))], [0, r_max * np.sin(np.radians(bt))],
                color="black", ls="--", lw=2, label=f"Batas {i + 1}: Hue {bt:.1f}°" if label else None)


def gambar_batas_linear(ax, garis, a_max, label=True):
    c, s = garis["c"], garis["s"]
    a_vals = np.linspace(0, a_max, 200)
    for i, bt in enumerate(garis["batas"]):
        if np.isfinite(bt) and abs(s) > 1e-9:
            ax.plot(a_vals, (bt - c * a_vals) / s, "k--", lw=2,
                    label=f"Batas {i + 1}: {c:.2f}a* + {s:.2f}b* = {bt:.2f}" if label else None)


def gambar_batas(ax, garis, r_max, a_max):
    if garis["metode"] == "Hue":
        gambar_batas_hue(ax, garis, r_max)
    else:
        gambar_batas_linear(ax, garis, a_max)


def judul_batas(garis):
    if garis["metode"] == "Hue":
        return f"Batas Hue {[round(b, 1) for b in garis['batas']]}° (akurasi {garis['acc'] * 100:.1f}%)"
    return f"Garis terbaik θ = {garis['theta']}° (akurasi {garis['acc'] * 100:.1f}%)"


def plot_ab_latih(df_cnw, garis):
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.5), gridspec_kw={"width_ratios": [1, 1.15]})

    ax = axes[0]
    if garis["metode"] == "Hue":
        rng = np.random.default_rng(0)
        for i, kat in enumerate(KATEGORI):
            sub = df_cnw[df_cnw["Undertone"] == kat]
            ax.scatter(sub["Hue"], i + rng.uniform(-0.15, 0.15, len(sub)), c=COLORS[kat],
                       edgecolors="white", s=80, alpha=0.85, zorder=3)
        for bt in garis["batas"]:
            ax.axvline(bt, color="black", ls="--", lw=2)
            ax.text(bt + 0.3, len(KATEGORI) - 0.45, f"{bt:.1f}°", ha="left", fontsize=10, fontweight="bold")
        ax.set_yticks(range(len(KATEGORI))); ax.set_yticklabels(KATEGORI)
        ax.set_ylim(-0.6, len(KATEGORI) - 0.3)
        ax.set_xlabel("Hue (derajat)"); ax.set_title("Sebaran Hue per undertone + batas")
        ax.grid(True, axis="x", ls=":", alpha=0.6)
    else:
        riw = garis["riwayat"]
        ax.plot(riw["theta"], riw["acc"] * 100, color="black")
        ax.axvline(garis["theta"], color="red", ls="--", label=f"θ terbaik = {garis['theta']}°")
        ax.set_xlabel("θ (derajat)"); ax.set_ylabel("Akurasi (%)"); ax.set_title("Akurasi vs sudut θ")
        ax.grid(True, ls=":", alpha=0.6); ax.legend()

    ax = axes[1]
    for kat in KATEGORI:
        sub = df_cnw[df_cnw["Undertone"] == kat]
        ax.scatter(sub["a*"], sub["b*"], c=COLORS[kat], label=kat, edgecolors="white", s=80, alpha=0.85)
    a_max = df_cnw["a*"].max() * 1.1
    gambar_batas(ax, garis, np.hypot(df_cnw["a*"], df_cnw["b*"]).max() * 2, a_max)
    ax.set_xlim(0, a_max); ax.set_ylim(0, df_cnw["b*"].max() * 1.1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Nilai a*"); ax.set_ylabel("Nilai b*")
    ax.set_title(judul_batas(garis))
    ax.grid(True, ls=":", alpha=0.6); ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    return fig


def plot_ab_prediksi(df_new, df_cnw, garis):
    n = len(df_new)
    fig, ax = plt.subplots(figsize=(11, 9))
    for kat in KATEGORI:
        sub = df_new[df_new["Undertone"] == kat]
        ax.scatter(sub["a*"], sub["b*"], c=COLORS[kat], s=14, alpha=0.45,
                   label=f"Data baru → {kat} ({len(sub)}, {len(sub) / max(n, 1) * 100:.0f}%)")
    for kat in KATEGORI:
        sub = df_cnw[df_cnw["Undertone"] == kat]
        ax.scatter(sub["a*"], sub["b*"], c=COLORS[kat], s=70, edgecolors="black", label=f"Data latih {kat}")
    a_max = max(df_new["a*"].quantile(0.999), df_cnw["a*"].max()) + 1
    b_max = max(df_new["b*"].quantile(0.999), df_cnw["b*"].max()) + 1
    r_max = max(np.hypot(df_new["a*"], df_new["b*"]).max(), np.hypot(df_cnw["a*"], df_cnw["b*"]).max()) * 1.1
    gambar_batas(ax, garis, r_max, a_max)
    ax.set_xlim(0, a_max); ax.set_ylim(0, b_max)
    ax.set_aspect("equal", adjustable="box"); ax.grid(True, ls=":", alpha=0.6)
    ax.set_xlabel("a*"); ax.set_ylabel("b*")
    metode = "batas Hue" if garis["metode"] == "Hue" else f"garis θ = {garis['theta']}°"
    ax.set_title(f"Data baru dipetakan dengan {metode} dari data latih")
    ax.legend(markerscale=1.5, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    return fig


def plot_level(df_, hasil, kolom_level, judul):
    """Gaya sama dengan plot level per undertone di notebook (median, rentang merah, batas putus-putus)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, ut in zip(axes, KATEGORI):
        if ut not in hasil:
            ax.set_title(f"{ut} - tidak ada model", fontweight="bold"); continue
        h = hasil[ut]
        levels, med, batas = h["levels"], h["med"], h["batas"]
        sub = df_[df_["Undertone"] == ut].dropna(subset=["L", kolom_level])
        L, y = sub["L"].values.astype(float), sub[kolom_level].values.astype(float)
        besar = len(sub) > 150
        pal = plt.get_cmap("tab10", len(levels))
        for j, lv in enumerate(levels):
            m = y == lv
            ax.scatter(L[m], [lv] * m.sum(), color=pal(j), edgecolors="white",
                       s=40 if besar else 70, alpha=0.6 if besar else 1, zorder=3)
            ax.plot(med[j], lv, marker="|", color="black", ms=18, mew=2, zorder=4)   # median latih
        L_ref = np.r_[L, h["rentang_L"]]
        tepi_plot = np.r_[L_ref.max() + 1, batas, L_ref.min() - 1]
        for k, lv in enumerate(levels):
            ax.hlines(lv, tepi_plot[k + 1], tepi_plot[k], color="crimson", lw=2.5, alpha=0.5, zorder=2)
        for bt in batas:
            ax.axvline(bt, color="black", ls="--", lw=1.2)
        ax.set_yticks(levels)
        ax.set_title(judul(ut, sub, h), fontweight="bold")
        ax.set_xlabel("Nilai L"); ax.grid(True, ls=":", alpha=0.6)
    semua_lv = sorted({float(l) for h in hasil.values() for l in h["levels"]})
    axes[0].set_yticks(semua_lv)          # sharey: tampilkan semua level dari ketiga undertone
    axes[0].set_ylabel("Level")
    plt.tight_layout()
    return fig


# ============================================================================ 5. excel
def tabel_model(garis, hasil, n_latih):
    if garis["metode"] == "Hue":
        par = [("metode undertone", "Decision Tree pada Hue = atan2(b*, a*)"),
               ("urutan kelas (Hue naik)", " < ".join(garis["urutan"])),
               ("batas Hue (derajat)", str([round(x, 3) for x in garis["batas"]]))]
    else:
        par = [("metode undertone", "Decision Tree pada skor = cos(θ)·a* + sin(θ)·b*"),
               ("theta (derajat)", garis["theta"]),
               ("skor", f"{garis['c']:.4f}*a* + {garis['s']:.4f}*b*"),
               ("urutan kelas (skor naik)", " < ".join(garis["urutan"])),
               ("batas skor", str([round(x, 3) for x in garis["batas"]]))]
    par += [("akurasi undertone (train)", round(garis["acc"], 4)), ("n data latih C/N/W", n_latih),
            ("peta level", str(PETA_LEVEL)), ("metode level", "median L per undertone")]
    for ut in KATEGORI:
        if ut in hasil:
            par += [(f"batas L level {ut}", str([round(float(b), 2) for b in sorted(hasil[ut]["batas"])])),
                    (f"rentang L latih {ut}", f"{hasil[ut]['rentang_L'][0]:.2f} – {hasil[ut]['rentang_L'][1]:.2f}"),
                    (f"akurasi level {ut} (train)", f"{hasil[ut]['tepat']:.2f}%")]
    df = pd.DataFrame(par, columns=["Parameter", "Nilai"])
    df["Nilai"] = df["Nilai"].astype(str)
    return df


def buat_excel(df_pred, info_model, aturan_level, ringkasan, tabel_ut_lv, evaluasi=None):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df_pred.round(4).to_excel(w, sheet_name="Prediksi", index=False)
        ringkasan.to_excel(w, sheet_name="Ringkasan")
        tabel_ut_lv.to_excel(w, sheet_name="Undertone_x_Level")
        info_model.to_excel(w, sheet_name="Model", index=False)
        aturan_level.to_excel(w, sheet_name="Aturan_Level", index=False)
        if evaluasi is not None:
            evaluasi.to_excel(w, sheet_name="Evaluasi")
    return buf.getvalue()


# ============================================================================ 6. cache
@st.cache_data(show_spinner=False)
def baca_excel(data, sheet):
    return pd.read_excel(io.BytesIO(data), sheet_name=sheet)


@st.cache_data(show_spinner=False)
def daftar_sheet(data):
    return pd.ExcelFile(io.BytesIO(data)).sheet_names


@st.cache_resource(show_spinner="Melatih model...")
def latih(df_cnw_json, metode):
    df_cnw = pd.read_json(io.StringIO(df_cnw_json), orient="split")
    a, b, y = df_cnw["a*"].values, df_cnw["b*"].values, df_cnw["Undertone"].values
    garis = latih_hue(a, b, y) if metode == "Hue" else latih_linear(a, b, y)
    df_cnw["Hue"] = hitung_hue(df_cnw["a*"], df_cnw["b*"])
    df_cnw["Skor_batas"] = skor_garis(df_cnw["a*"], df_cnw["b*"], garis)
    df_cnw["Pred_Undertone"] = prediksi_undertone(df_cnw["a*"], df_cnw["b*"], garis)
    df_lv, hasil = latih_level(df_cnw)
    return garis, df_cnw, df_lv, hasil


def baca_upload(file, kunci):
    data = file.getvalue() if file is not None else None
    if data is None:
        return None, None
    if file.name.lower().endswith(".csv"):
        return pd.read_csv(io.BytesIO(data)), file.name
    sheets = daftar_sheet(data)
    sheet = st.selectbox("Sheet", sheets, key=f"sheet_{kunci}") if len(sheets) > 1 else sheets[0]
    return baca_excel(data, sheet), f"{file.name} / {sheet}"


def pilih_kolom(df, jenis_list, kunci, wajib=True):
    kolom = list(df.columns)
    pilihan = {}
    cols = st.columns(len(jenis_list))
    for c, (jenis, label) in zip(cols, jenis_list):
        opsi = kolom if wajib else ["(tidak ada)"] + kolom
        tebakan = tebak_kolom(kolom, jenis)
        idx = opsi.index(tebakan) if tebakan in opsi else 0
        pilihan[jenis] = c.selectbox(label, opsi, index=idx, key=f"{kunci}_{jenis}")
    return pilihan


# ============================================================================ 7. UI
st.set_page_config(page_title="Skintone Mapping", layout="wide")
st.title("Skintone Mapping")
st.caption("Klasifikasi undertone Cool / Neutral / Warm dari a*–b*, lalu leveling lightness dari L "
           "per undertone.")

with st.sidebar:
    st.header("Pengaturan model")
    metode = st.radio("Batas undertone", ["Hue", "Kombinasi linear"],
                      help="Hue: batas berupa sudut atan2(b*, a*).\n\n"
                           "Kombinasi linear: batas berupa garis cos(θ)·a* + sin(θ)·b*, "
                           "θ dicari 0–360° tiap 0,5°.")
    metode = "Hue" if metode == "Hue" else "Linear"
    st.divider()
    st.markdown(f"**Peta level** {PETA_LEVEL}  \n**Toleransi** ±{TOLERANSI:g} level")

tab_latih, tab_prediksi = st.tabs(["1 · Data latih", "2 · Prediksi data baru"])

# ---------------------------------------------------------------------------- training
with tab_latih:
    st.subheader("Data latih")
    st.write("Upload data berlabel pakar. Setiap upload baru akan menghitung ulang semua batas.")
    file_latih = st.file_uploader("File data latih (.xlsx / .csv)", type=["xlsx", "xls", "csv"], key="up_latih")

    df_latih_raw, sumber = baca_upload(file_latih, "latih")
    if df_latih_raw is None and Path(FILE_LATIH_DEFAULT).exists():
        df_latih_raw, sumber = pd.read_excel(FILE_LATIH_DEFAULT), f"{FILE_LATIH_DEFAULT} (default)"

    if df_latih_raw is None:
        st.info("Upload file data latih dulu. Kolom yang dibutuhkan: L, a*, b*, "
                "Undertone - Skintone (mis. \"Neutral - 3,5\"), dan Overtone Level.")
        st.stop()

    st.caption(f"Sumber: {sumber}  ·  {len(df_latih_raw)} baris")
    kol = pilih_kolom(df_latih_raw, [("L", "Kolom L"), ("a*", "Kolom a*"), ("b*", "Kolom b*"),
                                     ("label", "Kolom undertone - skintone"), ("level", "Kolom level")], "latih")
    try:
        df_cnw0 = siapkan_data_latih(df_latih_raw, kol["L"], kol["a*"], kol["b*"], kol["label"], kol["level"])
    except Exception as e:
        st.error(f"Data latih tidak bisa dibaca dengan kolom yang dipilih: {e}")
        st.stop()
    hilang = [k for k in KATEGORI if (df_cnw0["Undertone"] == k).sum() == 0]
    if hilang:
        st.error(f"Data latih tidak punya subjek undertone murni {hilang}. Periksa kolom label.")
        st.stop()

    kolom_simpan = [c for c in df_cnw0.columns if c in
                    {"L", "a*", "b*", "Undertone", "Skintone", "Overtone Level", kol["label"]}
                    or c == tebak_kolom(df_cnw0.columns, "id")]
    garis, df_cnw, df_lv, hasil = latih(df_cnw0[kolom_simpan].reset_index(drop=True).to_json(orient="split"),
                                        metode)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Data C/N/W murni", len(df_cnw))
    m2.metric("Akurasi undertone", f"{garis['acc'] * 100:.1f}%")
    if metode == "Hue":
        m3.metric("Batas Hue", " | ".join(f"{b:.1f}°" for b in garis["batas"]))
    else:
        m3.metric("θ terbaik", f"{garis['theta']}°")
    m4.metric("Urutan kelas", " < ".join(garis["urutan"]))
    if metode == "Linear":
        st.caption(f"Skor = {garis['c']:.4f}·a* + {garis['s']:.4f}·b*  ·  batas skor "
                   f"{[round(x, 3) for x in garis['batas']]}")

    st.markdown("#### Batas undertone di bidang a*–b*")
    st.pyplot(plot_ab_latih(df_cnw, garis))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Akurasi per kategori**")
        st.dataframe(pd.DataFrame({
            "n": [(df_cnw["Undertone"] == k).sum() for k in KATEGORI],
            "Akurasi (%)": [(df_cnw.loc[df_cnw["Undertone"] == k, "Pred_Undertone"] == k).mean() * 100
                            for k in KATEGORI]}, index=KATEGORI).round(2), width="stretch")
    with c2:
        st.markdown("**Confusion matrix** (baris = asli, kolom = prediksi)")
        st.dataframe(pd.DataFrame(confusion_matrix(df_cnw["Undertone"], df_cnw["Pred_Undertone"], labels=KATEGORI),
                                  index=KATEGORI, columns=KATEGORI), width="stretch")

    st.markdown("#### Leveling lightness per undertone")
    st.pyplot(plot_level(df_lv, hasil, "Level", lambda ut, sub, h: f"{ut} - {h['tepat']:.1f}%"))
    for ut in KATEGORI:
        if ut in hasil and not hasil[ut]["urut"]:
            st.warning(f"{ut}: median L tidak urut, ada level yang rentangnya terbalik. Cek datanya.")

    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.markdown("**Aturan level**")
        st.dataframe(tabel_aturan_level(hasil), hide_index=True, width="stretch")
    with c2:
        st.markdown("**Akurasi level (train)**")
        st.dataframe(pd.DataFrame({
            "n": [hasil[u]["n"] for u in hasil],
            "Tepat (%)": [hasil[u]["tepat"] for u in hasil],
            f"±{TOLERANSI:g} level (%)": [hasil[u]["dekat"] for u in hasil]}, index=list(hasil)).round(2),
            width="stretch")

    info_model = tabel_model(garis, hasil, len(df_cnw))
    aturan_level = tabel_aturan_level(hasil)

# ---------------------------------------------------------------------------- prediksi
with tab_prediksi:
    st.subheader("Prediksi data baru")
    st.write(f"Batas yang dipakai: metode **{'Hue' if metode == 'Hue' else 'kombinasi linear'}** "
             f"dari data latih di tab 1.")
    file_pred = st.file_uploader("File yang mau diprediksi (.xlsx / .csv)", type=["xlsx", "xls", "csv"],
                                 key="up_pred")
    df_new_raw, sumber_pred = baca_upload(file_pred, "pred")
    if df_new_raw is None:
        st.info("Upload file berisi kolom L, a*, b*. Kolom lain (mis. Kode Subjek) ikut disimpan di Excel hasil.")
        st.stop()

    st.caption(f"Sumber: {sumber_pred}  ·  {len(df_new_raw)} baris")
    kp = pilih_kolom(df_new_raw, [("L", "Kolom L"), ("a*", "Kolom a*"), ("b*", "Kolom b*")], "pred")
    kl = pilih_kolom(df_new_raw, [("label", "Label pakar (opsional, untuk testing)")], "pred_lbl", wajib=False)

    df_new = df_new_raw.copy()
    df_new["L"] = pd.to_numeric(df_new[kp["L"]], errors="coerce")
    df_new["a*"] = pd.to_numeric(df_new[kp["a*"]], errors="coerce")
    df_new["b*"] = pd.to_numeric(df_new[kp["b*"]], errors="coerce")
    df_new = df_new.loc[:, ~df_new.columns.duplicated(keep="last")]
    n_awal = len(df_new)
    df_new = df_new.dropna(subset=["L", "a*", "b*"]).reset_index(drop=True)
    if len(df_new) < n_awal:
        st.warning(f"{n_awal - len(df_new)} baris dibuang karena L, a*, atau b* kosong / bukan angka.")
    if df_new.empty:
        st.error("Tidak ada baris dengan L, a*, b* yang valid.")
        st.stop()

    df_new["Hue_dihitung"] = hitung_hue(df_new["a*"], df_new["b*"])
    if metode == "Linear":
        df_new["Skor_linear"] = skor_garis(df_new["a*"], df_new["b*"], garis)
    df_new["Undertone"] = prediksi_undertone(df_new["a*"], df_new["b*"], garis)
    df_new["Level"] = prediksi_level(df_new["L"], df_new["Undertone"], hasil)
    df_new["L_di_luar_rentang_latih"] = [
        not (hasil[u]["rentang_L"][0] <= l <= hasil[u]["rentang_L"][1]) for u, l in zip(df_new["Undertone"], df_new["L"])
    ]

    n = len(df_new)
    jml = df_new["Undertone"].value_counts().reindex(KATEGORI, fill_value=0)
    m = st.columns(5)
    m[0].metric("Subjek", n)
    for i, k in enumerate(KATEGORI):
        m[i + 1].metric(k, f"{jml[k] / n * 100:.1f}%")
    m[4].metric("L di luar rentang latih", f"{df_new['L_di_luar_rentang_latih'].mean() * 100:.1f}%")

    st.markdown("#### Undertone di bidang a*–b*")
    st.pyplot(plot_ab_prediksi(df_new, df_cnw, garis))

    st.markdown("#### Leveling lightness (garis & median dari data latih)")
    st.pyplot(plot_level(df_new, hasil, "Level", lambda ut, sub, h: f"{ut} - n={len(sub)}"))

    ringkasan = (df_new.groupby("Undertone")
                 .agg(n=("L", "size"), L_rata2=("L", "mean"), L_median=("L", "median"),
                      a_rata2=("a*", "mean"), b_rata2=("b*", "mean"),
                      Level_median=("Level", "median"),
                      Level_modus=("Level", lambda x: x.mode().iloc[0]),
                      L_luar_rentang=("L_di_luar_rentang_latih", "sum"))
                 .reindex(KATEGORI))
    ringkasan.insert(1, "%", ringkasan["n"] / n * 100)
    ringkasan.loc["Total"] = [n, 100, df_new["L"].mean(), df_new["L"].median(),
                              df_new["a*"].mean(), df_new["b*"].mean(),
                              df_new["Level"].median(), df_new["Level"].mode().iloc[0],
                              df_new["L_di_luar_rentang_latih"].sum()]
    ringkasan = ringkasan.round(2)
    tabel_ut_lv = pd.crosstab(df_new["Level"], df_new["Undertone"], margins=True, margins_name="Total") \
                    .reindex(columns=KATEGORI + ["Total"], fill_value=0)
    tabel_ut_lv.index = [f"{x:g}" if isinstance(x, (int, float)) else str(x) for x in tabel_ut_lv.index]

    c1, c2 = st.columns([1.4, 1])
    c1.markdown("**Ringkasan per undertone**"); c1.dataframe(ringkasan, width="stretch")
    c2.markdown("**Jumlah subjek: level × undertone**"); c2.dataframe(tabel_ut_lv, width="stretch")

    # ---- testing (bila file punya label pakar)
    evaluasi = None
    if kl["label"] != "(tidak ada)":
        uji = split_undertone_skintone(df_new, column=kl["label"], new_undertone_col="Undertone_asli",
                                       new_skintone_col="Level_asli")
        uji["Undertone_asli"] = uji["Undertone_asli"].astype(str).str.strip().str.title()
        uji = uji[uji["Undertone_asli"].isin(KATEGORI)].copy()
        if len(uji):
            uji["Level_asli_gabung"] = gabung(pd.to_numeric(uji["Level_asli"], errors="coerce"))
            benar_lv = uji["Level"] == uji["Level_asli_gabung"]
            dekat_lv = (uji["Level"] - uji["Level_asli_gabung"]).abs() <= TOLERANSI
            baris = []
            for grup, d in [(k, uji[uji["Undertone_asli"] == k]) for k in KATEGORI] + [("Total", uji)]:
                ada_lv = d["Level_asli_gabung"].notna()
                baris.append({"Undertone asli": grup, "n": len(d),
                              "Akurasi undertone (%)": (d["Undertone"] == d["Undertone_asli"]).mean() * 100,
                              "Level tepat (%)": benar_lv[d.index][ada_lv].mean() * 100,
                              f"Level ±{TOLERANSI:g} (%)": dekat_lv[d.index][ada_lv].mean() * 100})
            evaluasi = pd.DataFrame(baris).set_index("Undertone asli").round(2)
            st.markdown("#### Testing terhadap label pakar")
            st.caption("Hanya baris dengan label undertone murni C/N/W. Akurasi level dihitung pada level asli "
                       "yang sudah digabung (2,5 → 2, 3,5 → 3).")
            c1, c2 = st.columns([1.4, 1])
            c1.dataframe(evaluasi, width="stretch")
            c2.dataframe(pd.DataFrame(confusion_matrix(uji["Undertone_asli"], uji["Undertone"], labels=KATEGORI),
                                      index=KATEGORI, columns=KATEGORI), width="stretch")
            df_new.loc[uji.index, "Undertone_asli"] = uji["Undertone_asli"]
            df_new.loc[uji.index, "Level_asli"] = uji["Level_asli_gabung"]
        else:
            st.info("Kolom label tidak berisi undertone murni Cool / Neutral / Warm, testing dilewati.")

    st.markdown("#### Tabel prediksi")
    st.dataframe(df_new, width="stretch", height=320)

    nama = Path(file_pred.name).stem
    st.download_button("Unduh Excel prediksi",
                       data=buat_excel(df_new, info_model, aturan_level, ringkasan, tabel_ut_lv, evaluasi),
                       file_name=f"Prediksi_{nama}_{metode}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       type="primary")
