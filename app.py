# -*- coding: utf-8 -*-
"""Trade Lab - 二階堂式リバウンド手法 ケーススタディDB（iPhoneデプロイ用）

これは実トレード記録ではない。「その場でどう見えたか / なぜ反発・失敗したか」を
事例として蓄積し、再現性のあるパターンへ昇華するための研究DB。

データ保存先: Googleスプレッドシート（cases タブ・1事例=1行） / 画像: Cloudinary。
Streamlit secrets:
  - spreadsheet_id            : 専用スプレッドシートID
  - gcp_service_account_json  : サービスアカウントJSON（まるごと文字列）
  - cloudinary_cloud_name / cloudinary_api_key / cloudinary_api_secret
"""
import io
import json
import datetime as dt

JST = dt.timezone(dt.timedelta(hours=9))


def now_jst():
    return dt.datetime.now(JST)


import numpy as np
import pandas as pd
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

try:
    import cloudinary
    import cloudinary.uploader
    _CLOUDINARY_LIB = True
except Exception:
    _CLOUDINARY_LIB = False


# =========================================================
# 憲法に基づく定数
# =========================================================
# パターン分類（憲法のフォルダ構成）
PATTERNS = [
    "GD急落リバ", "寄り天急落リバ", "二番底リバ", "VWAP回帰",
    "空売り買戻し", "急騰後押し目", "その他",
]
# 下げ止まりサイン
BOTTOMING_SIGNS = [
    "下ヒゲ", "出来高急増", "出来高減少", "十字線",
    "安値切り上げ", "高値切り上げ", "VWAP回復", "二番底形成", "その他",
]
EVALS = ["○", "△", "×"]

# 画像スロット: (キー, ラベル, 必須か)
IMG_SLOTS = [
    ("img_1min", "1分足", True),
    ("img_5min", "5分足", True),
    ("img_daily", "日足（推奨）", False),
    ("img_board", "板（任意）", False),
    ("img_tape", "歩み値（任意）", False),
    ("img_ranking", "ランキング（任意）", False),
]

STOCK_MASTER_SEED = {"7746": "岡本硝子"}


# =========================================================
# データ層（Googleスプレッドシート）
# =========================================================
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

CASES_HEADERS = [
    "id", "created_at", "case_date", "stock_code", "stock_name", "pattern",
    # 数値（入力）
    "prev_close", "open_price", "pre_high", "low_price",
    # 数値（自動）
    "gd_pct", "post_open_drop_pct", "total_drop_pct",
    # 時刻と所要
    "drop_start_time", "low_time", "drop_minutes",
    # サイン・評価
    "bottoming_signs", "sellout_eval", "sellout_reason",
    "entry_eval", "planned_entry", "planned_stop", "planned_target", "planned_rr",
    # 結果・学び
    "result_text", "max_rebound", "realized_rr", "learning",
    # 画像
    "img_1min", "img_5min", "img_daily", "img_board", "img_tape", "img_ranking",
]
STOCK_HEADERS = ["code", "name"]


def _load_credentials():
    if "gcp_service_account_json" in st.secrets:
        info = json.loads(st.secrets["gcp_service_account_json"])
    elif "gcp_service_account" in st.secrets:
        info = dict(st.secrets["gcp_service_account"])
    else:
        raise KeyError("secrets に gcp_service_account_json がありません。")
    return Credentials.from_service_account_info(info, scopes=SCOPES)


@st.cache_resource
def _spreadsheet():
    gc = gspread.authorize(_load_credentials())
    return gc.open_by_key(st.secrets["spreadsheet_id"])


def _ensure_headers(ws, headers):
    existing = ws.row_values(1)
    if not existing:
        if ws.col_count < len(headers):
            ws.add_cols(len(headers) - ws.col_count)
        ws.append_row(headers, value_input_option="RAW")
        return
    if headers[:len(existing)] == existing and len(headers) > len(existing):
        if ws.col_count < len(headers):
            ws.add_cols(len(headers) - ws.col_count)
        for idx in range(len(existing), len(headers)):
            ws.update_cell(1, idx + 1, headers[idx])


def _ws(name, headers):
    ss = _spreadsheet()
    try:
        ws = ss.worksheet(name)
    except gspread.exceptions.WorksheetNotFound:
        ws = ss.add_worksheet(title=name, rows=2000, cols=max(10, len(headers)))
        ws.append_row(headers, value_input_option="RAW")
        return ws
    _ensure_headers(ws, headers)
    return ws


@st.cache_resource
def init_db():
    _ws("cases", CASES_HEADERS)
    _ws("stock_master", STOCK_HEADERS)
    master = _load_stock_master()
    for code, name in STOCK_MASTER_SEED.items():
        if code not in master:
            upsert_stock(code, name)
    return True


# =========================================================
# Cloudinary
# =========================================================
def cloudinary_ready():
    if not _CLOUDINARY_LIB:
        return False
    return all(k in st.secrets for k in
               ["cloudinary_cloud_name", "cloudinary_api_key", "cloudinary_api_secret"])


def _config_cloudinary():
    cloudinary.config(
        cloud_name=st.secrets["cloudinary_cloud_name"],
        api_key=st.secrets["cloudinary_api_key"],
        api_secret=st.secrets["cloudinary_api_secret"],
        secure=True,
    )


def upload_one(file):
    """1ファイルを Cloudinary に上げ secure_url を返す。失敗時 None。"""
    if not file or not cloudinary_ready():
        return None
    _config_cloudinary()
    try:
        res = cloudinary.uploader.upload(
            io.BytesIO(file.getvalue()), folder="tradelab", resource_type="image")
        return res.get("secure_url")
    except Exception as e:
        st.warning(f"画像アップロード失敗（{getattr(file, 'name', '?')}）: {e}")
        return None


# =========================================================
# 計算（憲法の定義式）
# =========================================================
def calc_gd(prev_close, open_price):
    # GD率 = (前日終値 - 始値) / 前日終値 × 100
    if not prev_close or prev_close <= 0:
        return None
    return round((prev_close - open_price) / prev_close * 100, 2)


def calc_post_open_drop(open_price, low_price):
    # 寄り後急落率 = (始値 - 安値) / 始値 × 100
    if not open_price or open_price <= 0:
        return None
    return round((open_price - low_price) / open_price * 100, 2)


def calc_total_drop(prev_close, low_price):
    # 総急落率 = (前日終値 - 安値) / 前日終値 × 100
    if not prev_close or prev_close <= 0:
        return None
    return round((prev_close - low_price) / prev_close * 100, 2)


def _parse_hhmm(s):
    if not s or not isinstance(s, str) or ":" not in s:
        return None
    try:
        h, m = s.split(":")[:2]
        return int(h) * 60 + int(m)
    except Exception:
        return None


def calc_drop_minutes(start_hhmm, low_hhmm):
    # 値段が付いてから計測（寄り前は含めない）。start→low の経過分。
    a = _parse_hhmm(start_hhmm)
    b = _parse_hhmm(low_hhmm)
    if a is None or b is None:
        return None
    diff = b - a
    return diff if diff >= 0 else None


def calc_planned_rr(entry, stop, target):
    if not entry or not stop or not target or entry <= 0:
        return None
    risk = abs(entry - stop)
    reward = abs(target - entry)
    if risk <= 0:
        return None
    return round(reward / risk, 2)


def calc_realized_rr(entry, stop, max_rebound):
    if not entry or not stop or not max_rebound or max_rebound <= 0:
        return None
    risk = abs(entry - stop)
    if risk <= 0:
        return None
    return round(max_rebound / risk, 2)


# =========================================================
# 銘柄マスタ
# =========================================================
@st.cache_data(ttl=600)
def _load_stock_master():
    ws = _ws("stock_master", STOCK_HEADERS)
    out = {}
    for r in ws.get_all_records():
        code = str(r.get("code", "")).strip()
        name = str(r.get("name", "")).strip()
        if code:
            out[code] = name
    return out


def lookup_stock(code):
    if not code:
        return None
    return _load_stock_master().get(code.strip())


def upsert_stock(code, name):
    code = (code or "").strip()
    name = (name or "").strip()
    if not code or not name:
        return
    ws = _ws("stock_master", STOCK_HEADERS)
    codes = ws.col_values(1)
    if code in codes[1:]:
        ws.update_cell(codes.index(code) + 1, 2, name)
    else:
        ws.append_row([code, name], value_input_option="RAW")
    _load_stock_master.clear()


def all_stocks_df():
    m = _load_stock_master()
    if not m:
        return pd.DataFrame(columns=["code", "name"])
    return pd.DataFrame(sorted(m.items()), columns=["code", "name"])


# =========================================================
# ケース記録
# =========================================================
@st.cache_data(ttl=60)
def _load_cases_records():
    return _ws("cases", CASES_HEADERS).get_all_records()


def _next_id(recs):
    ids = [int(r["id"]) for r in recs if str(r.get("id", "")).strip().isdigit()]
    return max(ids) + 1 if ids else 1


def add_case(d):
    gd = calc_gd(d["prev_close"], d["open_price"])
    pod = calc_post_open_drop(d["open_price"], d["low_price"])
    tot = calc_total_drop(d["prev_close"], d["low_price"])
    dmin = calc_drop_minutes(d["drop_start_time"], d["low_time"])
    prr = calc_planned_rr(d["planned_entry"], d["planned_stop"], d["planned_target"])
    rrr = calc_realized_rr(d["planned_entry"], d["planned_stop"], d["max_rebound"])
    ws = _ws("cases", CASES_HEADERS)
    new_id = _next_id(_load_cases_records())
    created = now_jst().strftime("%Y-%m-%d %H:%M:%S")

    def blank(x):
        return "" if x is None else x

    ws.append_row([
        new_id, created, d["case_date"], d["stock_code"], d["stock_name"], d["pattern"],
        blank(d["prev_close"]), blank(d["open_price"]), blank(d["pre_high"]), blank(d["low_price"]),
        blank(gd), blank(pod), blank(tot),
        d["drop_start_time"], d["low_time"], blank(dmin),
        json.dumps(d["bottoming_signs"], ensure_ascii=False),
        d["sellout_eval"], d["sellout_reason"],
        d["entry_eval"], blank(d["planned_entry"]), blank(d["planned_stop"]),
        blank(d["planned_target"]), blank(prr),
        d["result_text"], blank(d["max_rebound"]), blank(rrr), d["learning"],
        d.get("img_1min", ""), d.get("img_5min", ""), d.get("img_daily", ""),
        d.get("img_board", ""), d.get("img_tape", ""), d.get("img_ranking", ""),
    ], value_input_option="RAW")
    _load_cases_records.clear()
    return new_id


def delete_case(cid):
    ws = _ws("cases", CASES_HEADERS)
    ids = ws.col_values(1)
    target = str(int(cid))
    for i, v in enumerate(ids):
        if i == 0:
            continue
        if str(v).strip() == target:
            ws.delete_rows(i + 1)
            break
    _load_cases_records.clear()


def _safe_json(s):
    try:
        v = json.loads(s) if s else []
        return v if isinstance(v, list) else []
    except Exception:
        return []


def get_cases_df():
    recs = _load_cases_records()
    if not recs:
        return pd.DataFrame()
    df = pd.DataFrame(recs)
    for c in CASES_HEADERS:
        if c not in df.columns:
            df[c] = None
    num_cols = ["id", "prev_close", "open_price", "pre_high", "low_price",
                "gd_pct", "post_open_drop_pct", "total_drop_pct", "drop_minutes",
                "planned_entry", "planned_stop", "planned_target", "planned_rr",
                "max_rebound", "realized_rr"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["id"] = df["id"].fillna(0).astype(int)
    df["bottoming_signs"] = df["bottoming_signs"].apply(_safe_json)
    for c in ["case_date", "stock_code", "stock_name", "pattern", "sellout_eval",
              "sellout_reason", "entry_eval", "result_text", "learning",
              "drop_start_time", "low_time",
              "img_1min", "img_5min", "img_daily", "img_board", "img_tape", "img_ranking"]:
        df[c] = df[c].fillna("").astype(str)
    df = df.sort_values(["case_date", "id"], ascending=[False, False]).reset_index(drop=True)
    return df


# =========================================================
# 分析（パターンの傾向を探す）
# =========================================================
def _rate(series, val):
    n = len(series)
    if n == 0:
        return 0.0
    return round((series == val).mean() * 100, 1)


def pattern_summary(df):
    """パターン分類別の傾向。"""
    if df.empty:
        return pd.DataFrame()
    rows = []
    for p in PATTERNS:
        sub = df[df["pattern"] == p]
        n = len(sub)
        if n == 0:
            continue
        rows.append({
            "パターン": p,
            "事例数": n,
            "売り枯れ○率": _rate(sub["sellout_eval"], "○"),
            "エントリー○率": _rate(sub["entry_eval"], "○"),
            "平均GD率": round(sub["gd_pct"].mean(), 2) if sub["gd_pct"].notna().any() else np.nan,
            "平均総急落率": round(sub["total_drop_pct"].mean(), 2) if sub["total_drop_pct"].notna().any() else np.nan,
            "平均急落分": round(sub["drop_minutes"].mean(), 1) if sub["drop_minutes"].notna().any() else np.nan,
            "平均想定RR": round(sub["planned_rr"].mean(), 2) if sub["planned_rr"].notna().any() else np.nan,
            "平均反発幅": round(sub["max_rebound"].mean(), 1) if sub["max_rebound"].notna().any() else np.nan,
        })
    return pd.DataFrame(rows)


def sign_summary(df, min_count=1):
    """下げ止まりサイン別：そのサインが付いた事例の売り枯れ○率・平均反発幅。
    どのサインが本物の売り枯れ／良い反発に効くかを見る。"""
    if df.empty:
        return pd.DataFrame()
    rows = []
    for sign in BOTTOMING_SIGNS:
        sub = df[df["bottoming_signs"].apply(lambda lst: sign in (lst or []))]
        n = len(sub)
        if n < min_count:
            continue
        rows.append({
            "サイン": sign,
            "出現数": n,
            "売り枯れ○率": _rate(sub["sellout_eval"], "○"),
            "平均反発幅": round(sub["max_rebound"].mean(), 1) if sub["max_rebound"].notna().any() else np.nan,
            "平均実現RR": round(sub["realized_rr"].mean(), 2) if sub["realized_rr"].notna().any() else np.nan,
        })
    res = pd.DataFrame(rows)
    if not res.empty:
        res = res.sort_values("売り枯れ○率", ascending=False).reset_index(drop=True)
    return res


def sellout_outcome(df):
    """売り枯れ評価(○/△/×)別の結果。下げ止まりと売り枯れの混同を検証する。"""
    if df.empty:
        return pd.DataFrame()
    rows = []
    for ev in EVALS:
        sub = df[df["sellout_eval"] == ev]
        n = len(sub)
        if n == 0:
            continue
        rows.append({
            "売り枯れ評価": ev,
            "事例数": n,
            "平均反発幅": round(sub["max_rebound"].mean(), 1) if sub["max_rebound"].notna().any() else np.nan,
            "平均実現RR": round(sub["realized_rr"].mean(), 2) if sub["realized_rr"].notna().any() else np.nan,
            "エントリー○率": _rate(sub["entry_eval"], "○"),
        })
    return pd.DataFrame(rows)


def drop_band_summary(df):
    """総急落率の帯別に、反発の出やすさを見る。"""
    if df.empty or df["total_drop_pct"].notna().sum() == 0:
        return pd.DataFrame()
    d = df[df["total_drop_pct"].notna()].copy()
    bins = [0, 5, 10, 15, 20, 100]
    labels = ["〜5%", "5〜10%", "10〜15%", "15〜20%", "20%〜"]
    d["帯"] = pd.cut(d["total_drop_pct"], bins=bins, labels=labels, right=False)
    rows = []
    for lab in labels:
        sub = d[d["帯"] == lab]
        n = len(sub)
        if n == 0:
            continue
        rows.append({
            "総急落率帯": lab,
            "事例数": n,
            "売り枯れ○率": _rate(sub["sellout_eval"], "○"),
            "平均反発幅": round(sub["max_rebound"].mean(), 1) if sub["max_rebound"].notna().any() else np.nan,
        })
    return pd.DataFrame(rows)


def generate_insights(df, min_count=3):
    if df.empty:
        return ["まだ事例がありません。1分足・5分足を添えて事例を貯めましょう。"]
    n = len(df)
    lines = [f"📚 蓄積事例数: {n}件。"]
    if n < 10:
        lines.append("⚠️ 憲法原則: 10事例で結論を出さない。まずは100事例を目標に蓄積を。")

    ps = pattern_summary(df)
    ps_valid = ps[ps["事例数"] >= min_count] if not ps.empty else ps
    if not ps_valid.empty and ps_valid["平均反発幅"].notna().any():
        best = ps_valid.sort_values("平均反発幅", ascending=False).iloc[0]
        lines.append(f"🏆 平均反発幅が最大のパターンは「{best['パターン']}」"
                     f"（平均{best['平均反発幅']} / 売り枯れ○率{best['売り枯れ○率']}% / {int(best['事例数'])}件）。")

    ss = sign_summary(df, min_count)
    if not ss.empty:
        top = ss.iloc[0]
        lines.append(f"🔎 売り枯れ○率が高いサインは「{top['サイン']}」"
                     f"（○率{top['売り枯れ○率']}% / 出現{int(top['出現数'])}件）。")

    so = sellout_outcome(df)
    if not so.empty and so["平均反発幅"].notna().any():
        o = so[so["売り枯れ評価"] == "○"]
        x = so[so["売り枯れ評価"] == "×"]
        if not o.empty and not x.empty and pd.notna(o.iloc[0]["平均反発幅"]) and pd.notna(x.iloc[0]["平均反発幅"]):
            lines.append(f"🧪 売り枯れ○の平均反発幅 {o.iloc[0]['平均反発幅']} vs ×の {x.iloc[0]['平均反発幅']}。"
                         "「下げ止まり≠売り枯れ」を数値で検証できます。")

    db = drop_band_summary(df)
    if not db.empty and db["売り枯れ○率"].notna().any():
        b = db.sort_values("売り枯れ○率", ascending=False).iloc[0]
        lines.append(f"📉 総急落率「{b['総急落率帯']}」帯が売り枯れ○率最高（{b['売り枯れ○率']}% / {int(b['事例数'])}件）。")

    lines.append("※ 個別銘柄ではなくパターンを見る。目的は未来予測ではなく期待値の高い局面の発見。")
    return lines


# =========================================================
# UI
# =========================================================
st.set_page_config(page_title="Trade Lab", page_icon="🔬", layout="centered")

try:
    init_db()
except Exception as e:
    st.error("スプレッドシートに接続できません。Secrets（gcp_service_account_json / spreadsheet_id）と、"
             "サービスアカウントへのシート共有（編集者）を確認してください。")
    st.exception(e)
    st.stop()

st.markdown(
    "<style>.block-container{padding-top:1.2rem;padding-bottom:3rem;}"
    "div[data-testid='stMetricValue']{font-size:1.3rem;}</style>",
    unsafe_allow_html=True,
)

st.sidebar.title("🔬 Trade Lab")
st.sidebar.caption("二階堂式リバウンド ケーススタディDB")
if not cloudinary_ready():
    st.sidebar.caption("ℹ️ 画像アップロード未設定（Cloudinary secrets / requirements を確認）")
page = st.sidebar.radio("メニュー",
                        ["🔬 ケース入力", "📚 ケース一覧", "📊 パターン分析", "🤖 AI考察", "🏷 銘柄マスタ"])


def _seed_time(key, default_str="09:00"):
    if key not in st.session_state:
        h, m = default_str.split(":")
        st.session_state[key] = dt.time(int(h), int(m))


def page_input():
    st.header("🔬 ケース入力")
    st.caption("実エントリーの有無は問わない。「どう見えたか/なぜ反発・失敗したか」を残す。")

    # 基本情報
    c1, c2 = st.columns(2)
    with c1:
        case_date = st.date_input("日付", value=now_jst().date())
    with c2:
        pattern = st.selectbox("パターン分類 ⭐必須", ["— 選択 —"] + PATTERNS, index=0)
    c3, c4 = st.columns([1, 2])
    with c3:
        code = st.text_input("銘柄コード ⭐必須", placeholder="例: 7746", max_chars=6)
    looked = lookup_stock(code) if code else None
    with c4:
        stock_name = st.text_input("銘柄名", value=looked or "",
                                   placeholder="未登録なら入力（次回から自動）")
    if code and looked:
        st.caption(f"✅ マスタ照合: {code} → {looked}")

    st.divider()
    # 数値
    st.subheader("数値")
    c5, c6 = st.columns(2)
    with c5:
        prev_close = st.number_input("前日終値", min_value=0.0, value=0.0, step=1.0, format="%.1f")
        pre_high = st.number_input("急落前高値", min_value=0.0, value=0.0, step=1.0, format="%.1f")
    with c6:
        open_price = st.number_input("始値", min_value=0.0, value=0.0, step=1.0, format="%.1f")
        low_price = st.number_input("安値", min_value=0.0, value=0.0, step=1.0, format="%.1f")

    gd = calc_gd(prev_close, open_price)
    pod = calc_post_open_drop(open_price, low_price)
    tot = calc_total_drop(prev_close, low_price)
    mcols = st.columns(3)
    mcols[0].metric("GD率", "—" if gd is None else f"{gd:+.2f}%")
    mcols[1].metric("寄り後急落率", "—" if pod is None else f"{pod:.2f}%")
    mcols[2].metric("総急落率", "—" if tot is None else f"{tot:.2f}%")

    # 時刻
    _seed_time("drop_start_w", "09:00")
    _seed_time("low_w", "09:05")
    c7, c8 = st.columns(2)
    with c7:
        drop_start = st.time_input("急落開始時刻", key="drop_start_w")
    with c8:
        low_time = st.time_input("安値時刻", key="low_w")
    dmin = calc_drop_minutes(drop_start.strftime("%H:%M"), low_time.strftime("%H:%M"))
    st.caption(f"⏱ 急落所要時間: {'—' if dmin is None else f'{dmin}分'}（値段が付いてからを計測）")

    st.divider()
    # 下げ止まりサイン
    st.subheader("下げ止まりサイン")
    signs = []
    sc = st.columns(2)
    for i, s in enumerate(BOTTOMING_SIGNS):
        with sc[i % 2]:
            if st.checkbox(s, key=f"sign_{s}"):
                signs.append(s)

    st.divider()
    # 売り枯れ評価（最重要）
    st.subheader("売り枯れ評価")
    st.caption("『下げ止まり（今売られてない）』と『売り枯れ（今後も売りが少ない）』を混同しない。狙うのは売り枯れ。")
    sellout_eval = st.radio("評価", EVALS, horizontal=True, index=None, key="sellout_eval_w")
    sellout_reason = st.text_area("理由", height=70, placeholder="なぜそう判断したか")

    st.divider()
    # エントリー評価
    st.subheader("エントリー評価")
    entry_eval = st.radio("評価", EVALS, horizontal=True, index=None, key="entry_eval_w")
    e1, e2, e3 = st.columns(3)
    with e1:
        planned_entry = st.number_input("想定エントリー", min_value=0.0, value=0.0, step=1.0, format="%.1f")
    with e2:
        planned_stop = st.number_input("想定損切り", min_value=0.0, value=0.0, step=1.0, format="%.1f")
    with e3:
        planned_target = st.number_input("想定利確", min_value=0.0, value=0.0, step=1.0, format="%.1f")
    prr = calc_planned_rr(planned_entry, planned_stop, planned_target)
    if prr is not None:
        st.caption(f"📐 想定リスクリワード: {prr}")

    st.divider()
    # 結果
    st.subheader("結果")
    result_text = st.text_area("その後どうなったか", height=70)
    max_rebound = st.number_input("最大反発幅（安値からの戻り幅・円）", min_value=0.0, value=0.0, step=1.0, format="%.1f")
    rrr = calc_realized_rr(planned_entry, planned_stop, max_rebound)
    if rrr is not None:
        st.caption(f"📐 実現リスクリワード（反発幅÷想定リスク）: {rrr}")

    # 学び
    learning = st.text_area("学び（簡潔に）", height=80)

    st.divider()
    # 画像
    st.subheader("📸 画像")
    if not cloudinary_ready():
        st.warning("Cloudinary未設定のため画像アップロードは無効です。")
    img_files = {}
    for key, label, required in IMG_SLOTS:
        tag = " ⭐必須" if required else ""
        img_files[key] = st.file_uploader(f"{label}{tag}", type=["png", "jpg", "jpeg", "webp"],
                                          key=f"up_{key}", accept_multiple_files=False)

    st.divider()
    if st.button("💾 ケースを保存", type="primary", use_container_width=True):
        errors = []
        if pattern == "— 選択 —":
            errors.append("パターン分類")
        if not code:
            errors.append("銘柄コード")
        if cloudinary_ready():
            if not img_files["img_1min"]:
                errors.append("1分足画像（必須）")
            if not img_files["img_5min"]:
                errors.append("5分足画像（必須）")
        if errors:
            st.error("未入力: " + " / ".join(errors))
            return

        # 画像アップロード
        urls = {}
        for key, label, required in IMG_SLOTS:
            f = img_files.get(key)
            if f is not None:
                u = upload_one(f)
                if u is None and required:
                    st.error(f"{label}のアップロードに失敗しました。通信を確認して再試行してください。")
                    return
                urls[key] = u or ""
            else:
                urls[key] = ""

        if stock_name:
            upsert_stock(code, stock_name)
        add_case({
            "case_date": case_date.isoformat(), "pattern": pattern,
            "stock_code": code, "stock_name": stock_name,
            "prev_close": prev_close or None, "open_price": open_price or None,
            "pre_high": pre_high or None, "low_price": low_price or None,
            "drop_start_time": drop_start.strftime("%H:%M"),
            "low_time": low_time.strftime("%H:%M"),
            "bottoming_signs": signs,
            "sellout_eval": sellout_eval or "", "sellout_reason": sellout_reason,
            "entry_eval": entry_eval or "",
            "planned_entry": planned_entry or None, "planned_stop": planned_stop or None,
            "planned_target": planned_target or None,
            "result_text": result_text, "max_rebound": max_rebound or None,
            "learning": learning,
            **urls,
        })
        st.success("ケースを保存しました。次の事例へ。")
        st.balloons()


def page_list():
    st.header("📚 ケース一覧")
    df = get_cases_df()
    if df.empty:
        st.info("まだ事例がありません。")
        return

    code_q = ""
    pat_q = "すべて"
    so_q = "すべて"
    with st.expander("🔍 絞り込み"):
        f1, f2 = st.columns(2)
        with f1:
            code_q = st.text_input("銘柄コード", placeholder="例: 7746").strip()
            pat_q = st.selectbox("パターン分類", ["すべて"] + PATTERNS)
        with f2:
            so_q = st.selectbox("売り枯れ評価", ["すべて"] + EVALS)

    fdf = df.copy()
    if code_q:
        fdf = fdf[fdf["stock_code"].astype(str).str.contains(code_q)]
    if pat_q != "すべて":
        fdf = fdf[fdf["pattern"] == pat_q]
    if so_q != "すべて":
        fdf = fdf[fdf["sellout_eval"] == so_q]

    st.caption(f"表示: {len(fdf)} / 全{len(df)}件")
    show = fdf[["id", "case_date", "stock_code", "stock_name", "pattern",
                "total_drop_pct", "sellout_eval", "entry_eval", "max_rebound"]].rename(columns={
        "id": "ID", "case_date": "日付", "stock_code": "コード", "stock_name": "銘柄",
        "pattern": "パターン", "total_drop_pct": "総急落率", "sellout_eval": "売枯",
        "entry_eval": "IN評価", "max_rebound": "反発幅"})
    st.dataframe(show, use_container_width=True, hide_index=True)
    st.download_button("⬇️ CSVエクスポート（バックアップ）",
                       fdf.to_csv(index=False).encode("utf-8-sig"),
                       file_name="tradelab_cases.csv", mime="text/csv",
                       use_container_width=True)

    st.divider()
    st.subheader("詳細 / 削除")
    if fdf.empty:
        st.caption("該当なし。")
        return
    sel = st.selectbox("ID選択", fdf["id"].tolist())
    row = df[df["id"] == sel].iloc[0]
    st.markdown(f"### {row['stock_code']} {row['stock_name']} — {row['pattern']}")
    st.write(f"日付: {row['case_date']}")
    mm = st.columns(3)
    mm[0].metric("GD率", "—" if pd.isna(row["gd_pct"]) else f"{row['gd_pct']:+.2f}%")
    mm[1].metric("総急落率", "—" if pd.isna(row["total_drop_pct"]) else f"{row['total_drop_pct']:.2f}%")
    mm[2].metric("急落所要", "—" if pd.isna(row["drop_minutes"]) else f"{int(row['drop_minutes'])}分")
    st.write("下げ止まりサイン:", "、".join(row["bottoming_signs"]) or "—")
    st.write(f"売り枯れ評価: **{row['sellout_eval'] or '—'}**　/　エントリー評価: **{row['entry_eval'] or '—'}**")
    if row["sellout_reason"]:
        st.caption("売り枯れ理由: " + row["sellout_reason"])
    if pd.notna(row["planned_rr"]):
        st.write("想定RR:", row["planned_rr"], "／ 実現RR:", "—" if pd.isna(row["realized_rr"]) else row["realized_rr"])
    if row["result_text"]:
        st.write("結果:", row["result_text"])
    if pd.notna(row["max_rebound"]):
        st.write("最大反発幅:", row["max_rebound"])
    if row["learning"]:
        st.info("学び: " + row["learning"])
    for key, label, _ in IMG_SLOTS:
        u = row[key]
        if u:
            st.caption(label)
            st.image(u, use_container_width=True)
    if st.button("🗑 このケースを削除", type="secondary"):
        delete_case(int(sel))
        st.warning("削除しました。再読み込みします。")
        st.rerun()


def _style(df):
    return df.style.format(na_rep="—", precision=2)


def page_analysis():
    st.header("📊 パターン分析")
    df = get_cases_df()
    if df.empty:
        st.info("まだ事例がありません。")
        return
    st.caption(f"蓄積 {len(df)} 事例。個別銘柄ではなくパターンを見る。")
    min_count = st.slider("サインの最小出現数", 1, 20, 1)

    st.subheader("パターン分類別")
    ps = pattern_summary(df)
    st.dataframe(_style(ps) if not ps.empty else ps, use_container_width=True, hide_index=True)

    st.subheader("下げ止まりサイン別（売り枯れ○率順）")
    ss = sign_summary(df, min_count)
    if ss.empty:
        st.caption("該当なし。")
    else:
        st.dataframe(_style(ss), use_container_width=True, hide_index=True)

    st.subheader("売り枯れ評価別の結果")
    st.caption("○の反発幅が△×より明確に大きいほど、評価が機能している証拠。")
    so = sellout_outcome(df)
    st.dataframe(_style(so) if not so.empty else so, use_container_width=True, hide_index=True)

    st.subheader("総急落率の帯別")
    db = drop_band_summary(df)
    if db.empty:
        st.caption("該当なし。")
    else:
        st.dataframe(_style(db), use_container_width=True, hide_index=True)


def page_ai():
    st.header("🤖 AI考察")
    st.caption("蓄積事例から傾向を自動で言語化（ルールベース）。")
    df = get_cases_df()
    min_count = st.slider("採用する最小事例数", 1, 20, 3)
    if st.button("🔍 考察する", type="primary", use_container_width=True):
        for line in generate_insights(df, min_count):
            st.markdown(f"- {line}")


def page_master():
    st.header("🏷 銘柄マスタ")
    st.caption("コード→銘柄名。ケース入力時に自動照合されます。")
    with st.form("add_stock"):
        c1, c2 = st.columns([1, 2])
        code = c1.text_input("コード")
        name = c2.text_input("銘柄名")
        if st.form_submit_button("追加 / 更新"):
            if code and name:
                upsert_stock(code, name)
                st.success(f"{code} → {name} を登録しました。")
            else:
                st.error("コードと銘柄名を入力してください。")
    st.dataframe(all_stocks_df(), use_container_width=True, hide_index=True)


{
    "🔬 ケース入力": page_input,
    "📚 ケース一覧": page_list,
    "📊 パターン分析": page_analysis,
    "🤖 AI考察": page_ai,
    "🏷 銘柄マスタ": page_master,
}[page]()
