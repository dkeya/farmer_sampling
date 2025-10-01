import streamlit as st 
import pandas as pd
import numpy as np
import altair as alt
import re
from io import BytesIO
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

st.set_page_config(page_title="SAFIC | Webinar Responses Analyzer", page_icon="🌾", layout="wide")

# =============================
# Utils
# =============================
def load_df(upload):
    """
    Robust loader for your webinar export:
    - Handles UTF-16 + TAB files with a 'Summary' section on top
    - Reassembles wrapped lines in long text fields (no quotes)
    - Still works for normal UTF-8 CSV or Excel files
    """
    raw = upload.read()
    upload.seek(0)

    # --- Case 1: UTF-16 TSV with header "2. Participants" ---
    is_utf16 = raw[:2] in (b"\xff\xfe", b"\xfe\xff")
    if is_utf16:
        text = raw.decode("utf-16")
        lines = text.splitlines()

        # find the "2. Participants" section
        start = None
        for i, line in enumerate(lines):
            if line.strip().startswith("2. Participants"):
                start = i + 1
                break
        if start is None:
            st.error("Could not locate the '2. Participants' section in the file.")
            st.stop()

        header_line = lines[start]
        cols = header_line.split("\t")
        expected = len(cols)

        records = []
        buf = ""
        # accumulate lines until we reach expected tabs
        for line in lines[start + 1:]:
            if not line.strip():
                continue
            buf = line if not buf else f"{buf}\n{line}"
            parts = buf.split("\t")
            if len(parts) >= expected:
                # if extra tabs spill over, glue extras into the last column
                row = parts[:expected - 1] + ["\t".join(parts[expected - 1:])]
                records.append(row)
                buf = ""

        df = pd.DataFrame(records, columns=[c.strip() for c in cols])
        return df

    # --- Case 2: Normal CSV/Excel (UTF-8/1252/etc.) ---
    # Excel disguised as CSV? (xlsx zip starts with 'PK')
    if raw[:2] == b"PK":
        upload.seek(0)
        df = pd.read_excel(upload)
        df.columns = [str(c).strip() for c in df.columns]
        return df

    # Try common CSV encodings
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        try:
            upload.seek(0)
            df = pd.read_csv(upload, encoding=enc, sep=None, engine="python")
            df.columns = [str(c).strip() for c in df.columns]
            return df
        except Exception:
            pass

    # Final fallback: Excel engines
    try:
        upload.seek(0)
        df = pd.read_excel(upload)
        df.columns = [str(c).strip() for c in df.columns]
        return df
    except Exception as e:
        st.error(f"Could not read the file as CSV or Excel. Last error: {e}")
        st.stop()

def coerce_datetime(series):
    try:
        s = pd.to_datetime(series, errors="coerce", dayfirst=False, infer_datetime_format=True)
    except Exception:
        s = pd.to_datetime(series, errors="coerce")
    return s

def normalize_text(x: str) -> str:
    if pd.isna(x): return ""
    x = str(x)
    x = re.sub(r"\s+", " ", x)
    return x.strip()

def clean_email(e):
    e = normalize_text(e).lower()
    return e if re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", e) else ""

def email_domain(e):
    e = clean_email(e)
    return e.split("@")[-1] if "@" in e else ""

def infer_sector(org, email):
    txt = f"{normalize_text(org)} {email_domain(email)}".lower()
    # Government
    if any(k in txt for k in [
        "@go.ke", "ministry", "commission", "state dept", "department",
        "county", "authority", "service", "bureau", "regulator"
    ]):
        return "Government"
    # Dev Partner / Multilateral
    if any(k in txt for k in [
        "@fao.org","@wfp.org","@agra.org","@giz.de","@worldbank.org","world bank",
        "@ifad.org","@undp.org","@unicef.org","@who.int","@ilo.org","@cgiar.org",
        "akademiya2063"
    ]):
        return "Dev Partner/Multilateral"
    # Academia/Research
    if any(k in txt for k in [
        "university","@strathmore.edu","@uonbi.ac.ke","@nairobi.ac.ke","@ku.ac.ke",
        "institute","research","sbs","tegemeo","akademiya2063"
    ]):
        return "Academia/Research"
    # NGO/Nonprofit
    if any(k in txt for k in [
        "foundation","ngo","mercycorps","digitalgreen","wofaak","hope for refugee",
        "cabi","caritas","solidaridad","save the children","oxfam","care"
    ]):
        return "NGO/Nonprofit"
    # Otherwise
    return "Private Sector"

def bytes_xlsx(sheets: dict):
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        for name, d in sheets.items():
            d.to_excel(writer, index=False, sheet_name=(name[:31] or "Sheet1"))
    bio.seek(0)
    return bio

# Keyword dictionaries (extend freely)
THEME_MAP = {
    "Access": ["access", "accessible", "inaccess", "findable", "visibility", "opacity"],
    "Quality/Accuracy": ["quality", "accur", "authentic", "verifi", "bias", "errone", "clean", "reliab"],
    "Interoperability": ["interoperab", "format", "integrat", "standard", "harmonis", "schema"],
    "Timeliness": ["timeliness", "outdated", "late", "real-time", "realtime"],
    "Fragmentation/Silos": ["fragment", "silo", "scattered", "isolat"],
    "Connectivity/Infra": ["connectiv", "internet", "bandwidth", "infrastruct"],
    "Trust/Privacy/Governance": ["privacy", "trust", "consent", "share data", "governance", "ethical"],
    "Availability/Coverage": ["availab", "insufficient", "inadequ", "unavailab", "patchy", "incomplete"],
    "Standardization": ["standard", "metadata", "geo-refer", "georef", "taxonomy"],
    "Finance/Cost": ["fund", "cost", "finance", "budget"],
}

USECASE_MAP = {
    "Market Linkages & Prices": ["market", "buyer", "price", "demand", "supply", "export", "auction"],
    "Policy & Program Design": ["policy", "program", "track implementation", "impact assessment", "evidence"],
    "Financing/Credit/Insurance": ["loan", "credit", "insurance", "underwriting", "actuar", "churn", "approval"],
    "Logistics & Supply Chain": ["logistic", "cold chain", "inventory", "transport", "aggregation"],
    "Production Planning": ["plan", "planting", "harvest", "production", "variety", "yields", "soil", "weather"],
    "PH Loss & Storage": ["post-harvest", "post harvest", "loss", "storage"],
    "Traceability/Safety/Disease": ["traceab", "health", "disease", "movement"],
    "Advisory & Extension": ["advisory", "extension", "farmer decision", "recommendation"],
    "Research & Monitoring": ["thesis", "research", "monitor", "evaluation", "learning"],
    "Investment & Strategy": ["invest", "roi", "capital", "business case"],
    "Risk & Resilience": ["risk", "shock", "volatility", "drought"],
    "Fortification/Nutrition": ["fortification", "nutrition"],
    "Irrigation/Water Mgmt": ["irrigation", "water"],
    "Seed Systems": ["seed system", "seed"],
    "Livestock": ["livestock", "red meat", "fodder", "slaughter"],
}

def find_tags(text, mapping):
    txt = normalize_text(text).lower()
    tags = set()
    for tag, kws in mapping.items():
        for k in kws:
            if k in txt:
                tags.add(tag)
                break
    return sorted(tags)

def topic_model(texts, n_topics=5, max_features=4000):
    docs = [normalize_text(t) for t in texts if isinstance(t, str) and normalize_text(t)]
    if len(docs) < 6:
        return None, None, None
    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.9, min_df=2, max_features=max_features)
    X = vectorizer.fit_transform(docs)
    n_topics = max(2, min(n_topics, X.shape[0]-1))
    nmf = NMF(n_components=n_topics, random_state=42, init="nndsvd", max_iter=600)
    W = nmf.fit_transform(X)
    H = nmf.components_
    terms = np.array(vectorizer.get_feature_names_out())
    topic_terms = []
    for k in range(n_topics):
        top = terms[H[k].argsort()[::-1][:10]]
        topic_terms.append(", ".join(top))
    doc_topic = W.argmax(axis=1)
    return topic_terms, doc_topic, W

def tfidf_top_terms(texts, n=15):
    docs = [normalize_text(t) for t in texts if isinstance(t, str) and normalize_text(t)]
    if len(docs) < 5: 
        return []
    v = TfidfVectorizer(stop_words="english", max_df=0.9, min_df=2)
    X = v.fit_transform(docs)
    scores = np.asarray(X.sum(axis=0)).ravel()
    terms = np.array(v.get_feature_names_out())
    order = scores.argsort()[::-1][:n]
    return terms[order].tolist()

# ---------- NEW: Auto-label helper for topics ----------
def auto_label(topic_terms: str, kind: str) -> str:
    """
    Returns a short human label for a comma-separated topic_terms string.
    kind ∈ {"challenge","usecase"} just biases the defaults.
    """
    txt = topic_terms.lower()

    # Generic detectors
    if any(w in txt for w in ["interoperab", "standard", "schema", "harmonis"]):
        return "Interoperability & standards"
    if any(w in txt for w in ["timeliness", "timely", "outdated", "delay", "realtime", "real-time"]):
        return "Timeliness"
    if any(w in txt for w in ["quality", "accur", "reliab", "verifi", "inconsist"]):
        return "Data quality & reliability"
    if "access" in txt:
        return "Access"
    if any(w in txt for w in ["availability", "unavail", "coverage", "insufficient", "patchy"]):
        return "Availability/coverage"

    # Use-case leaning
    if any(w in txt for w in ["market", "price", "demand", "buyer", "trend", "auction"]):
        return "Market linkages, prices & trends"
    if any(w in txt for w in ["decision", "advis", "service", "recommend"]):
        return "Decision support & advisory"
    if any(w in txt for w in ["production", "harvest", "yields", "variety", "planning"]):
        return "Production planning"
    if any(w in txt for w in ["post-harvest", "post harvest", "loss", "storage", "cost"]):
        return "Post-harvest loss & costs"
    if any(w in txt for w in ["traceab", "safety", "disease", "movement"]):
        return "Traceability & health"
    if any(w in txt for w in ["risk", "shock", "drought", "weather"]):
        return "Risk & resilience"
    if any(w in txt for w in ["loan", "credit", "insur", "finance"]):
        return "Financing/insurance"

    # Fallbacks
    return "General theme" if kind == "challenge" else "General use-case"

# =============================
# App Header + Audience Mode
# =============================
st.title("🌾 Unlocking Private Sector Data Gaps — Webinar Insights Dashboard")

# Audience Mode hides host-only controls and shows explainer text for viewers
audience_mode = st.sidebar.toggle("🎭 Audience Mode", value=True, help="Turn OFF to reveal Host Controls (file upload & mapping).")

if audience_mode:
    st.markdown(
        """
**What you’re seeing:** This dashboard summarizes the **actual registration responses** for today’s webinar.  
It highlights (1) the **main challenges** participants face with agricultural data and (2) **where agri-market data can help**.  
We use simple text analysis to group open-ended answers into clear themes for discussion.
        """
    )
else:
    st.caption("Host View: load data and configure mappings here.")

# ----- Host Controls (hidden during Audience Mode) -----
with st.sidebar.expander("🛠️ Host Controls", expanded=not audience_mode):
    up = st.file_uploader("Input data (CSV or Excel)", type=["csv","xlsx","xls"], help="Upload the registration export.")
    host_has_file = up is not None

# If no file uploaded
if 'up' not in locals() or up is None:
    if not audience_mode:
        st.info("Upload the file you received/exported (CSV/XLSX) to proceed.")
        st.stop()
    else:
        st.warning("Waiting for data… (host will load the registration file).")
        st.stop()

df = load_df(up)

# ---------- Auto map columns ----------
col_map = {}
for c in df.columns:
    lc = c.lower()
    if "registration time" in lc: col_map["time"] = c
    if lc.startswith("registration email"): col_map["email"] = c
    if lc.startswith("registration first"): col_map["fname"] = c
    if lc.startswith("registration last"): col_map["lname"] = c
    if "organization" in lc: col_map["org"] = c
    if "job title" in lc: col_map["title"] = c
    if "what challenges" in lc: col_map["challenges"] = c
    if "where could agri-market data" in lc or "operations or project work" in lc:
        col_map["usecase"] = c
    if "registration status" in lc: col_map["status"] = c

# ---------- Manual mapping fallback (only visible in Host View) ----------
expected_keys = ["time","email","fname","lname","org","title","status","challenges","usecase"]
missing_any = any(k not in col_map for k in expected_keys)

if missing_any and audience_mode:
    st.warning("Host note: some expected columns weren’t auto-detected. Switch OFF Audience Mode to map them.")
elif missing_any and not audience_mode:
    st.warning("Some expected columns weren’t auto-detected. Map them below.")
    cols = df.columns.tolist()
    def pick(lbl, key):
        default = col_map.get(key)
        return st.selectbox(lbl, ["(none)"] + cols, index=(cols.index(default)+1 if default in cols else 0))
    col_map["time"]       = pick("Registration Time", "time")
    col_map["email"]      = pick("Registration Email", "email")
    col_map["fname"]      = pick("First Name", "fname")
    col_map["lname"]      = pick("Last Name", "lname")
    col_map["org"]        = pick("Organization", "org")
    col_map["title"]      = pick("Job Title", "title")
    col_map["status"]     = pick("Registration Status", "status")
    col_map["challenges"] = pick("Q1: Challenges", "challenges")
    col_map["usecase"]    = pick("Q2: Where market data helps", "usecase")

# ---------- Normalize ----------
out = pd.DataFrame({
    "time": coerce_datetime(df.get(col_map.get("time"), pd.Series([None]*len(df)))),
    "email": df.get(col_map.get("email"), ""),
    "first_name": df.get(col_map.get("fname"), ""),
    "last_name": df.get(col_map.get("lname"), ""),
    "org": df.get(col_map.get("org"), ""),
    "job_title": df.get(col_map.get("title"), ""),
    "status": df.get(col_map.get("status"), ""),
    "challenges_raw": df.get(col_map.get("challenges"), ""),
    "usecase_raw": df.get(col_map.get("usecase"), ""),
})
out["email"] = out["email"].map(clean_email)
out["domain"] = out["email"].map(email_domain)
out["sector"] = [infer_sector(o, e) for o, e in zip(out["org"], out["email"])]
out["name"] = (out["first_name"].fillna("").astype(str).str.strip()+" "+out["last_name"].fillna("").astype(str).str.strip()).str.strip()
out["challenges_raw"] = out["challenges_raw"].astype(str)
out["usecase_raw"] = out["usecase_raw"].astype(str)

# Deduplicate by email (keep latest by time)
out = out.sort_values("time").drop_duplicates(subset=["email"], keep="last").reset_index(drop=True)

# Tag themes
out["challenge_tags"] = out["challenges_raw"].apply(lambda x: find_tags(x, THEME_MAP))
out["usecase_tags"]   = out["usecase_raw"].apply(lambda x: find_tags(x, USECASE_MAP))

def explode_tags(series):
    return series.explode().dropna().replace("", np.nan).dropna()

challenge_exploded = explode_tags(out["challenge_tags"])
usecase_exploded   = explode_tags(out["usecase_tags"])

# =============================
# Sidebar Filters
# =============================
st.sidebar.header("Filters")
st.sidebar.markdown("**🔎 Filter responses**")

# ---- Sector checkboxes (always visible) ----
sectors = sorted([s for s in out["sector"].dropna().unique().tolist()])

# Keep checkbox state across reruns and sync if the sector list changes
if "sector_checks" not in st.session_state:
    st.session_state["sector_checks"] = {s: True for s in sectors}
else:
    for s in sectors:
        st.session_state["sector_checks"].setdefault(s, True)
    for s in list(st.session_state["sector_checks"].keys()):
        if s not in sectors:
            st.session_state["sector_checks"].pop(s, None)

cA, cB = st.sidebar.columns(2)
with cA:
    if st.button("Select all"):
        for s in sectors: st.session_state["sector_checks"][s] = True
with cB:
    if st.button("Clear all"):
        for s in sectors: st.session_state["sector_checks"][s] = False

selected_sectors = []
for s in sectors:
    checked = st.sidebar.checkbox(s, value=st.session_state["sector_checks"][s], key=f"sector_{s}")
    st.session_state["sector_checks"][s] = checked
    if checked:
        selected_sectors.append(s)

# ---- Date range slider ----
date_min = out["time"].min()
date_max = out["time"].max()
if pd.notna(date_min) and pd.notna(date_max):
    start, end = st.sidebar.slider(
        "Registration window",
        value=(date_min.to_pydatetime(), date_max.to_pydatetime()),
        min_value=date_min.to_pydatetime(),
        max_value=date_max.to_pydatetime(),
    )
else:
    start, end = None, None

# ---- Apply filters ----
mask = pd.Series(True, index=out.index)
if selected_sectors:
    mask &= out["sector"].isin(selected_sectors)
if start and end:
    mask &= out["time"].between(start, end)
f = out[mask].copy()

# =============================
# KPIs
# =============================
st.subheader("At a glance")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Responses", f"{len(f):,}")
c2.metric("Unique organizations", f"{f['org'].astype(str).str.strip().replace('', np.nan).nunique():,}")
c3.metric("Sectors", f"{f['sector'].nunique():,}")
if pd.notna(f["time"]).any():
    c4.metric("Last registration", f["time"].max().strftime("%Y-%m-%d %H:%M"))

# ----- Audience explainer & legend -----
with st.expander("ℹ️ About this analysis (for attendees)", expanded=True if audience_mode else False):
    st.markdown(
        """
**How to read this dashboard**
- **Who Registered & From Which Sectors**: who signed up and from which sector.
- **What Challenges Are Participants Facing?**: grouped themes from *“What challenges do you currently face…?”*
- **Where Agri-Market Data Could Help**: grouped themes from *“Where could agri-market data make a difference…?”*
- **Emerging Topics & Keywords**: unsupervised clusters (NMF) and frequent terms to surface talking points.

**Theme Legend — Challenges**
*Access · Quality/Accuracy · Interoperability · Timeliness · Fragmentation/Silos · Connectivity/Infra · Trust/Privacy/Governance · Availability/Coverage · Standardization · Finance/Cost*

**Theme Legend — Use-cases**
*Market Linkages & Prices · Policy & Program Design · Financing/Credit/Insurance · Logistics & Supply Chain · Production Planning · PH Loss & Storage · Traceability/Safety/Disease · Advisory & Extension · Research & Monitoring · Investment & Strategy · Risk & Resilience · Fortification/Nutrition · Irrigation/Water Mgmt · Seed Systems · Livestock*

*Note:* Free-text answers are grouped by keywords; counts reflect how often a theme appears across responses.
        """
    )

# =============================
# Trends & Composition
# =============================
st.subheader("Who Registered & From Which Sectors")
colA, colB = st.columns(2)

if f["time"].notna().any():
    ts = f.dropna(subset=["time"]).copy()
    ts["date"] = ts["time"].dt.date
    daily = ts.groupby("date", as_index=False).size()
    line = alt.Chart(daily).mark_line(point=True).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("size:Q", title="Registrations"),
        tooltip=["date:T","size:Q"]
    ).properties(height=280)
    colA.altair_chart(line, use_container_width=True)
else:
    colA.info("No valid timestamps found.")

sector_counts = f["sector"].value_counts().reset_index()
sector_counts.columns = ["sector","count"]
if not sector_counts.empty:
    pie = alt.Chart(sector_counts).mark_arc(innerRadius=60).encode(
        theta="count:Q",
        color=alt.Color("sector:N", legend=None),
        tooltip=["sector","count"]
    ).properties(height=280)
    colB.altair_chart(pie, use_container_width=True)
else:
    colB.info("No sectors to display.")

# Top orgs
top_orgs = (
    f["org"].astype(str).str.strip()
    .replace("", np.nan).dropna()
    .value_counts().head(15).reset_index()
)
top_orgs.columns = ["organization","count"]
if not top_orgs.empty:
    st.altair_chart(
        alt.Chart(top_orgs).mark_bar().encode(
            x=alt.X("count:Q", title="Count"),
            y=alt.Y("organization:N", sort='-x', title=None),
            tooltip=["organization","count"]
        ).properties(height=350),
        use_container_width=True
    )
else:
    st.info("No organizations to display.")

# =============================
# Challenges (themes)
# =============================
st.subheader("What Challenges Are Participants Facing?")

ch_freq = challenge_exploded.value_counts().reset_index()
ch_freq.columns = ["theme","count"]

col1, col2 = st.columns(2)
if not ch_freq.empty:
    col1.altair_chart(
        alt.Chart(ch_freq).mark_bar().encode(
            x=alt.X("count:Q", title="Mentions"),
            y=alt.Y("theme:N", sort='-x', title=None),
            tooltip=["theme","count"]
        ).properties(height=350),
        use_container_width=True
    )
else:
    col1.info("No challenge themes detected.")

hm = (
    f.explode("challenge_tags")
     .dropna(subset=["challenge_tags"])
     .groupby(["sector","challenge_tags"], as_index=False)
     .size()
)
if not hm.empty:
    col2.altair_chart(
        alt.Chart(hm).mark_rect().encode(
            x=alt.X("sector:N", title=None),
            y=alt.Y("challenge_tags:N", title=None),
            color=alt.Color("size:Q", title="Count"),
            tooltip=["sector","challenge_tags","size"]
        ).properties(height=350),
        use_container_width=True
    )
else:
    col2.info("No challenge tags after filtering.")

with st.expander("🔎 Browse challenge responses"):
    st.dataframe(f[["name","org","sector","job_title","challenges_raw"]].sort_values("org"), use_container_width=True)

# =============================
# Use-cases (themes)
# =============================
st.subheader("Where Agri-Market Data Could Help")

uc_freq = usecase_exploded.value_counts().reset_index()
uc_freq.columns = ["use_case","count"]

col3, col4 = st.columns(2)
if not uc_freq.empty:
    col3.altair_chart(
        alt.Chart(uc_freq).mark_bar().encode(
            x=alt.X("count:Q", title="Mentions"),
            y=alt.Y("use_case:N", sort='-x', title=None),
            tooltip=["use_case","count"]
        ).properties(height=350),
        use_container_width=True
    )
else:
    col3.info("No use-case themes detected.")

uc_hm = (
    f.explode("usecase_tags")
     .dropna(subset=["usecase_tags"])
     .groupby(["sector","usecase_tags"], as_index=False)
     .size()
)
if not uc_hm.empty:
    col4.altair_chart(
        alt.Chart(uc_hm).mark_rect().encode(
            x=alt.X("sector:N", title=None),
            y=alt.Y("usecase_tags:N", title=None),
            color=alt.Color("size:Q", title="Count"),
            tooltip=["sector","usecase_tags","size"]
        ).properties(height=350),
        use_container_width=True
    )
else:
    col4.info("No use-case tags after filtering.")

with st.expander("🔎 Browse use-case responses"):
    st.dataframe(f[["name","org","sector","job_title","usecase_raw"]].sort_values("org"), use_container_width=True)

# =============================
# Topic discovery + Keywords
# =============================
st.subheader("Emerging Topics From Open-Ended Responses")
left, right = st.columns(2)

ch_topics, ch_doc_topic, _ = topic_model(f["challenges_raw"].tolist(), n_topics=5)
if ch_topics:
    left.markdown("**Challenges — Top topics**")
    for i, t in enumerate(ch_topics, 1):
        label = auto_label(t, "challenge")
        left.write(f"**T{i} ({label}):** {t}")
else:
    left.info("Not enough challenge text to build topics.")

uc_topics, uc_doc_topic, _ = topic_model(f["usecase_raw"].tolist(), n_topics=5)
if uc_topics:
    right.markdown("**Use-cases — Top topics**")
    for i, t in enumerate(uc_topics, 1):
        label = auto_label(t, "usecase")
        right.write(f"**T{i} ({label}):** {t}")
else:
    right.info("Not enough use-case text to build topics.")

st.subheader("Top keywords")
kw_ch = tfidf_top_terms(f["challenges_raw"].tolist(), 15)
kw_uc = tfidf_top_terms(f["usecase_raw"].tolist(), 15)
colk1, colk2 = st.columns(2)
colk1.write("**Challenges – keywords:** " + (", ".join(kw_ch) if kw_ch else "Not enough text"))
colk2.write("**Use-cases – keywords:** " + (", ".join(kw_uc) if kw_uc else "Not enough text"))

# =============================
# Talking Points
# =============================
st.subheader("🎤 Quick notes")

def topn(df_counts, n=5):
    return ", ".join([f"{a} ({int(b)})" for a,b in df_counts.head(n).values])

points = []
if not ch_freq.empty:
    points.append(f"Top challenge themes: {topn(ch_freq, 5)}.")
if not uc_freq.empty:
    points.append(f"Top use-case themes: {topn(uc_freq, 5)}.")
if 'sector_counts' in locals() and not sector_counts.empty:
    lead_sector = sector_counts.sort_values('count', ascending=False).iloc[0]
    points.append(f"Most represented sector: {lead_sector['sector']} ({int(lead_sector['count'])}).")
if 'top_orgs' in locals() and not top_orgs.empty:
    points.append(f"Top participating org (by sign-ups): {top_orgs.iloc[0]['organization']} ({int(top_orgs.iloc[0]['count'])}).")
if f['time'].notna().any():
    points.append(f"Latest registration: {f['time'].max().strftime('%b %d, %Y %H:%M')}.")

for i, p in enumerate(points, 1):
    st.write(f"**{i}. {p}**")

# =============================
# Export (filtered views + summaries)
# =============================
st.subheader("⬇️ Export")
summary_tabs = {
    "registrations_filtered": f,
    "challenge_theme_counts": ch_freq,
    "usecase_theme_counts": uc_freq,
}
if not hm.empty:    summary_tabs["challenge_theme_by_sector"] = hm
if not uc_hm.empty: summary_tabs["usecase_theme_by_sector"] = uc_hm

bio = bytes_xlsx(summary_tabs)
st.download_button(
    "Download Excel summary",
    bio,
    file_name="safic_webinar_summary.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)