import argparse
import html
import json
import math
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from scipy import stats

try:
    import statsmodels.api as sm
except Exception as e:
    raise ImportError(
        "This script requires statsmodels for logistic regression. "
        "Install it with: python -m pip install statsmodels"
    ) from e


DEFAULT_OP_PATH = "/data7/chenyitong/Winning_Arguments/2262_unique_ops.json"
DEFAULT_PAIR_PATH = "/data7/chenyitong/Winning_Arguments/single_turn_pairs.json"
DEFAULT_OUT_DIR = "/data7/chenyitong/Winning_Arguments/final_new_Experiment_Analysis_Results/human_root_reply_41_feature_logistic_regression_vad_proxy_fast"


URL_RE = re.compile(r"\b(?:https?://|www\.)[^\s\]\)>\"]+", re.I)
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+")
QUOTE_BLOCK_RE = re.compile(r"(?m)^(?:\s*&gt;|\s*>).*$")
SENTENCE_RE = re.compile(r"[.!?]+")
EXAMPLE_RE = re.compile(r"\b(for example|for instance|e\.g\.|such as|to illustrate|as an example)\b", re.I)
EDIT_RE = re.compile(r"(?is)(?:^|\n)\s*(edit|update)\s*[:：].*?$")
MOD_FOOTNOTE_RE = re.compile(r"(?is)\n\s*_{3,}\s*\n\s*(?:&gt;|>)?\s*\*?\s*hello,\s+users\s+of\s+cmv!.*$")


BASE_STOPWORDS = {
    "a","about","above","after","again","against","all","am","an","and","any","are","aren't","as","at",
    "be","because","been","before","being","below","between","both","but","by","can't","cannot","could",
    "couldn't","did","didn't","do","does","doesn't","doing","don't","down","during","each","few","for",
    "from","further","had","hadn't","has","hasn't","have","haven't","having","he","he'd","he'll","he's",
    "her","here","here's","hers","herself","him","himself","his","how","how's","i","i'd","i'll","i'm",
    "i've","if","in","into","is","isn't","it","it's","its","itself","let's","me","more","most","mustn't",
    "my","myself","no","nor","not","of","off","on","once","only","or","other","ought","our","ours",
    "ourselves","out","over","own","same","shan't","she","she'd","she'll","she's","should","shouldn't",
    "so","some","such","than","that","that's","the","their","theirs","them","themselves","then","there",
    "there's","these","they","they'd","they'll","they're","they've","this","those","through","to","too",
    "under","until","up","very","was","wasn't","we","we'd","we'll","we're","we've","were","weren't",
    "what","what's","when","when's","where","where's","which","while","who","who's","whom","why",
    "why's","with","won't","would","wouldn't","you","you'd","you'll","you're","you've","your","yours",
    "yourself","yourselves"
}

POSITIVE_FALLBACK = {
    "good","great","better","best","right","correct","true","reasonable","useful","helpful","happy","love",
    "like","important","benefit","beneficial","positive","success","successful","safe","fair","clear","clearly",
    "agree","valid","valuable","strong","stronger","possible","improve","improved","improvement","nice",
    "excellent","effective","efficient","support","supported","advantage","advantages","gain","gains"
}

NEGATIVE_FALLBACK = {
    "bad","worse","worst","wrong","false","terrible","awful","hate","harm","harmful","negative","danger",
    "dangerous","problem","problems","issue","issues","risk","risks","risky","fail","failure","failed",
    "weak","weaker","poor","poorly","unfair","unclear","invalid","impossible","damage","damaging",
    "loss","lose","losing","hurt","hurts","stupid","sinister","cynical","hollow"
}

HEDGE_FALLBACK = {
    "maybe","perhaps","possibly","probably","likely","unlikely","might","may","could","can","would",
    "should","seem","seems","seemed","seeming","appear","appears","appeared","apparently","arguably",
    "roughly","about","around","almost","somewhat","generally","usually","sometimes","often","mostly",
    "partly","potentially","presumably","relatively","fairly","rather","sort","kind","estimate","estimated"
}

FIRST_PERSON = {"i","me","my","mine","myself","i'm","i've","i'd","i'll"}
FIRST_PERSON_PLURAL = {"we","us","our","ours","ourselves","we're","we've","we'd","we'll"}
SECOND_PERSON = {"you","your","yours","yourself","yourselves","you're","you've","you'd","you'll"}
DEFINITE_ARTICLES = {"the"}
INDEFINITE_ARTICLES = {"a", "an"}
NUMBERED_WORDS = {
    "first","second","third","fourth","fifth","sixth","seventh","eighth","ninth","tenth",
    "firstly","secondly","thirdly","fourthly","fifthly","lastly","finally"
}

FEATURE_SPECS = [
    ("reply_frac_in_all", "reply frac. in all", "interplay", "down", 0.0001),
    ("reply_frac_in_content", "reply frac. in content", "interplay", "down", 0.0001),
    ("op_frac_in_stopwords", "OP frac. in stopwords", "interplay", "up", 0.0001),
    ("common_in_stopwords", "#common in stopwords", "interplay", "up", 0.0001),
    ("reply_frac_in_stopwords", "reply frac. in stopwords", "interplay", "down", 0.0001),
    ("op_frac_in_all", "OP frac. in all", "interplay", "up", 0.0001),
    # 移除 common_in_all 以解决完全多重共线性问题 (它等于 common_in_stopwords + common_in_content)
    # ("common_in_all", "#common in all", "interplay", "up", 0.0001),
    ("jaccard_in_content", "Jaccard in content", "interplay", "down", 0.0001),
    ("jaccard_in_stopwords", "Jaccard in stopwords", "interplay", "up", 0.0001),
    ("common_in_content", "#common in content", "interplay", "up", 0.0001),
    ("op_frac_in_content", "OP frac. in content", "interplay", "up", 0.05),
    ("jaccard_in_all", "Jaccard in all", "interplay", "down", 0.05),

    ("num_words", "#words", "argument_only", "up", 0.0001),
    ("num_definite_articles", "#definite articles", "argument_only", "up", 0.0001),
    ("num_indefinite_articles", "#indefinite articles", "argument_only", "up", 0.0001),
    ("num_positive_words", "#positive words", "argument_only", "up", 0.0001),
    ("num_2nd_person_pronouns", "#2nd person pronoun", "argument_only", "up", 0.0001),
    ("num_links", "#links", "argument_only", "up", 0.0001),
    ("num_negative_words", "#negative words", "argument_only", "up", 0.0001),
    ("num_hedges", "#hedges", "argument_only", "up", 0.0001),
    ("num_1st_person_pronouns", "#1st person pronouns", "argument_only", "up", 0.0001),
    ("num_1st_person_plural_pronouns", "#1st person plural pronoun", "argument_only", "up", 0.0001),
    ("num_dotcom_links", "#.com links", "argument_only", "up", 0.0001),
    ("frac_links", "frac. links", "argument_only", "up", 0.0001),
    ("frac_dotcom_links", "frac. .com links", "argument_only", "up", 0.0001),
    ("num_examples", "#examples", "argument_only", "up", 0.05),
    ("frac_definite_articles", "frac. definite articles", "argument_only", "up", 0.05),
    ("num_question_marks", "#question marks", "argument_only", "up", 0.05),
    ("num_pdf_links", "#PDF links", "argument_only", "up", 0.05),
    ("frac_positive_words", "frac. positive words", "argument_only", "down", 0.05),
    ("arousal", "arousal", "argument_only", "down", 0.05),
    ("valence", "valence", "argument_only", "down", 0.05),
    ("word_entropy", "word entropy", "argument_only", "up", 0.0001),
    ("num_sentences", "#sentences", "argument_only", "up", 0.0001),
    ("type_token_ratio", "type-token ratio", "argument_only", "down", 0.0001),
    ("num_paragraphs", "#paragraphs", "argument_only", "up", 0.0001),
    ("num_italics", "#italics", "argument_only", "up", 0.0001),
    ("bullet_list", "bullet list", "argument_only", "up", 0.0001),
    ("num_bolds", "#bolds", "argument_only", "up", 0.01),
    ("numbered_words", "numbered words", "argument_only", "up", 0.05),
    ("frac_italics", "frac. italics", "argument_only", "up", 0.05),
]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_records(raw):
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        if "data" in raw and isinstance(raw["data"], list):
            return raw["data"]
        if "items" in raw and isinstance(raw["items"], list):
            return raw["items"]
        if all(isinstance(v, dict) for v in raw.values()):
            return list(raw.values())
    raise ValueError("Unsupported JSON structure")


def load_word_set(path, fallback):
    if not path:
        return set(fallback)
    words = set()
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip().lower()
            if not line or line.startswith(";") or line.startswith("#"):
                continue
            words.add(line.split()[0])
    return words or set(fallback)


def find_col(columns, candidates):
    lower = {str(c).lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    for c in columns:
        cl = str(c).lower()
        for cand in candidates:
            if cand.lower() in cl:
                return c
    return None


def load_liwc_categories(path):
    if not path:
        return None, None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    categories = {}
    words = {}
    section = 0
    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line == "%":
                section += 1
                continue
            parts = line.split()
            if section == 1 and len(parts) >= 2:
                categories[parts[0]] = parts[1].lower()
            elif section >= 2 and len(parts) >= 2:
                pattern = parts[0].lower()
                cat_names = {categories.get(cid, "").lower() for cid in parts[1:]}
                if "posemo" in cat_names or "positive emotion" in cat_names or "positive" in cat_names:
                    words.setdefault(pattern, set()).add("posemo")
                if "negemo" in cat_names or "negative emotion" in cat_names or "negative" in cat_names:
                    words.setdefault(pattern, set()).add("negemo")

    pos, neg = set(), set()
    for pattern, cats in words.items():
        if pattern.endswith("*"):
            stem = pattern[:-1]
            if "posemo" in cats:
                pos.add(stem + "*")
            if "negemo" in cats:
                neg.add(stem + "*")
        else:
            if "posemo" in cats:
                pos.add(pattern)
            if "negemo" in cats:
                neg.add(pattern)
    return pos, neg


def split_lexicon(lexicon):
    exact = set()
    stems = []
    for item in lexicon or []:
        item = str(item).lower().strip()
        if not item:
            continue
        if item.endswith("*"):
            stem = item[:-1]
            if stem:
                stems.append(stem)
        else:
            exact.add(item)
    return exact, tuple(stems)


def count_lexicon_fast(tokens, lexicon_exact, lexicon_stems=None):
    if not tokens:
        return 0
    counts = Counter(tokens)
    total = sum(counts[t] for t in lexicon_exact if t in counts)

    if lexicon_stems:
        for token, c in counts.items():
            if token in lexicon_exact:
                continue
            for stem in lexicon_stems:
                if token.startswith(stem):
                    total += c
                    break
    return int(total)


def normalize_url(url):
    if not url:
        return ""
    u = html.unescape(str(url)).strip()
    return u.rstrip(".,;:!?)]}>'\"")


def parsed_url(url):
    u = normalize_url(url)
    if u.lower().startswith("www."):
        u = "http://" + u
    return urlparse(u)


def is_dotcom_url(url):
    netloc = parsed_url(url).netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc.endswith(".com")


def is_pdf_url(url):
    return parsed_url(url).path.lower().endswith(".pdf")


def _detect_scale_and_unit(series):
    x = pd.to_numeric(series, errors="coerce")
    xmin = x.min(skipna=True)
    xmax = x.max(skipna=True)

    if pd.isna(xmin) or pd.isna(xmax) or xmax == xmin:
        return x, "unknown"

    if xmin >= 0 and xmax <= 1:
        return x, "0_to_1"

    if xmin >= -1 and xmax <= 1:
        return (x + 1.0) / 2.0, "-1_to_1"

    if xmin >= 1 and xmax <= 9:
        return (x - 1.0) / 8.0, "1_to_9"

    return (x - xmin) / (xmax - xmin), "minmax"


def _polar_masks(raw_series, unit_series, scale_name, threshold):
    raw = pd.to_numeric(raw_series, errors="coerce")
    unit = pd.to_numeric(unit_series, errors="coerce")

    if scale_name == "-1_to_1":
        pos = raw >= threshold
        neg = raw <= -threshold
    else:
        pos_threshold = (threshold + 1.0) / 2.0
        neg_threshold = (1.0 - threshold) / 2.0
        pos = unit >= pos_threshold
        neg = unit <= neg_threshold

    return pos.fillna(False), neg.fillna(False)


def load_vad_lexicon_and_polar(path, polar_threshold=0.333, unigrams_only=True):
    if not path:
        return {}, set(), set(), "missing"

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    sep = "\t" if p.suffix.lower() in {".tsv", ".txt"} else ","
    df = pd.read_csv(p, sep=sep)

    word_col = find_col(df.columns, ["word", "term", "Words"])
    val_col = find_col(df.columns, ["valence", "valence_mean", "V.Mean.Sum", "V.Mean"])
    aro_col = find_col(df.columns, ["arousal", "arousal_mean", "A.Mean.Sum", "A.Mean"])

    if word_col is None or val_col is None or aro_col is None:
        raise ValueError(f"VAD lexicon must contain word/term, valence, and arousal columns. Existing columns: {list(df.columns)}")

    terms = df[word_col].astype(str).str.lower().str.strip()
    val_unit, val_scale = _detect_scale_and_unit(df[val_col])
    aro_unit, aro_scale = _detect_scale_and_unit(df[aro_col])

    tmp = pd.DataFrame({
        "word": terms,
        "valence_raw": pd.to_numeric(df[val_col], errors="coerce"),
        "arousal_raw": pd.to_numeric(df[aro_col], errors="coerce"),
        "valence": val_unit,
        "arousal": aro_unit,
    }).dropna(subset=["word", "valence", "arousal"])

    tmp = tmp[tmp["word"] != ""]
    if unigrams_only:
        tmp = tmp[~tmp["word"].str.contains(r"\s+", regex=True)]

    tmp = tmp.drop_duplicates(subset=["word"], keep="first")

    vad = {
        row.word: {"valence": float(row.valence), "arousal": float(row.arousal)}
        for row in tmp.itertuples(index=False)
    }

    pos_mask, neg_mask = _polar_masks(tmp["valence_raw"], tmp["valence"], val_scale, polar_threshold)
    pos_words = set(tmp.loc[pos_mask, "word"])
    neg_words = set(tmp.loc[neg_mask, "word"])

    scale_report = f"valence_scale={val_scale}; arousal_scale={aro_scale}; polar_threshold={polar_threshold}; unigrams_only={unigrams_only}"
    return vad, pos_words, neg_words, scale_report


def clean_text(text):
    if text is None:
        return ""
    text = html.unescape(str(text))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = MOD_FOOTNOTE_RE.sub("", text)
    text = EDIT_RE.sub("", text)
    return text.strip()


def text_for_tokens(text):
    text = clean_text(text)
    text = QUOTE_BLOCK_RE.sub(" QUOTETOKEN ", text)
    text = URL_RE.sub(" URLTOKEN ", text)
    return text


def tokenize(text):
    return [m.group(0).lower() for m in WORD_RE.finditer(text_for_tokens(text))]


def token_sets(text, stopwords):
    toks = tokenize(text)
    all_set = set(toks)
    stop_set = {t for t in all_set if t in stopwords}
    content_set = all_set - stop_set
    return all_set, stop_set, content_set


def safe_div(a, b):
    return float(a) / float(b) if b else np.nan


def jaccard(a, b):
    u = len(a | b)
    return float(len(a & b)) / float(u) if u else np.nan


def count_regex(pattern, text):
    return len(pattern.findall(text))


def count_markdown_bolds(text):
    return len(re.findall(r"(\*\*[^*\n][\s\S]*?[^*\n]\*\*|__[^_\n][\s\S]*?[^_\n]__)", text))


def count_markdown_italics(text):
    t = re.sub(r"\*\*[\s\S]*?\*\*", " ", text)
    t = re.sub(r"__[\s\S]*?__", " ", t)
    star = re.findall(r"(?<!\*)\*(?!\s|\*)([\s\S]*?)(?<!\s|\*)\*(?!\*)", t)
    under = re.findall(r"(?<!_)_(?!\s|_)([\s\S]*?)(?<!\s|_)_(?!_)", t)
    return len(star) + len(under)


def has_bullet_list(text):
    for line in clean_text(text).split("\n"):
        if re.match(r"^\s*[-*+]\s+\S+", line):
            return 1
    return 0


def count_paragraphs(text):
    blocks = [b.strip() for b in re.split(r"\n\s*\n+", clean_text(text)) if b.strip()]
    return len(blocks)


def count_sentences(text):
    parts = [p.strip() for p in SENTENCE_RE.split(clean_text(text)) if p.strip()]
    return len(parts)


def word_entropy(tokens):
    if not tokens:
        return np.nan
    c = Counter(tokens)
    n = float(sum(c.values()))
    return -sum((v / n) * math.log(v / n) for v in c.values())


def avg_word_score(tokens, vad, key, stopwords):
    if not vad or not tokens:
        return np.nan
    counts = Counter(t for t in tokens if t not in stopwords)
    numerator = 0.0
    denominator = 0
    for token, c in counts.items():
        if token in vad:
            numerator += vad[token][key] * c
            denominator += c
    return numerator / denominator if denominator else np.nan


def argument_features(text, resources):
    raw = clean_text(text)
    toks = tokenize(raw)
    n = len(toks)
    urls = [normalize_url(u) for u in URL_RE.findall(raw)]
    dotcom_urls = [u for u in urls if is_dotcom_url(u)]
    pdf_urls = [u for u in urls if is_pdf_url(u)]
    italics = count_markdown_italics(raw)
    counts = Counter(toks)

    feats = {}
    feats["num_words"] = n
    feats["num_definite_articles"] = sum(counts[w] for w in DEFINITE_ARTICLES)
    feats["num_indefinite_articles"] = sum(counts[w] for w in INDEFINITE_ARTICLES)
    feats["num_positive_words"] = count_lexicon_fast(
        toks, resources["positive_words_exact"], resources["positive_words_stems"]
    )
    feats["num_2nd_person_pronouns"] = sum(counts[w] for w in SECOND_PERSON)
    feats["num_links"] = len(urls)
    feats["num_negative_words"] = count_lexicon_fast(
        toks, resources["negative_words_exact"], resources["negative_words_stems"]
    )
    feats["num_hedges"] = sum(counts[w] for w in resources["hedges"])
    feats["num_1st_person_pronouns"] = sum(counts[w] for w in FIRST_PERSON)
    feats["num_1st_person_plural_pronouns"] = sum(counts[w] for w in FIRST_PERSON_PLURAL)
    feats["num_dotcom_links"] = len(dotcom_urls)
    feats["frac_links"] = safe_div(feats["num_links"], n)
    feats["frac_dotcom_links"] = safe_div(feats["num_dotcom_links"], n)
    feats["num_examples"] = count_regex(EXAMPLE_RE, raw)
    feats["frac_definite_articles"] = safe_div(feats["num_definite_articles"], n)
    feats["num_question_marks"] = raw.count("?")
    feats["num_pdf_links"] = len(pdf_urls)
    feats["frac_positive_words"] = safe_div(feats["num_positive_words"], n)
    feats["arousal"] = avg_word_score(toks, resources["vad"], "arousal", resources["stopwords"])
    feats["valence"] = avg_word_score(toks, resources["vad"], "valence", resources["stopwords"])
    feats["word_entropy"] = word_entropy(toks)
    feats["num_sentences"] = count_sentences(raw)
    feats["type_token_ratio"] = safe_div(len(set(toks)), n)
    feats["num_paragraphs"] = count_paragraphs(raw)
    feats["num_italics"] = italics
    feats["bullet_list"] = has_bullet_list(raw)
    feats["num_bolds"] = count_markdown_bolds(raw)
    feats["numbered_words"] = sum(counts[w] for w in NUMBERED_WORDS)
    feats["frac_italics"] = safe_div(italics, n)
    return feats


def interplay_features(reply_text, op_text, stopwords):
    a_all, a_stop, a_content = token_sets(reply_text, stopwords)
    o_all, o_stop, o_content = token_sets(op_text, stopwords)

    feats = {}
    feats["reply_frac_in_all"] = safe_div(len(a_all & o_all), len(a_all))
    feats["reply_frac_in_content"] = safe_div(len(a_content & o_content), len(a_content))
    feats["op_frac_in_stopwords"] = safe_div(len(a_stop & o_stop), len(o_stop))
    feats["common_in_stopwords"] = len(a_stop & o_stop)
    feats["reply_frac_in_stopwords"] = safe_div(len(a_stop & o_stop), len(a_stop))
    feats["op_frac_in_all"] = safe_div(len(a_all & o_all), len(o_all))
    # feats["common_in_all"] = len(a_all & o_all) # <-- 仍然在底层特征提取，但不在 FEATURE_SPECS 里了
    feats["jaccard_in_content"] = jaccard(a_content, o_content)
    feats["jaccard_in_stopwords"] = jaccard(a_stop, o_stop)
    feats["common_in_content"] = len(a_content & o_content)
    feats["op_frac_in_content"] = safe_div(len(a_content & o_content), len(o_content))
    feats["jaccard_in_all"] = jaccard(a_all, o_all)
    return feats


def all_features(reply_text, op_text, resources):
    feats = {}
    feats.update(interplay_features(reply_text, op_text, resources["stopwords"]))
    feats.update(argument_features(reply_text, resources))
    return feats


def p_to_arrows(p, direction):
    if pd.isna(p):
        return ""
    arrow = "↑" if direction == "up" else "↓"
    if p < 0.0001:
        return arrow * 4
    if p < 0.001:
        return arrow * 3
    if p < 0.01:
        return arrow * 2
    if p < 0.05:
        return arrow
    return ""


def paired_summary(df, feature_id, feature_name, group, expected_direction, paper_threshold, alpha, m):
    s = pd.to_numeric(df[f"{feature_id}_success"], errors="coerce")
    f = pd.to_numeric(df[f"{feature_id}_failure"], errors="coerce")
    tmp = pd.DataFrame({"s": s, "f": f}).dropna()
    n = len(tmp)

    if n < 2:
        return {
            "feature_id": feature_id,
            "paper_feature_name": feature_name,
            "feature_group": group,
            "n_pairs": n,
            "success_mean": np.nan,
            "failure_mean": np.nan,
            "mean_diff_success_minus_failure": np.nan,
            "sd_diff": np.nan,
            "se_diff": np.nan,
            "t_stat": np.nan,
            "p_raw": np.nan,
            "p_bonferroni": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "ci_bonferroni_low": np.nan,
            "ci_bonferroni_high": np.nan,
            "cohens_dz": np.nan,
            "observed_direction": "",
            "paper_expected_direction": expected_direction,
            "paper_p_threshold": paper_threshold,
            "observed_arrows_raw": "",
            "observed_arrows_bonferroni": "",
            "direction_matches_paper": np.nan,
            "replicates_paper_by_bonferroni_threshold": np.nan,
            "success_greater_pair_fraction": np.nan,
            "failure_greater_pair_fraction": np.nan,
            "equal_pair_fraction": np.nan,
        }

    diff = tmp["s"].to_numpy() - tmp["f"].to_numpy()
    mean_diff = float(np.mean(diff))
    sd_diff = float(np.std(diff, ddof=1))
    se_diff = float(sd_diff / math.sqrt(n)) if n > 0 else np.nan
    t_stat, p_raw = stats.ttest_rel(tmp["s"], tmp["f"], nan_policy="omit")
    p_raw = float(p_raw)
    p_bonf = float(min(p_raw * m, 1.0))

    tcrit95 = float(stats.t.ppf(1 - alpha / 2, n - 1))
    ci95_low = mean_diff - tcrit95 * se_diff
    ci95_high = mean_diff + tcrit95 * se_diff

    alpha_b = alpha / m
    tcritb = float(stats.t.ppf(1 - alpha_b / 2, n - 1))
    cib_low = mean_diff - tcritb * se_diff
    cib_high = mean_diff + tcritb * se_diff

    observed_direction = "up" if mean_diff > 0 else "down" if mean_diff < 0 else "flat"
    direction_matches = observed_direction == expected_direction
    dz = mean_diff / sd_diff if sd_diff else np.nan

    return {
        "feature_id": feature_id,
        "paper_feature_name": feature_name,
        "feature_group": group,
        "n_pairs": n,
        "success_mean": float(tmp["s"].mean()),
        "failure_mean": float(tmp["f"].mean()),
        "mean_diff_success_minus_failure": mean_diff,
        "sd_diff": sd_diff,
        "se_diff": se_diff,
        "t_stat": float(t_stat),
        "p_raw": p_raw,
        "p_bonferroni": p_bonf,
        "ci95_low": float(ci95_low),
        "ci95_high": float(ci95_high),
        "ci_bonferroni_low": float(cib_low),
        "ci_bonferroni_high": float(cib_high),
        "cohens_dz": float(dz) if not pd.isna(dz) else np.nan,
        "observed_direction": observed_direction,
        "paper_expected_direction": expected_direction,
        "paper_p_threshold": paper_threshold,
        "observed_arrows_raw": p_to_arrows(p_raw, observed_direction),
        "observed_arrows_bonferroni": p_to_arrows(p_bonf, observed_direction),
        "direction_matches_paper": bool(direction_matches),
        "replicates_paper_by_bonferroni_threshold": bool(direction_matches and p_bonf < paper_threshold),
        "success_greater_pair_fraction": float(np.mean(diff > 0)),
        "failure_greater_pair_fraction": float(np.mean(diff < 0)),
        "equal_pair_fraction": float(np.mean(diff == 0)),
    }


def build_op_map(op_path):
    raw = load_json(op_path)
    records = normalize_records(raw)
    out = {}
    for r in records:
        if not isinstance(r, dict):
            continue
        rid = r.get("id") or r.get("root") or r.get("name")
        txt = r.get("text") or r.get("body") or r.get("selftext") or ""
        if rid:
            out[str(rid)] = clean_text(txt)
    return out


def iter_pairs(pair_path):
    raw = load_json(pair_path)
    if isinstance(raw, dict):
        for k, v in raw.items():
            if isinstance(v, dict) and "success" in v and "failure" in v:
                yield str(k), v
    elif isinstance(raw, list):
        for i, v in enumerate(raw):
            if isinstance(v, dict) and "success" in v and "failure" in v:
                yield f"p_{i}", v
    else:
        raise ValueError("Unsupported pair JSON structure")


def root_id_from_pair(pair):
    s = pair.get("success", {})
    f = pair.get("failure", {})
    return s.get("root") or s.get("reply-to") or f.get("root") or f.get("reply-to")


def save_feature_level_files(wide_df, out_dir):
    feature_dir = Path(out_dir) / "feature_level_csvs"
    feature_dir.mkdir(parents=True, exist_ok=True)
    base_cols = ["pair_id", "root_id", "success_id", "failure_id"]
    for feature_id, _, _, _, _ in FEATURE_SPECS:
        cols = base_cols + [f"{feature_id}_success", f"{feature_id}_failure"]
        tmp = wide_df[cols].copy()
        tmp["difference_success_minus_failure"] = tmp[f"{feature_id}_success"] - tmp[f"{feature_id}_failure"]
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", feature_id)
        tmp.to_csv(feature_dir / f"{safe_name}.csv", index=False)

    meta = pd.DataFrame([
        {
            "feature_id": f[0],
            "paper_feature_name": f[1],
            "feature_group": f[2],
            "paper_expected_direction": f[3],
            "paper_p_threshold": f[4],
            "file": f"feature_level_csvs/{re.sub(r'[^A-Za-z0-9_.-]+', '_', f[0])}.csv",
        }
        for f in FEATURE_SPECS
    ])
    meta.to_csv(Path(out_dir) / "feature_file_index.csv", index=False)


def choose_sentiment_lexicons(args, vad_pos, vad_neg):
    liwc_pos, liwc_neg = load_liwc_categories(args.liwc_dict_path)

    source = args.sentiment_source
    if source == "auto":
        if liwc_pos is not None and liwc_neg is not None:
            return liwc_pos, liwc_neg, "liwc"
        if args.vad_lexicon and vad_pos and vad_neg:
            return vad_pos, vad_neg, f"vad_polar_threshold_{args.vad_polar_threshold}"
        return load_word_set(args.positive_words_path, POSITIVE_FALLBACK), load_word_set(args.negative_words_path, NEGATIVE_FALLBACK), "fallback_internal_approx_not_liwc"

    if source == "liwc":
        if liwc_pos is None or liwc_neg is None:
            raise ValueError("--sentiment_source liwc requires --liwc_dict_path")
        return liwc_pos, liwc_neg, "liwc"

    if source == "vad_polar":
        if not args.vad_lexicon:
            raise ValueError("--sentiment_source vad_polar requires --vad_lexicon")
        return vad_pos, vad_neg, f"vad_polar_threshold_{args.vad_polar_threshold}"

    if source == "fallback":
        return load_word_set(args.positive_words_path, POSITIVE_FALLBACK), load_word_set(args.negative_words_path, NEGATIVE_FALLBACK), "fallback_internal_approx_not_liwc"

    raise ValueError(f"Unknown sentiment_source: {source}")



def auc_score(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    mask = np.isfinite(p)
    y = y[mask]
    p = p[mask]
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = pd.Series(p).rank(method="average").to_numpy()
    rank_sum_pos = np.sum(ranks[y == 1])
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def prepare_design_data(df, feature_ids, scaler=None):
    X_raw = df[feature_ids].replace([np.inf, -np.inf], np.nan)

    if scaler is None:
        means = X_raw.mean(axis=0, skipna=True)
        stds = X_raw.std(axis=0, skipna=True, ddof=0).replace(0, np.nan)
        scaler = {"means": means, "stds": stds}

    X_imp = X_raw.fillna(scaler["means"])
    X_z = (X_imp - scaler["means"]) / scaler["stds"]
    X_z = X_z.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    keep_features = [c for c in feature_ids if X_z[c].std(ddof=0) > 1e-12]
    X_z = X_z[keep_features].copy()
    X_z.columns = [f"z_{c}" for c in keep_features]
    return X_z, keep_features, scaler


def fit_glm_binomial(y, X, cluster_groups=None, cov_type="cluster"):
    X = sm.add_constant(X, has_constant="add")
    model = sm.GLM(y, X, family=sm.families.Binomial())

    if cov_type == "cluster" and cluster_groups is not None:
        try:
            return model.fit(cov_type="cluster", cov_kwds={"groups": cluster_groups})
        except Exception:
            return model.fit(cov_type="HC1")

    if cov_type == "HC1":
        return model.fit(cov_type="HC1")

    return model.fit()


def loglike_null(y, cluster_groups=None, cov_type="cluster"):
    X0 = pd.DataFrame(index=np.arange(len(y)))
    return fit_glm_binomial(y, X0, cluster_groups=cluster_groups, cov_type=cov_type)


def feature_name_map():
    return {
        fid: {
            "paper_feature_name": name,
            "feature_group": group,
            "paper_expected_direction": expected_direction,
            "paper_p_threshold": paper_threshold,
        }
        for fid, name, group, expected_direction, paper_threshold in FEATURE_SPECS
    }


def result_to_coef_table(result, fmap, regression_type, feature_ids_used):
    params = result.params
    conf = result.conf_int()
    rows = []

    for term in params.index:
        if term == "const":
            term_type = "intercept"
            feature_id = ""
            paper_feature_name = "Intercept"
            feature_group = ""
            expected_direction = ""
            paper_threshold = np.nan
        elif term.startswith("z_"):
            term_type = "feature"
            feature_id = term[2:]
            spec = fmap.get(feature_id, {})
            paper_feature_name = spec.get("paper_feature_name", feature_id)
            feature_group = spec.get("feature_group", "")
            expected_direction = spec.get("paper_expected_direction", "")
            paper_threshold = spec.get("paper_p_threshold", np.nan)
        else:
            term_type = "other"
            feature_id = ""
            paper_feature_name = term
            feature_group = ""
            expected_direction = ""
            paper_threshold = np.nan

        coef = float(params[term])
        ci_low = float(conf.loc[term, 0])
        ci_high = float(conf.loc[term, 1])
        observed_direction = "up" if coef > 0 else "down" if coef < 0 else "flat"

        rows.append({
            "regression": regression_type,
            "term": term,
            "term_type": term_type,
            "feature_id": feature_id,
            "paper_feature_name": paper_feature_name,
            "feature_group": feature_group,
            "paper_expected_direction": expected_direction,
            "paper_p_threshold": paper_threshold,
            "coef_logit": coef,
            "std_err": float(result.bse[term]),
            "z_value": float(result.tvalues[term]),
            "p_value": float(result.pvalues[term]),
            "ci95_low_logit": ci_low,
            "ci95_high_logit": ci_high,
            "odds_ratio": float(np.exp(coef)) if abs(coef) < 700 else np.nan,
            "or_ci95_low": float(np.exp(ci_low)) if abs(ci_low) < 700 else np.nan,
            "or_ci95_high": float(np.exp(ci_high)) if abs(ci_high) < 700 else np.nan,
            "observed_direction": observed_direction,
            "direction_matches_paper": bool(observed_direction == expected_direction) if expected_direction else np.nan,
            "n_features_used": len(feature_ids_used),
        })

    return pd.DataFrame(rows)


def fit_summary(result, null_result, y, pred, regression_type, n_features):
    llf = float(result.llf)
    ll_null = float(null_result.llf)
    lr_stat = max(0.0, 2.0 * (llf - ll_null))
    df_diff = max(1, int(result.df_model - null_result.df_model))
    lr_p = float(stats.chi2.sf(lr_stat, df_diff))

    y_arr = np.asarray(y).astype(int)
    pred_arr = np.asarray(pred)
    pred_label = (pred_arr >= 0.5).astype(int)

    return {
        "regression": regression_type,
        "n_obs": int(len(y_arr)),
        "n_pairs": int(len(y_arr) / 2),
        "positive_rate": float(np.mean(y_arr)),
        "n_features": int(n_features),
        "log_likelihood": llf,
        "null_log_likelihood": ll_null,
        "mcfadden_pseudo_r2": float(1.0 - llf / ll_null) if ll_null != 0 else np.nan,
        "aic": float(result.aic),
        "bic_llf": float(-2 * llf + len(result.params) * np.log(len(y_arr))),
        "lr_statistic_vs_null": lr_stat,
        "lr_df": df_diff,
        "lr_p_value": lr_p,
        "accuracy_at_0_5": float(np.mean(pred_label == y_arr)),
        "auc": auc_score(y_arr, pred_arr),
        "converged": bool(getattr(result, "converged", True)),
    }


def build_human_reply_level_dataset(wide_df):
    rows = []

    for _, r in wide_df.iterrows():
        success_row = {
            "pair_id": r["pair_id"],
            "root_id": r["root_id"],
            "reply_id": r["success_id"],
            "condition": "success",
            "human_delta": 1,
        }
        failure_row = {
            "pair_id": r["pair_id"],
            "root_id": r["root_id"],
            "reply_id": r["failure_id"],
            "condition": "failure",
            "human_delta": 0,
        }

        for feature_id, _, _, _, _ in FEATURE_SPECS:
            success_row[feature_id] = r.get(f"{feature_id}_success", np.nan)
            failure_row[feature_id] = r.get(f"{feature_id}_failure", np.nan)

        rows.append(success_row)
        rows.append(failure_row)

    return pd.DataFrame(rows)


def run_human_logistic_regression(wide_df, out_dir, cov_type="cluster"):
    out_dir = Path(out_dir)
    feature_ids = [f[0] for f in FEATURE_SPECS]
    fmap = feature_name_map()

    reply_df = build_human_reply_level_dataset(wide_df)
    reply_df.to_csv(out_dir / "human_reply_level_feature_dataset.csv", index=False)

    df = reply_df.dropna(subset=["human_delta"]).copy()
    df["human_delta"] = df["human_delta"].astype(int)

    X_z, used_features, scaler = prepare_design_data(df, feature_ids)
    y = df["human_delta"].astype(int).to_numpy()
    groups = df["pair_id"].astype(str).to_numpy() if cov_type == "cluster" else None

    result = fit_glm_binomial(
        y,
        X_z.reset_index(drop=True),
        cluster_groups=groups,
        cov_type=cov_type,
    )
    null_result = loglike_null(
        y,
        cluster_groups=groups,
        cov_type=cov_type,
    )
    pred = result.predict(sm.add_constant(X_z.reset_index(drop=True), has_constant="add"))

    regression_type = "human_logistic_regression_clustered_by_pair" if cov_type == "cluster" else f"human_logistic_regression_{cov_type}"
    coef_df = result_to_coef_table(result, fmap, regression_type, used_features)
    fit_df = pd.DataFrame([fit_summary(result, null_result, y, pred, regression_type, len(used_features))])

    scaler_report = pd.DataFrame({
        "feature_id": scaler["means"].index,
        "mean_used_for_imputation": scaler["means"].values,
        "std_used_for_standardization": scaler["stds"].values,
    })

    coef_df.to_csv(out_dir / "human_logistic_regression_coefficients_all.csv", index=False)
    coef_df[coef_df["term_type"] == "feature"].to_csv(out_dir / "human_logistic_regression_feature_coefficients.csv", index=False)
    fit_df.to_csv(out_dir / "human_logistic_regression_fit_summary.csv", index=False)
    scaler_report.to_csv(out_dir / "human_logistic_feature_standardization_report.csv", index=False)

    with open(out_dir / "human_logistic_regression_model_summary.txt", "w", encoding="utf-8") as f:
        f.write(str(result.summary()))
        f.write("\n\nNotes:\n")
        f.write("- Dependent variable: human_delta, where success reply = 1 and failure reply = 0.\n")
        f.write("- Features are the same text features used in the original paired feature analysis.\n")
        f.write("- Features are z-standardized over all reply-level rows before fitting.\n")
        f.write("- Standard errors are clustered by pair_id by default because each pair contributes one success and one failure reply.\n")
        f.write("- There is no model fixed effect or per-model robustness check because the human data form a single source, not seven separate models.\n")

    return reply_df, coef_df, fit_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--op_path", default=DEFAULT_OP_PATH)
    parser.add_argument("--pair_path", default=DEFAULT_PAIR_PATH)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--vad_lexicon", default=None)
    parser.add_argument("--vad_polar_threshold", type=float, default=0.333)
    parser.add_argument("--sentiment_source", choices=["auto", "liwc", "vad_polar", "fallback"], default="auto")
    parser.add_argument("--stopwords_path", default=None)
    parser.add_argument("--positive_words_path", default=None)
    parser.add_argument("--negative_words_path", default=None)
    parser.add_argument("--hedges_path", default=None)
    parser.add_argument("--liwc_dict_path", default=None)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--progress_every", type=int, default=500)
    parser.add_argument("--cov_type", choices=["cluster", "HC1", "nonrobust"], default="cluster")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading VAD lexicon...")
    vad, vad_pos, vad_neg, vad_scale_report = load_vad_lexicon_and_polar(
        args.vad_lexicon,
        polar_threshold=args.vad_polar_threshold,
        unigrams_only=True,
    )
    print(f"VAD entries: {len(vad)}")
    print(f"VAD positive proxy words: {len(vad_pos)}")
    print(f"VAD negative proxy words: {len(vad_neg)}")
    print(f"VAD scale: {vad_scale_report}")

    positive_words, negative_words, sentiment_source_used = choose_sentiment_lexicons(args, vad_pos, vad_neg)
    pos_exact, pos_stems = split_lexicon(positive_words)
    neg_exact, neg_stems = split_lexicon(negative_words)

    print(f"Sentiment source used: {sentiment_source_used}")
    print(f"Human logistic coefficients: {out_dir / 'human_logistic_regression_feature_coefficients.csv'}")
    print(f"Human logistic fit summary: {out_dir / 'human_logistic_regression_fit_summary.csv'}")
    print(f"Positive lexicon: exact={len(pos_exact)}, stems={len(pos_stems)}")
    print(f"Negative lexicon: exact={len(neg_exact)}, stems={len(neg_stems)}")

    resources = {
        "stopwords": load_word_set(args.stopwords_path, BASE_STOPWORDS),
        "positive_words": positive_words,
        "negative_words": negative_words,
        "positive_words_exact": pos_exact,
        "positive_words_stems": pos_stems,
        "negative_words_exact": neg_exact,
        "negative_words_stems": neg_stems,
        "hedges": load_word_set(args.hedges_path, HEDGE_FALLBACK),
        "vad": vad,
    }

    print("Loading OP map...")
    op_map = build_op_map(args.op_path)
    print(f"OP count: {len(op_map)}")

    rows = []
    errors = []

    print("Extracting features...")
    for idx, (pair_id, pair) in enumerate(iter_pairs(args.pair_path), start=1):
        if args.progress_every > 0 and idx % args.progress_every == 0:
            print(f"Processed pairs: {idx}; usable rows: {len(rows)}; errors: {len(errors)}", flush=True)

        s = pair.get("success", {})
        f = pair.get("failure", {})
        root_id = root_id_from_pair(pair)
        op_text = op_map.get(str(root_id), "")

        if not op_text:
            errors.append({"pair_id": pair_id, "root_id": root_id, "reason": "missing_op"})
            continue

        success_text = s.get("text") or s.get("body") or ""
        failure_text = f.get("text") or f.get("body") or ""

        try:
            sf = all_features(success_text, op_text, resources)
            ff = all_features(failure_text, op_text, resources)
            row = {
                "pair_id": pair_id,
                "root_id": root_id,
                "success_id": s.get("id"),
                "failure_id": f.get("id"),
            }
            for feature_id, _, _, _, _ in FEATURE_SPECS:
                row[f"{feature_id}_success"] = sf.get(feature_id, np.nan)
                row[f"{feature_id}_failure"] = ff.get(feature_id, np.nan)
            rows.append(row)
        except Exception as e:
            errors.append({"pair_id": pair_id, "root_id": root_id, "reason": repr(e)})

    wide_df = pd.DataFrame(rows)
    wide_path = out_dir / "root_reply_feature_values_wide.csv"
    wide_df.to_csv(wide_path, index=False)

    print("Saving long feature table...")
    long_rows = []
    for _, r in wide_df.iterrows():
        for feature_id, feature_name, group, expected_direction, paper_threshold in FEATURE_SPECS:
            sv = r.get(f"{feature_id}_success", np.nan)
            fv = r.get(f"{feature_id}_failure", np.nan)
            long_rows.append({
                "pair_id": r["pair_id"],
                "root_id": r["root_id"],
                "success_id": r["success_id"],
                "failure_id": r["failure_id"],
                "feature_id": feature_id,
                "paper_feature_name": feature_name,
                "feature_group": group,
                "paper_expected_direction": expected_direction,
                "paper_p_threshold": paper_threshold,
                "success_value": sv,
                "failure_value": fv,
                "difference_success_minus_failure": sv - fv if pd.notna(sv) and pd.notna(fv) else np.nan,
            })
    pd.DataFrame(long_rows).to_csv(out_dir / "root_reply_feature_values_long.csv", index=False)

    print("Running paired tests...")
    m = len(FEATURE_SPECS)
    summary_rows = [
        paired_summary(
            wide_df,
            feature_id,
            feature_name,
            group,
            expected_direction,
            paper_threshold,
            args.alpha,
            m,
        )
        for feature_id, feature_name, group, expected_direction, paper_threshold in FEATURE_SPECS
    ]

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "root_reply_paired_test_summary.csv", index=False)

    with open(out_dir / "root_reply_paired_test_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)

    print("Running human logistic regression...")
    human_reply_df, human_logit_coef_df, human_logit_fit_df = run_human_logistic_regression(
        wide_df,
        out_dir,
        cov_type=args.cov_type,
    )

    save_feature_level_files(wide_df, out_dir)
    pd.DataFrame(errors).to_csv(out_dir / "skipped_or_error_pairs.csv", index=False)

    resource_report = {
        "n_pairs_used": int(len(wide_df)),
        "n_pairs_skipped_or_error": int(len(errors)),
        "n_features": int(m),
        "bonferroni_factor": int(m),
        "op_path": args.op_path,
        "pair_path": args.pair_path,
        "out_dir": str(out_dir),
        "stopwords_source": args.stopwords_path or "fallback_internal",
        "positive_words_source": sentiment_source_used,
        "negative_words_source": sentiment_source_used,
        "positive_words_exact_count": int(len(pos_exact)),
        "positive_words_stem_count": int(len(pos_stems)),
        "negative_words_exact_count": int(len(neg_exact)),
        "negative_words_stem_count": int(len(neg_stems)),
        "hedges_source": args.hedges_path or "fallback_internal",
        "liwc_dict_source": args.liwc_dict_path or "missing_liwc",
        "vad_lexicon_source": args.vad_lexicon or "missing_vad",
        "vad_scale_report": vad_scale_report,
        "human_logistic_regression": "enabled",
        "human_logistic_cov_type": args.cov_type,
        "human_logistic_note": "Dependent variable is human_delta. Success reply = 1, failure reply = 0. No model fixed effects or per-model robustness check are used because humans are treated as a single source. Standard errors are clustered by pair_id by default.",
        "note": "If LIWC is unavailable, positive/negative word counts are approximated using NRC VAD valence polar words. This is a proxy, not a strict LIWC replication.",
    }
    with open(out_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(resource_report, f, ensure_ascii=False, indent=2)

    measured_df = summary_df[summary_df["n_pairs"] >= 2].copy()
    unmeasured_df = summary_df[summary_df["n_pairs"] < 2].copy()
    measured_m = len(measured_df)
    replicated = int(measured_df["replicates_paper_by_bonferroni_threshold"].fillna(False).sum())
    direction_matched = int(measured_df["direction_matches_paper"].fillna(False).sum())
    unmeasured_features = ", ".join(unmeasured_df["paper_feature_name"].tolist())

    with open(out_dir / "quick_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Pairs used: {len(wide_df)}\n")
        f.write(f"Pairs skipped/error: {len(errors)}\n")
        f.write(f"Features total: {m}\n")
        f.write(f"Features measured: {measured_m}\n")
        f.write(f"Features unmeasured: {len(unmeasured_df)}\n")
        f.write(f"Unmeasured feature names: {unmeasured_features}\n")
        f.write(f"Direction matched paper among measured: {direction_matched}/{measured_m}\n")
        f.write(f"Replicated by Bonferroni threshold and direction among measured: {replicated}/{measured_m}\n")
        f.write(f"Sentiment source used: {sentiment_source_used}\n")
        f.write(f"VAD scale report: {vad_scale_report}\n")
        f.write(f"Summary CSV: {out_dir / 'root_reply_paired_test_summary.csv'}\n")
        f.write(f"Wide feature CSV: {wide_path}\n")
        f.write(f"Human logistic coefficients CSV: {out_dir / 'human_logistic_regression_feature_coefficients.csv'}\n")
        f.write(f"Human logistic fit summary CSV: {out_dir / 'human_logistic_regression_fit_summary.csv'}\n")

    print(f"Done. Results saved to: {out_dir}")
    print(f"Pairs used: {len(wide_df)}")
    print(f"Pairs skipped/error: {len(errors)}")
    print(f"Features total: {m}")
    print(f"Features measured: {measured_m}")
    print(f"Features unmeasured: {len(unmeasured_df)}")
    if unmeasured_features:
        print(f"Unmeasured feature names: {unmeasured_features}")
    print(f"Direction matched paper among measured: {direction_matched}/{measured_m}")
    print(f"Replicated by Bonferroni threshold and direction among measured: {replicated}/{measured_m}")
    print(f"Sentiment source used: {sentiment_source_used}")


if __name__ == "__main__":
    main()
