import argparse
import html
import json
import math
import os
import re
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from scipy import stats


try:
    import statsmodels.api as sm
except Exception as e:
    raise ImportError(
        "This script requires statsmodels. Install it with: python -m pip install statsmodels"
    ) from e


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

DEFAULT_OP_PATH = os.path.join(DATA, "dataset", "2262_unique_ops.json")
DEFAULT_PAIR_PATH = os.path.join(DATA, "dataset", "single_turn_pairs.json")
DEFAULT_OUT_DIR = os.path.join(ROOT, "regression_output", "observer")

DEFAULT_AGENT_RESULT_PATHS = [
    os.path.join(DATA, "third_person", "final_v3_results_observer_deepseek-ai_DeepSeek-V3.json"),
    os.path.join(DATA, "third_person", "final_v3_results_observer_google_gemini-2.5-flash-lite.json"),
    os.path.join(DATA, "third_person", "final_v3_results_observer_gpt-4o-mini.json"),
    os.path.join(DATA, "third_person", "final_v3_results_observer_Pro_MiniMaxAI_MiniMax-M2.5.json"),
    os.path.join(DATA, "third_person", "final_v3_results_observer_Pro_zai-org_GLM-4.7.json"),
    os.path.join(DATA, "third_person", "final_v3_results_observer_Qwen_Qwen2.5-32B-Instruct.json"),
    os.path.join(DATA, "third_person", "final_v3_results_observer_Qwen_Qwen2.5-72B-Instruct.json"),
    os.path.join(DATA, "third_person", "final_v3_results_observer_gpt-5.5.json"),
]


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


def infer_model_name(path):
    stem = Path(path).stem
    name = stem
    for prefix in [
        "final_v3_results_observer_",
        "final_new_observer_results_",
        "final_new_results_",
        "final_results_observer_",
        "final_results_",
        "observer_results_",
        "results_",
    ]:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return name.replace("/", "_")


def parse_bool_like(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value in (0, 1):
            return bool(value)
        return None
    if isinstance(value, str):
        s = value.strip()
        low = s.lower()

        if low in {"true", "false"}:
            return low == "true"
        if low in {"yes", "no"}:
            return low == "yes"
        if low in {"1", "0"}:
            return low == "1"

        m = re.search(r'"?delta_awarded"?\s*[:=]\s*(true|false)', low)
        if m:
            return m.group(1) == "true"

        try:
            parsed = json.loads(s)
            return parse_bool_from_obj(parsed)
        except Exception:
            return None
    return None


def parse_bool_from_obj(obj):
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, dict):
        priority_keys = [
            "agent_delta", "delta_awarded", "delta", "award_delta", "awarded_delta",
            "changed_view", "view_changed", "is_persuasive", "persuaded",
            "prediction", "pred", "label", "answer", "decision"
        ]
        for k in priority_keys:
            if k in obj:
                b = parse_bool_like(obj[k])
                if b is not None:
                    return b

        for k in ["response", "content", "message", "output", "raw_output", "text", "completion", "result"]:
            if k in obj and isinstance(obj[k], str):
                b = parse_bool_like(obj[k])
                if b is not None:
                    return b
        return None

    if isinstance(obj, str):
        return parse_bool_like(obj)
    return None


def get_first_value(obj, keys):
    if not isinstance(obj, dict):
        return None
    for k in keys:
        if k in obj and obj[k] not in (None, ""):
            val = obj[k]
            if isinstance(val, list) and val:
                return val[0]
            return val
    return None


def normalize_condition(value):
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in {"success", "successful", "positive", "winner", "winning", "delta", "human_success"}:
        return "success"
    if s in {"failure", "failed", "negative", "loser", "losing", "no_delta", "human_failure"}:
        return "failure"

    if "human_success" in s or s.endswith("_success") or "branch_a" in s:
        return "success"
    if "human_failure" in s or s.endswith("_failure") or "branch_b" in s:
        return "failure"

    return None


def extract_agent_predictions(agent_json, model_name):
    records = []

    def add_record(pair_id, condition, reply_id, label, path):
        if label is None:
            return
        if not pair_id and not reply_id:
            return
        records.append({
            "model": model_name,
            "pair_id": str(pair_id) if pair_id is not None else None,
            "condition": condition,
            "reply_id": str(reply_id) if reply_id is not None else None,
            "agent_delta_awarded": int(bool(label)),
            "path": "/".join(map(str, path[-8:])),
        })

    def parse_known_pair_record(obj, path=()):
        if not isinstance(obj, dict):
            return False

        pair_id = obj.get("pair_id") or obj.get("pair") or obj.get("pair_key")
        if pair_id is None:
            return False

        found = False
        for branch_key, condition in [
            ("branch_A_human_success", "success"),
            ("branch_B_human_failure", "failure"),
            ("branch_A_success", "success"),
            ("branch_B_failure", "failure"),
            ("success", "success"),
            ("failure", "failure"),
        ]:
            branch = obj.get(branch_key)
            if isinstance(branch, dict):
                # H1: skip API-failure branches (recorded as a fake False label)
                if branch.get("raw_error") or branch.get("error") or \
                   (branch.get("justification") or "").strip() == "API Error":
                    continue
                label = parse_bool_like(branch.get("agent_delta"))
                if label is None:
                    label = parse_bool_like(branch.get("delta_awarded"))
                if label is None:
                    label = parse_bool_from_obj(branch)

                reply_id = (
                    branch.get("reply_id")
                    or branch.get("comment_id")
                    or branch.get("id")
                    or branch.get("response_id")
                )

                add_record(pair_id, condition, reply_id, label, path + (branch_key,))
                found = found or (label is not None)

        return found

    if isinstance(agent_json, list):
        for i, item in enumerate(agent_json):
            parse_known_pair_record(item, path=(i,))
    elif isinstance(agent_json, dict):
        if not parse_known_pair_record(agent_json, path=()):
            for k, v in agent_json.items():
                if isinstance(v, dict):
                    if re.match(r"^p_\d+$", str(k)) and "pair_id" not in v:
                        vv = dict(v)
                        vv["pair_id"] = str(k)
                        parse_known_pair_record(vv, path=(k,))
                    else:
                        parse_known_pair_record(v, path=(k,))

    def walk(obj, pair_ctx=None, condition_ctx=None, reply_id_ctx=None, path=()):
        if isinstance(obj, dict):
            pair_here = pair_ctx
            condition_here = condition_ctx
            reply_id_here = reply_id_ctx

            pair_candidate = get_first_value(obj, ["pair_id", "pair", "pair_key", "sample_id", "example_id"])
            if pair_candidate is not None and re.match(r"^p_\d+$", str(pair_candidate)):
                pair_here = str(pair_candidate)

            condition_candidate = get_first_value(obj, ["condition", "reply_type", "candidate_type", "human_label", "gold_type"])
            condition_norm = normalize_condition(condition_candidate)
            if condition_norm:
                condition_here = condition_norm

            reply_candidate = get_first_value(obj, ["reply_id", "comment_id", "challenger_id", "response_id"])
            if reply_candidate is not None:
                reply_id_here = str(reply_candidate)

            if "id" in obj and isinstance(obj.get("id"), str) and obj["id"].startswith("t1_"):
                reply_id_here = obj["id"]

            explicit_label = None
            for key in ["agent_delta", "delta_awarded", "award_delta", "awarded_delta", "prediction", "pred", "answer", "decision"]:
                if key in obj:
                    explicit_label = parse_bool_like(obj[key])
                    if explicit_label is not None:
                        break

            if explicit_label is not None:
                add_record(pair_here, condition_here, reply_id_here, explicit_label, path)

            for k, v in obj.items():
                next_pair = pair_here
                next_condition = condition_here
                next_reply = reply_id_here

                ks = str(k)
                if re.match(r"^p_\d+$", ks):
                    next_pair = ks
                cond_from_key = normalize_condition(ks)
                if cond_from_key:
                    next_condition = cond_from_key
                if ks.startswith("t1_"):
                    next_reply = ks

                walk(v, next_pair, next_condition, next_reply, path + (ks,))

        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                walk(item, pair_ctx, condition_ctx, reply_id_ctx, path + (i,))

        elif isinstance(obj, str):
            label = parse_bool_like(obj)
            if label is not None and (pair_ctx or reply_id_ctx):
                add_record(pair_ctx, condition_ctx, reply_id_ctx, label, path)

    walk(agent_json)

    if not records:
        return pd.DataFrame(columns=["model", "pair_id", "condition", "reply_id", "agent_delta_awarded", "path"])

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["model", "pair_id", "condition", "reply_id", "agent_delta_awarded"])
    df = df.drop_duplicates(subset=["model", "pair_id", "condition"], keep="first")

    return df


def build_agent_lookup(pred_df):
    by_pair_condition = {}
    by_reply_id = {}

    if pred_df.empty:
        return by_pair_condition, by_reply_id

    for _, r in pred_df.iterrows():
        label = int(r["agent_delta_awarded"])
        pair_id = r.get("pair_id")
        condition = r.get("condition")
        reply_id = r.get("reply_id")

        if pd.notna(pair_id) and pd.notna(condition):
            key = (str(pair_id), str(condition))
            by_pair_condition.setdefault(key, []).append(label)

        if pd.notna(reply_id):
            by_reply_id.setdefault(str(reply_id), []).append(label)

    by_pair_condition = {k: int(round(np.mean(v))) for k, v in by_pair_condition.items()}
    by_reply_id = {k: int(round(np.mean(v))) for k, v in by_reply_id.items()}
    return by_pair_condition, by_reply_id


def build_base_feature_dataset(pair_path, op_path, resources, progress_every=500):
    op_map = build_op_map(op_path)
    rows = []
    errors = []

    for idx, (pair_id, pair) in enumerate(iter_pairs(pair_path), start=1):
        if progress_every and idx % progress_every == 0:
            print(f"Feature extraction: processed pairs={idx}, rows={len(rows)}, errors={len(errors)}", flush=True)

        s = pair.get("success", {})
        f = pair.get("failure", {})
        root_id = root_id_from_pair(pair)
        op_text = op_map.get(str(root_id), "")
        if not op_text:
            errors.append({"pair_id": pair_id, "root_id": root_id, "reason": "missing_op"})
            continue

        for condition, item, human_delta in [("success", s, 1), ("failure", f, 0)]:
            text = item.get("text") or item.get("body") or ""
            try:
                feats = all_features(text, op_text, resources)
                row = {
                    "pair_id": pair_id,
                    "root_id": root_id,
                    "reply_id": item.get("id"),
                    "condition": condition,
                    "human_delta": human_delta,
                    "reply_user": item.get("user"),
                    "timestamp": item.get("timestamp"),
                }
                row.update(feats)
                rows.append(row)
            except Exception as e:
                errors.append({
                    "pair_id": pair_id,
                    "root_id": root_id,
                    "condition": condition,
                    "reply_id": item.get("id"),
                    "reason": repr(e),
                })

    return pd.DataFrame(rows), pd.DataFrame(errors)


def attach_agent_labels(base_df, agent_paths, out_dir):
    all_rows = []
    parse_reports = []
    unmatched_rows = []
    raw_pred_frames = []

    for path in agent_paths:
        model_name = infer_model_name(path)
        p = Path(path)
        if not p.exists():
            parse_reports.append({
                "model": model_name,
                "path": str(path),
                "status": "missing_file",
                "n_extracted_predictions": 0,
                "n_matched_rows": 0,
                "n_unmatched_rows": len(base_df),
            })
            continue

        try:
            raw = load_json(p)
            pred_df = extract_agent_predictions(raw, model_name)
            raw_pred_frames.append(pred_df)
            by_pair_condition, by_reply_id = build_agent_lookup(pred_df)

            model_rows = []
            unmatched = []
            for _, r in base_df.iterrows():
                label = None
                key = (str(r["pair_id"]), str(r["condition"]))
                if key in by_pair_condition:
                    label = by_pair_condition[key]
                elif pd.notna(r.get("reply_id")) and str(r["reply_id"]) in by_reply_id:
                    label = by_reply_id[str(r["reply_id"])]

                if label is None:
                    unmatched.append({
                        "model": model_name,
                        "pair_id": r["pair_id"],
                        "condition": r["condition"],
                        "reply_id": r["reply_id"],
                    })
                    continue

                row = r.to_dict()
                row["model"] = model_name
                row["agent_delta_awarded"] = int(label)
                model_rows.append(row)

            all_rows.extend(model_rows)
            unmatched_rows.extend(unmatched)

            parse_reports.append({
                "model": model_name,
                "path": str(path),
                "status": "ok",
                "n_extracted_predictions": int(len(pred_df)),
                "n_unique_pair_condition_predictions": int(len(by_pair_condition)),
                "n_unique_reply_id_predictions": int(len(by_reply_id)),
                "n_matched_rows": int(len(model_rows)),
                "n_unmatched_rows": int(len(unmatched)),
                "agent_positive_rate_in_matched": float(np.mean([x["agent_delta_awarded"] for x in model_rows])) if model_rows else np.nan,
            })

        except Exception as e:
            parse_reports.append({
                "model": model_name,
                "path": str(path),
                "status": "error",
                "error": repr(e),
                "n_extracted_predictions": 0,
                "n_matched_rows": 0,
                "n_unmatched_rows": len(base_df),
            })

    agent_df = pd.DataFrame(all_rows)
    parse_report_df = pd.DataFrame(parse_reports)
    unmatched_df = pd.DataFrame(unmatched_rows)

    if raw_pred_frames:
        raw_preds = pd.concat(raw_pred_frames, ignore_index=True)
    else:
        raw_preds = pd.DataFrame()

    raw_preds.to_csv(Path(out_dir) / "agent_raw_extracted_predictions.csv", index=False)
    parse_report_df.to_csv(Path(out_dir) / "agent_prediction_parse_report.csv", index=False)
    unmatched_df.to_csv(Path(out_dir) / "agent_unmatched_rows.csv", index=False)

    return agent_df, parse_report_df, unmatched_df


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
    data = df.copy()
    X_raw = data[feature_ids].replace([np.inf, -np.inf], np.nan)

    if scaler is None:
        means = X_raw.mean(axis=0, skipna=True)
        stds = X_raw.std(axis=0, skipna=True, ddof=0)
        stds = stds.replace(0, np.nan)
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


def loglike_null(y, model_dummies=None, cluster_groups=None, cov_type="cluster"):
    if model_dummies is not None and len(model_dummies.columns) > 0:
        X0 = model_dummies.copy()
    else:
        X0 = pd.DataFrame(index=np.arange(len(y)))
    return fit_glm_binomial(y, X0, cluster_groups=cluster_groups, cov_type=cov_type)


def result_to_coef_table(result, feature_name_map, model_label, regression_type, feature_ids_used):
    params = result.params
    conf = result.conf_int()
    rows = []

    for term in params.index:
        if term == "const":
            clean_name = "Intercept"
            term_type = "intercept"
            feature_id = ""
            paper_feature_name = "Intercept"
            feature_group = ""
        elif term.startswith("z_"):
            feature_id = term[2:]
            clean_name = feature_id
            term_type = "feature"
            spec = feature_name_map.get(feature_id, {})
            paper_feature_name = spec.get("paper_feature_name", feature_id)
            feature_group = spec.get("feature_group", "")
        elif term.startswith("model_"):
            clean_name = term
            term_type = "model_fixed_effect"
            feature_id = ""
            paper_feature_name = term.replace("model_", "")
            feature_group = ""
        else:
            clean_name = term
            term_type = "other"
            feature_id = ""
            paper_feature_name = term
            feature_group = ""

        coef = float(params[term])
        ci_low = float(conf.loc[term, 0])
        ci_high = float(conf.loc[term, 1])
        rows.append({
            "regression": regression_type,
            "model": model_label,
            "term": clean_name,
            "term_type": term_type,
            "feature_id": feature_id,
            "paper_feature_name": paper_feature_name,
            "feature_group": feature_group,
            "coef_logit": coef,
            "std_err": float(result.bse[term]),
            "z_value": float(result.tvalues[term]),
            "p_value": float(result.pvalues[term]),
            "ci95_low_logit": ci_low,
            "ci95_high_logit": ci_high,
            "odds_ratio": float(np.exp(coef)) if abs(coef) < 700 else np.nan,
            "or_ci95_low": float(np.exp(ci_low)) if abs(ci_low) < 700 else np.nan,
            "or_ci95_high": float(np.exp(ci_high)) if abs(ci_high) < 700 else np.nan,
            "n_features_used": len(feature_ids_used),
        })
    return pd.DataFrame(rows)


def fit_summary(result, null_result, y, pred, model_label, regression_type, n_features, n_model_fe=0):
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
        "model": model_label,
        "n_obs": int(len(y_arr)),
        "positive_rate": float(np.mean(y_arr)),
        "n_features": int(n_features),
        "n_model_fixed_effects": int(n_model_fe),
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


def run_regressions(agent_df, out_dir, cov_type="cluster"):
    feature_ids = [f[0] for f in FEATURE_SPECS]
    feature_name_map = {
        fid: {"paper_feature_name": name, "feature_group": group}
        for fid, name, group, _, _ in FEATURE_SPECS
    }

    df = agent_df.copy()
    df = df.dropna(subset=["agent_delta_awarded"])
    df["agent_delta_awarded"] = df["agent_delta_awarded"].astype(int)

    X_z, used_features, scaler = prepare_design_data(df, feature_ids)
    y = df["agent_delta_awarded"].astype(int).to_numpy()

    coef_tables = []
    fit_rows = []
    errors = []

    # Pooled regression with model fixed effects.
    try:
        model_dummies = pd.get_dummies(df["model"].astype(str), prefix="model", drop_first=True, dtype=float)
        X_pool = pd.concat([X_z.reset_index(drop=True), model_dummies.reset_index(drop=True)], axis=1)
        groups = df["pair_id"].astype(str).to_numpy() if cov_type == "cluster" else None
        pool_res = fit_glm_binomial(y, X_pool, cluster_groups=groups, cov_type=cov_type)
        null_res = loglike_null(y, model_dummies.reset_index(drop=True), cluster_groups=groups, cov_type=cov_type)
        pred = pool_res.predict(sm.add_constant(X_pool, has_constant="add"))

        coef_tables.append(result_to_coef_table(pool_res, feature_name_map, "POOLED_WITH_MODEL_FE", "pooled_model_fixed_effects", used_features))
        fit_rows.append(fit_summary(pool_res, null_res, y, pred, "POOLED_WITH_MODEL_FE", "pooled_model_fixed_effects", len(used_features), n_model_fe=model_dummies.shape[1]))

        # ===== 新增：输出 observer pooled GLM summary txt 文件 =====
        with open(Path(out_dir) / "observer_logistic_regression_model_summary.txt", "w", encoding="utf-8") as f:
            f.write(str(pool_res.summary()))
            f.write("\n\nNotes:\n")
            f.write("- Dependent variable: agent_delta_awarded parsed from observer files, where predicted success = 1 and predicted failure = 0.\n")
            f.write("- Human success/failure branch labels are retained only for alignment; the regression target is the observer model judgment.\n")
            f.write("- Features are the same text features used in the original paired feature analysis.\n")
            f.write("- Features are z-standardized over all reply-level rows before fitting.\n")
            f.write(f"- Standard errors are clustered by pair_id (using cov_type='{cov_type}').\n")
            f.write("- This summary represents the POOLED observer model WITH model fixed effects to account for multiple observer LLMs.\n")
        # ===== 新增结束 =====

    except Exception as e:
        errors.append({"regression": "pooled_model_fixed_effects", "model": "POOLED_WITH_MODEL_FE", "error": repr(e)})

    # Pooled regression without model fixed effects, saved for comparison.
    try:
        groups = df["pair_id"].astype(str).to_numpy() if cov_type == "cluster" else None
        pool_no_fe_res = fit_glm_binomial(y, X_z.reset_index(drop=True), cluster_groups=groups, cov_type=cov_type)
        null_no_fe_res = loglike_null(y, None, cluster_groups=groups, cov_type=cov_type)
        pred_no_fe = pool_no_fe_res.predict(sm.add_constant(X_z.reset_index(drop=True), has_constant="add"))

        coef_tables.append(result_to_coef_table(pool_no_fe_res, feature_name_map, "POOLED_NO_MODEL_FE", "pooled_no_model_fixed_effects", used_features))
        fit_rows.append(fit_summary(pool_no_fe_res, null_no_fe_res, y, pred_no_fe, "POOLED_NO_MODEL_FE", "pooled_no_model_fixed_effects", len(used_features), n_model_fe=0))

    except Exception as e:
        errors.append({"regression": "pooled_no_model_fixed_effects", "model": "POOLED_NO_MODEL_FE", "error": repr(e)})

    # Per-model robustness checks.
    for model_name, sub in df.groupby("model"):
        try:
            if sub["agent_delta_awarded"].nunique() < 2:
                errors.append({
                    "regression": "per_model_robustness",
                    "model": model_name,
                    "error": "dependent variable has only one class",
                    "n_obs": len(sub),
                    "positive_rate": float(sub["agent_delta_awarded"].mean()),
                })
                continue

            X_sub, used_sub_features, _ = prepare_design_data(sub, feature_ids, scaler=scaler)
            y_sub = sub["agent_delta_awarded"].astype(int).to_numpy()
            groups_sub = sub["pair_id"].astype(str).to_numpy() if cov_type == "cluster" else None

            res_sub = fit_glm_binomial(y_sub, X_sub.reset_index(drop=True), cluster_groups=groups_sub, cov_type=cov_type)
            null_sub = loglike_null(y_sub, None, cluster_groups=groups_sub, cov_type=cov_type)
            pred_sub = res_sub.predict(sm.add_constant(X_sub.reset_index(drop=True), has_constant="add"))

            coef_tables.append(result_to_coef_table(res_sub, feature_name_map, model_name, "per_model_robustness", used_sub_features))
            fit_rows.append(fit_summary(res_sub, null_sub, y_sub, pred_sub, model_name, "per_model_robustness", len(used_sub_features), n_model_fe=0))

        except Exception as e:
            errors.append({"regression": "per_model_robustness", "model": model_name, "error": repr(e), "n_obs": len(sub)})

    coef_df = pd.concat(coef_tables, ignore_index=True) if coef_tables else pd.DataFrame()
    fit_df = pd.DataFrame(fit_rows)
    error_df = pd.DataFrame(errors)

    coef_df.to_csv(Path(out_dir) / "agent_logistic_regression_coefficients_all.csv", index=False)
    fit_df.to_csv(Path(out_dir) / "agent_logistic_regression_fit_summary.csv", index=False)
    error_df.to_csv(Path(out_dir) / "agent_logistic_regression_errors.csv", index=False)

    if not coef_df.empty:
        pooled_feature = coef_df[
            (coef_df["regression"] == "pooled_model_fixed_effects") &
            (coef_df["term_type"] == "feature")
        ].copy()
        pooled_feature.to_csv(Path(out_dir) / "agent_pooled_model_fe_feature_coefficients.csv", index=False)

        pooled_fe = coef_df[
            (coef_df["regression"] == "pooled_model_fixed_effects") &
            (coef_df["term_type"] == "model_fixed_effect")
        ].copy()
        pooled_fe.to_csv(Path(out_dir) / "agent_pooled_model_fixed_effects.csv", index=False)

        per_model = coef_df[
            (coef_df["regression"] == "per_model_robustness") &
            (coef_df["term_type"] == "feature")
        ].copy()
        per_model.to_csv(Path(out_dir) / "agent_per_model_robustness_feature_coefficients.csv", index=False)

        if not per_model.empty:
            wide_coef = per_model.pivot_table(
                index=["feature_id", "paper_feature_name", "feature_group"],
                columns="model",
                values="coef_logit",
                aggfunc="first",
            ).reset_index()
            wide_p = per_model.pivot_table(
                index=["feature_id", "paper_feature_name", "feature_group"],
                columns="model",
                values="p_value",
                aggfunc="first",
            ).reset_index()
            wide_coef.to_csv(Path(out_dir) / "agent_per_model_robustness_coefficients_wide.csv", index=False)
            wide_p.to_csv(Path(out_dir) / "agent_per_model_robustness_pvalues_wide.csv", index=False)

    scaler_report = pd.DataFrame({
        "feature_id": scaler["means"].index,
        "mean_used_for_imputation": scaler["means"].values,
        "std_used_for_standardization": scaler["stds"].values,
    })
    scaler_report.to_csv(Path(out_dir) / "feature_standardization_report.csv", index=False)

    return coef_df, fit_df, error_df


def save_agent_dataset_summary(agent_df, out_dir):
    if agent_df.empty:
        return

    summary = agent_df.groupby("model").agg(
        n_obs=("agent_delta_awarded", "size"),
        positive_rate=("agent_delta_awarded", "mean"),
        n_pairs=("pair_id", "nunique"),
        human_success_rows=("human_delta", "sum"),
    ).reset_index()
    summary["human_success_rate_in_rows"] = summary["human_success_rows"] / summary["n_obs"]
    summary.to_csv(Path(out_dir) / "agent_dataset_summary_by_model.csv", index=False)

    ctab = pd.crosstab(
        [agent_df["model"], agent_df["human_delta"]],
        agent_df["agent_delta_awarded"],
        rownames=["model", "human_delta"],
        colnames=["agent_delta_awarded"],
        dropna=False,
    ).reset_index()
    ctab.to_csv(Path(out_dir) / "agent_human_label_vs_agent_prediction_crosstab.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--op_path", default=DEFAULT_OP_PATH)
    parser.add_argument("--pair_path", default=DEFAULT_PAIR_PATH)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--agent_result_paths", nargs="*", default=DEFAULT_AGENT_RESULT_PATHS)
    parser.add_argument("--observer_result_paths", nargs="*", default=None)

    parser.add_argument("--vad_lexicon", default=None)
    parser.add_argument("--vad_polar_threshold", type=float, default=0.333)
    parser.add_argument("--sentiment_source", choices=["auto", "liwc", "vad_polar", "fallback"], default="auto")
    parser.add_argument("--stopwords_path", default=None)
    parser.add_argument("--positive_words_path", default=None)
    parser.add_argument("--negative_words_path", default=None)
    parser.add_argument("--hedges_path", default=None)
    parser.add_argument("--liwc_dict_path", default=None)

    parser.add_argument("--cov_type", choices=["cluster", "HC1", "nonrobust"], default="cluster")
    parser.add_argument("--progress_every", type=int, default=500)
    args = parser.parse_args()

    if args.observer_result_paths:
        args.agent_result_paths = args.observer_result_paths

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

    print("Building base feature dataset...")
    base_df, feature_errors = build_base_feature_dataset(
        args.pair_path,
        args.op_path,
        resources,
        progress_every=args.progress_every,
    )
    base_df.to_csv(out_dir / "base_reply_feature_dataset_without_agent_labels.csv", index=False)
    feature_errors.to_csv(out_dir / "feature_extraction_errors.csv", index=False)
    print(f"Base feature rows: {len(base_df)}")
    print(f"Feature extraction errors: {len(feature_errors)}")

    print("Attaching agent labels...")
    agent_df, parse_report, unmatched = attach_agent_labels(base_df, args.agent_result_paths, out_dir)
    agent_df.to_csv(out_dir / "agent_reply_feature_dataset.csv", index=False)
    save_agent_dataset_summary(agent_df, out_dir)

    print(f"Agent regression rows: {len(agent_df)}")
    print(f"Unmatched agent rows: {len(unmatched)}")

    if agent_df.empty:
        raise RuntimeError("No agent labels matched the base reply feature dataset. Check agent JSON structure and agent_prediction_parse_report.csv.")

    print("Running pooled and per-model logistic regressions...")
    coef_df, fit_df, error_df = run_regressions(agent_df, out_dir, cov_type=args.cov_type)

    metadata = {
        "op_path": args.op_path,
        "pair_path": args.pair_path,
        "agent_result_paths": args.agent_result_paths,
        "out_dir": str(out_dir),
        "n_base_reply_rows": int(len(base_df)),
        "n_agent_regression_rows": int(len(agent_df)),
        "n_unmatched_agent_rows": int(len(unmatched)),
        "sentiment_source_used": sentiment_source_used,
        "vad_lexicon_source": args.vad_lexicon or "missing_vad",
        "vad_scale_report": vad_scale_report,
        "cov_type": args.cov_type,
        "excluded_feature": "common_in_all",
        "n_features_after_exclusion": 40,
        "regression_note": "Observer-result version. Dependent variable is agent_delta_awarded from observer files. Pooled regression includes model fixed effects; per-model regressions are robustness checks. Feature common_in_all is excluded, so regressions use the remaining 40 features.",
    }
    with open(out_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    with open(out_dir / "quick_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Base reply feature rows: {len(base_df)}\n")
        f.write(f"Feature extraction errors: {len(feature_errors)}\n")
        f.write(f"Agent regression rows: {len(agent_df)}\n")
        f.write("Features used: 40; excluded feature: common_in_all\n")
        f.write(f"Unmatched agent rows: {len(unmatched)}\n")
        f.write(f"Models matched: {agent_df['model'].nunique()}\n")
        f.write(f"Sentiment source used: {sentiment_source_used}\n")
        f.write(f"VAD scale: {vad_scale_report}\n")
        f.write(f"Coefficient file: {out_dir / 'agent_logistic_regression_coefficients_all.csv'}\n")
        f.write(f"Fit summary file: {out_dir / 'agent_logistic_regression_fit_summary.csv'}\n")
        f.write(f"Model summary text file: {out_dir / 'observer_logistic_regression_model_summary.txt'}\n")
        f.write(f"Regression errors file: {out_dir / 'agent_logistic_regression_errors.csv'}\n")

    print("Done.")
    print(f"Results saved to: {out_dir}")
    print(f"Models matched: {agent_df['model'].nunique()}")
    print(f"Coefficient file: {out_dir / 'agent_logistic_regression_coefficients_all.csv'}")
    print(f"Fit summary file: {out_dir / 'agent_logistic_regression_fit_summary.csv'}")
    print(f"Model summary text file: {out_dir / 'observer_logistic_regression_model_summary.txt'}")
    if not error_df.empty:
        print(f"Some regressions had errors. See: {out_dir / 'agent_logistic_regression_errors.csv'}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
