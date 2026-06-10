import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.preprocessing import FunctionTransformer

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# ── Regex ─────────────────────────────────────────────────────────────────────
_URL_RE   = re.compile(r"https?://\S+|www\.\S+")
_HTML_RE  = re.compile(r"<.*?>")
_DIGIT_RE = re.compile(r"[0-9]+")
_ALPHA_RE = re.compile(r"[^a-zA-Z]")
_SPEC_RE  = re.compile(r"[^a-z\s]")
_SPACE_RE = re.compile(r"\s+")
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)

_CONTRACTIONS = [
    (r"won't",   "will not"),
    (r"can't",   "can not"),
    (r"don't",   "do not"),
    (r"ain't",   "am not"),
    (r"shan't",  "shall not"),
    (r"let's",   "let us"),
    (r"ma'am",   "madam"),
    (r"o'clock", "of the clock"),
    (r"y'all",   "you all"),
    (r"n't",     " not"),
    (r"'re",     " are"),
    (r"'s",      " is"),
    (r"'d",      " would"),
    (r"'ll",     " will"),
    (r"'ve",     " have"),
    (r"'m",      " am"),
]

# ── Stop-words: NLTK + frequent social-media noise ────────────────────────────
_CUSTOM_STOP = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "is", "are", "was", "were", "it", "this", "that", "i",
    "you", "he", "she", "we", "they", "be", "have", "has", "had", "do",
    "does", "did", "will", "would", "can", "could", "not", "no", "so",
    "from", "by", "as", "my", "your", "our", "its", "his", "her", "their",
    "what", "who", "how", "when", "where", "why", "rt", "amp", "via",
    "am", "up", "out", "about", "all", "just", "if", "into", "there",
    "than", "more", "after", "been", "over", "new", "now", "get", "one",
    "two", "also", "http",
}
_NLTK_STOP = set(stopwords.words("english"))
_ALL_STOP   = _NLTK_STOP | _CUSTOM_STOP

_stemmer    = PorterStemmer()
_lemmatizer = WordNetLemmatizer()


def preprocess_data(text: str) -> str:
    """Stemmed, model-ready string for TF-IDF."""
    text = str(text)
    text = _URL_RE.sub(" ", text)
    for pat, repl in _CONTRACTIONS:
        text = re.sub(pat, repl, text)
    text = _HTML_RE.sub(" ", text)
    text = _DIGIT_RE.sub(" ", text)
    text = _EMOJI_RE.sub(" ", text)
    text = re.sub(r"@\S+", " ", text)
    text = _ALPHA_RE.sub(" ", text)
    tokens = [
        _stemmer.stem(w)
        for w in text.lower().split()
        if w not in _NLTK_STOP and len(w) > 2
    ]
    return " ".join(tokens)


def clean_text(text: str) -> str:
    """Lowercased, URL-stripped string — basis for lemmatized tokens (SHAP).

    Raises:
        ValueError: When *text* is empty or whitespace-only.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    text = str(text).lower()
    text = _URL_RE.sub("", text)
    text = _SPEC_RE.sub("", text)
    return _SPACE_RE.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    """Lemmatized tokens; used for vocabulary analysis and SHAP explanations."""
    return [
        _lemmatizer.lemmatize(t)
        for t in clean_text(text).split()
        if t not in _ALL_STOP and len(t) > 2
    ]


def make_preprocessor() -> FunctionTransformer:
    """sklearn-compatible transformer wrapping preprocess_data."""
    return FunctionTransformer(
        lambda series: series.map(preprocess_data),
        validate=False,
    )
