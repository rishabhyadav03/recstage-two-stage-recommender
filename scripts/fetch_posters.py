"""Fetch movie posters, backdrops, and overviews from TMDB for MovieLens 1M.

Features:
- Title cleaning: Extracts year, handles trailing articles (The, A, An, Le, La, Les, L', Il, Lo, El, Der, Die, Das, etc.),
  splits off parenthetical alternate/foreign titles, and strips punctuation variants.
- Search strategies (in strict order, stopping at first accepted match):
  1. /search/movie with query + primary_release_year
  2. /search/movie with year - 1 and year + 1
  3. /search/movie without year (accepts only if release year within +/- 1)
  4. alternate/foreign titles with steps 1, 2, 3
  5. /search/multi as last resort
- Match acceptance: Normalized string comparison with RapidFuzz ratio (threshold ~85)
  against both title and original_title. Prefers exact year, then higher popularity.
- Rate-limit and error safe: Exponential backoff on 429 (respects Retry-After), timeouts, and 5xx.
  Never caches transient failures.
- Resumable: Atomically saves progress to data/posters.json.
"""
import concurrent.futures
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from rapidfuzz import fuzz

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fetch_posters")

TMDB_BASE_URL = "https://api.themoviedb.org/3"
DATA_DIR = Path("data")
POSTERS_FILE = DATA_DIR / "posters.json"
MOVIES_FILE = Path("data/raw/ml-1m/movies.dat")
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"

ARTICLE_REGEX = re.compile(
    r"^(.*?),\s*(The|A|An|Le|La|Les|L\'|Il|Lo|I|Gli|El|Los|Las|Der|Die|Das)(?:\s*)$",
    re.IGNORECASE,
)


def load_env_key() -> Optional[str]:
    """Retrieve TMDB_API_KEY from os.environ or .env file."""
    key = os.environ.get("TMDB_API_KEY")
    if key and key.strip():
        return key.strip()

    if ENV_FILE.exists():
        try:
            with open(ENV_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        if k.strip() == "TMDB_API_KEY":
                            val = v.strip().strip("\"'").strip()
                            if val:
                                return val
        except Exception as e:
            logger.warning(f"Error reading {ENV_FILE}: {e}")

    return None


def move_article(s: str) -> str:
    """Move trailing article to front: 'Bug\'s Life, A' -> 'A Bug\'s Life'."""
    s = s.strip()
    m = ARTICLE_REGEX.match(s)
    if m:
        base = m.group(1).strip()
        art = m.group(2).strip()
        if art.endswith("'"):
            return f"{art}{base}"
        return f"{art} {base}"
    return s


def clean_title(raw_title: str) -> Tuple[str, Optional[int], List[str]]:
    """Clean title: extract year, alternate/foreign titles, and relocate articles."""
    raw = raw_title.strip()
    year = None

    # 1. Extract trailing year (YYYY)
    m_year = re.search(r"\((\d{4})\)\s*$", raw)
    if m_year:
        year = int(m_year.group(1))
        t_without_year = raw[: m_year.start()].strip()
    else:
        t_without_year = raw

    # 2. Extract parenthetical alternate/foreign titles
    raw_alts = []

    def extract_paren(match):
        content = match.group(1).strip()
        if content:
            raw_alts.append(content)
        return ""

    t_main_raw = re.sub(r"\(([^)]+)\)", extract_paren, t_without_year).strip()
    t_main_raw = re.sub(r"\s+", " ", t_main_raw).strip()

    main_title = move_article(t_main_raw)

    cleaned_alts = []
    # If main starts with an article, also add version without article
    for prefix in ("The ", "A ", "An "):
        if main_title.startswith(prefix):
            without_art = main_title[len(prefix):].strip()
            if without_art and without_art not in cleaned_alts:
                cleaned_alts.append(without_art)

    # Process all parenthetical items, splitting on commas or 'a.k.a.'
    for raw_alt in raw_alts:
        parts = re.split(r",|\ba\.k\.a\.\b|/|;", raw_alt, flags=re.IGNORECASE)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            moved = move_article(part)
            if moved and moved.lower() != main_title.lower() and moved not in cleaned_alts:
                cleaned_alts.append(moved)
            for prefix in ("The ", "A ", "An "):
                if moved.startswith(prefix):
                    no_p = moved[len(prefix):].strip()
                    if no_p and no_p not in cleaned_alts:
                        cleaned_alts.append(no_p)

    return main_title, year, cleaned_alts



import unicodedata

def strip_accents(s: str) -> str:
    """Remove diacritics/accents from characters: 'Prêt-à-Porter' -> 'Pret-a-Porter'."""
    if not s:
        return ""
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def normalize_title(s: str) -> str:
    """Normalize string for fuzzy comparison: strip accents, lowercase, strip punctuation, collapse whitespace."""
    if not s:
        return ""
    s = strip_accents(s)
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()


class TMDBClient:
    """HTTP client with retry, backoff, and rate-limit handling for TMDB."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "RecStage-MovieLens/1.0",
            "Accept": "application/json",
        })

    def request(self, endpoint: str, params: Dict[str, Any], max_retries: int = 5) -> Tuple[Optional[List[Dict[str, Any]]], bool]:
        """Make TMDB API request with exponential backoff on transient errors.

        Returns (results_list, is_transient_error).
        """
        p = dict(params)
        p["api_key"] = self.api_key
        url = f"{TMDB_BASE_URL}{endpoint}"

        for attempt in range(max_retries):
            try:
                resp = self.session.get(url, params=p, timeout=(5.0, 10.0))
                if resp.status_code == 200:
                    data = resp.json()
                    return data.get("results", []), False
                elif resp.status_code == 429:
                    retry_after = float(resp.headers.get("Retry-After", str(1.0 + attempt * 0.5)))
                    time.sleep(retry_after)
                    continue
                elif resp.status_code in (500, 502, 503, 504):
                    time.sleep(0.5 * (2**attempt))
                    continue
                else:
                    # 400, 401, 404 - permanent failure
                    return [], False
            except (requests.Timeout, requests.ConnectionError):
                time.sleep(0.5 * (2**attempt))
                continue
            except Exception as e:
                logger.debug(f"Request exception: {e}")
                time.sleep(0.5 * (2**attempt))
                continue

        # All retries exhausted due to network/server errors
        return None, True


NUM_WORDS = {
    "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
    "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten",
    "11": "eleven", "12": "twelve", "13": "thirteen", "14": "fourteen",
    "15": "fifteen", "16": "sixteen", "17": "seventeen", "18": "eighteen",
    "19": "nineteen", "20": "twenty", "21": "twenty one", "24": "twenty four",
    "50": "fifty", "100": "one hundred", "101": "one hundred and one",
    "1000": "one thousand",
}

ROMAN_NUMS = {
    "1": "I", "2": "II", "3": "III", "4": "IV", "5": "V",
    "6": "VI", "7": "VII", "8": "VIII", "9": "IX", "10": "X",
}


def expand_number_variants(title: str) -> List[str]:
    """Expand digit/word/roman/fraction variants: '101' <-> 'one hundred and one', '8 1/2' <-> '8½', etc."""
    variants = [title]

    # Fraction variants
    if "1/2" in title:
        v_half = title.replace("1/2", "½")
        if v_half not in variants:
            variants.append(v_half)
    if "½" in title:
        v_frac = title.replace("½", "1/2")
        if v_frac not in variants:
            variants.append(v_frac)

    # Accent-stripped variant
    unaccented = strip_accents(title)
    if unaccented != title and unaccented not in variants:
        variants.append(unaccented)

    # Digits to words
    t_words = title
    for num, word in sorted(NUM_WORDS.items(), key=lambda x: -len(x[0])):
        pattern = r"\b" + re.escape(num) + r"\b"
        if re.search(pattern, t_words, flags=re.I):
            t_words = re.sub(pattern, word, t_words, flags=re.I)
    if t_words.lower() != title.lower() and t_words not in variants:
        variants.append(t_words)

    if "101" in title:
        v101 = re.sub(r"\b101\b", "one hundred and one", title, flags=re.I)
        if v101 not in variants:
            variants.append(v101)
        v101_alt = re.sub(r"\b101\b", "one hundred one", title, flags=re.I)
        if v101_alt not in variants:
            variants.append(v101_alt)

    # Words to digits
    t_digits = title
    for num, word in sorted(NUM_WORDS.items(), key=lambda x: -len(x[1])):
        pattern = r"\b" + re.escape(word) + r"\b"
        if re.search(pattern, t_digits, flags=re.I):
            t_digits = re.sub(pattern, num, t_digits, flags=re.I)
    if t_digits.lower() != title.lower() and t_digits not in variants:
        variants.append(t_digits)

    # Roman numerals <-> Arabic digits
    for arabic, roman in ROMAN_NUMS.items():
        p_rom = r"\b" + re.escape(roman) + r"\b"
        if re.search(p_rom, title):
            v_ara = re.sub(p_rom, arabic, title)
            if v_ara not in variants:
                variants.append(v_ara)
        p_ara = r"\b" + re.escape(arabic) + r"\b"
        if re.search(p_ara, title):
            v_rom = re.sub(p_ara, roman, title)
            if v_rom not in variants:
                variants.append(v_rom)

    return variants

    return variants


def evaluate_match(
    target_title: str,
    target_year: Optional[int],
    candidate: Dict[str, Any],
    threshold: float = 85.0,
) -> Tuple[bool, float, int]:
    """Check if candidate satisfies fuzzy match and year criteria.

    Returns: (is_accepted, fuzzy_score, year_diff)
    """
    c_title = candidate.get("title") or ""
    c_orig = candidate.get("original_title") or ""
    rel_date = candidate.get("release_date") or ""
    c_year = int(rel_date[:4]) if len(rel_date) >= 4 and rel_date[:4].isdigit() else None

    # Number-expanded variants check
    target_variants = expand_number_variants(target_title)
    cand_variants = expand_number_variants(c_title) + expand_number_variants(c_orig)

    best_score = 0.0
    for tv in target_variants:
        nt = normalize_title(tv)
        for cv in cand_variants:
            nc = normalize_title(cv)
            s1 = fuzz.ratio(nt, nc)
            s2 = fuzz.token_sort_ratio(nt, nc)
            len_ratio = min(len(nt), len(nc)) / max(len(nt), len(nc), 1)
            s3 = fuzz.token_set_ratio(nt, nc) if len_ratio >= 0.65 else 0.0
            cand_score = max(s1, s2, s3)
            if cand_score > best_score:
                best_score = cand_score

    # Safe partial ratio for colon/franchise subtitles when length ratio is reasonable
    if best_score < threshold:
        norm_target = normalize_title(target_title)
        norm_c_title = normalize_title(c_title)
        norm_c_orig = normalize_title(c_orig)
        for cand_norm in (norm_c_title, norm_c_orig):
            if not cand_norm:
                continue
            if (norm_target in cand_norm or cand_norm in norm_target):
                len_ratio = min(len(norm_target), len(cand_norm)) / max(len(norm_target), len(cand_norm), 1)
                if len_ratio >= 0.65 and len(norm_target) >= 5:
                    part_score = fuzz.partial_ratio(norm_target, cand_norm)
                    if part_score >= 90.0:
                        best_score = max(best_score, 88.0)

    if best_score < threshold:
        return False, best_score, 999

    # Year check with tiered tolerance based on fuzzy confidence:
    # International distribution delays (UK/Japan/France to US theatrical release in MovieLens)
    # often differ by 2 to 3 years.
    max_year_diff = 1
    if best_score >= 95.0:
        max_year_diff = 3
    elif best_score >= 90.0:
        max_year_diff = 2

    year_diff = 0
    if target_year is not None and c_year is not None:
        year_diff = abs(c_year - target_year)
        if year_diff > max_year_diff:
            return False, best_score, year_diff
    elif target_year is not None and c_year is None:
        year_diff = 2

    return True, best_score, year_diff



def pick_best_accepted(
    results: List[Dict[str, Any]],
    target_title: str,
    target_year: Optional[int],
    threshold: float = 85.0,
) -> Optional[Dict[str, Any]]:
    """Filter candidates by acceptance criteria and rank by year, score, popularity."""
    accepted = []
    for cand in results:
        is_ok, score, y_diff = evaluate_match(target_title, target_year, cand, threshold=threshold)
        if is_ok:
            has_poster = 1 if cand.get("poster_path") else 0
            pop = float(cand.get("popularity", 0.0) or 0.0)
            votes = int(cand.get("vote_count", 0) or 0)
            # Tuple for sorting: has_poster (desc), exact_year (desc), score (desc), popularity (desc)
            exact_year = 1 if y_diff == 0 else 0
            rank_key = (has_poster, exact_year, -y_diff, score, pop, votes)
            accepted.append((rank_key, cand))

    if not accepted:
        return None

    accepted.sort(key=lambda x: x[0], reverse=True)
    return accepted[0][1]


def search_tmdb_all_strategies(
    client: TMDBClient,
    raw_title: str,
) -> Tuple[Optional[Dict[str, Any]], bool]:
    """Execute search strategies in strict order.

    Prioritizes matches with actual posters; only falls back to posterless matches
    if all strategies are exhausted.
    """
    main_title, year, alts = clean_title(raw_title)

    # Build queries to test: [main_title] + [alt_titles] + [stripped punctuation] + [number variants]
    raw_queries = [main_title]
    for alt in alts:
        if alt not in raw_queries:
            raw_queries.append(alt)

    # Punctuation stripped variant
    no_punct = re.sub(r"[^\w\s]", " ", main_title)
    no_punct = re.sub(r"\s+", " ", no_punct).strip()
    if no_punct and no_punct.lower() != main_title.lower() and no_punct not in raw_queries:
        raw_queries.append(no_punct)

    queries_to_try = []
    for q in raw_queries:
        for v in expand_number_variants(q):
            if v not in queries_to_try:
                queries_to_try.append(v)

    had_transient_error = False
    best_fallback = None

    def check_results(res_list: List[Dict[str, Any]], q_str: str) -> Optional[Dict[str, Any]]:
        nonlocal best_fallback
        best = pick_best_accepted(res_list, q_str, year)
        if best:
            if best.get("poster_path"):
                return best
            elif best_fallback is None:
                best_fallback = best
        return None

    # Strategy 1 & 2 & 3 for each query
    for q in queries_to_try:
        # 1. Exact year
        if year:
            res, trans = client.request("/search/movie", {"query": q, "primary_release_year": year, "include_adult": "false"})
            if trans:
                had_transient_error = True
            elif res:
                winner = check_results(res, q)
                if winner:
                    return winner, False

        # 2. Year +/- 1
        if year:
            for y_off in (-1, 1):
                res, trans = client.request("/search/movie", {"query": q, "primary_release_year": year + y_off, "include_adult": "false"})
                if trans:
                    had_transient_error = True
                elif res:
                    winner = check_results(res, q)
                    if winner:
                        return winner, False

        # 3. Query without year
        res, trans = client.request("/search/movie", {"query": q, "include_adult": "false"})
        if trans:
            had_transient_error = True
        elif res:
            winner = check_results(res, q)
            if winner:
                return winner, False

    # 4. /search/multi as last resort
    res, trans = client.request("/search/multi", {"query": main_title, "include_adult": "false"})
    if trans:
        had_transient_error = True
    elif res:
        movie_res = [r for r in res if r.get("media_type") == "movie"]
        winner = check_results(movie_res, main_title)
        if winner:
            return winner, False

    if best_fallback is not None:
        return best_fallback, False

    return None, had_transient_error


def format_poster_entry(
    movie_id: int,
    raw_title: str,
    tmdb_item: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Format final output record for posters.json."""
    main_title, year, _ = clean_title(raw_title)
    if tmdb_item is None:
        return {
            "movie_id": movie_id,
            "clean_title": main_title,
            "year": year,
            "tmdb_id": None,
            "tmdb_title": None,
            "poster_path": None,
            "poster_url": None,
            "backdrop_path": None,
            "overview": "",
            "release_year": str(year or ""),
            "vote_average": None,
            "popularity": None,
        }

    p_path = tmdb_item.get("poster_path")
    b_path = tmdb_item.get("backdrop_path")
    p_url = f"https://image.tmdb.org/t/p/w342{p_path}" if p_path else None
    rel_date = tmdb_item.get("release_date") or ""
    rel_year = rel_date[:4] if len(rel_date) >= 4 else str(year or "")

    return {
        "movie_id": movie_id,
        "clean_title": main_title,
        "year": year,
        "tmdb_id": tmdb_item.get("id"),
        "tmdb_title": tmdb_item.get("title"),
        "poster_path": p_path,
        "poster_url": p_url,
        "backdrop_path": b_path,
        "overview": tmdb_item.get("overview") or "",
        "release_year": rel_year,
        "vote_average": tmdb_item.get("vote_average"),
        "popularity": tmdb_item.get("popularity"),
    }


def save_posters_cache(cache: Dict[str, Any]) -> None:
    """Atomically write cache to posters.json."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp_file = POSTERS_FILE.with_suffix(".tmp")
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    tmp_file.replace(POSTERS_FILE)


def process_single_movie(
    item: Tuple[int, str],
    client: TMDBClient,
) -> Tuple[int, Optional[Dict[str, Any]], bool]:
    """Worker task for processing one movie."""
    movie_id, raw_title = item
    best_match, had_transient = search_tmdb_all_strategies(client, raw_title)
    if had_transient and best_match is None:
        # Transient failure - do NOT cache
        return movie_id, None, True

    entry = format_poster_entry(movie_id, raw_title, best_match)
    return movie_id, entry, False


def main():
    api_key = load_env_key()
    if not api_key:
        logger.error("TMDB_API_KEY is not set in environment or .env file.")
        sys.exit(1)

    if not MOVIES_FILE.exists():
        logger.error(f"Movies file not found at {MOVIES_FILE}")
        sys.exit(1)

    # 1. Load existing cache
    cache: Dict[str, Any] = {}
    if POSTERS_FILE.exists():
        try:
            with open(POSTERS_FILE, "r", encoding="utf-8") as f:
                cache = json.load(f)
            logger.info(f"Loaded existing cache with {len(cache)} entries.")
        except Exception as e:
            logger.warning(f"Failed to read existing {POSTERS_FILE}: {e}")

    # 2. Load movies catalog
    movies_df = pd.read_csv(
        MOVIES_FILE,
        sep="::",
        engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
    )
    total_catalog = len(movies_df)
    logger.info(f"Loaded {total_catalog} movies from {MOVIES_FILE}")

    # 3. Identify movies to fetch
    # We re-fetch if:
    #  - Not in cache
    #  - poster_path starts with 'http' (old IMDb/Amazon fallback that 404s)
    #  - poster_path is None or missing
    #  - tmdb_id is None
    pending = []
    already_clean = 0
    for _, row in movies_df.iterrows():
        mid = int(row["movie_id"])
        c_entry = cache.get(str(mid))
        if c_entry and c_entry.get("tmdb_id") and c_entry.get("poster_path") and not str(c_entry.get("poster_path")).startswith("http"):
            already_clean += 1
        else:
            pending.append((mid, str(row["title"])))

    logger.info(f"Movies already have pristine TMDB posters: {already_clean}")
    logger.info(f"Movies to process/upgrade: {len(pending)}")

    client = TMDBClient(api_key)

    if pending:
        batch_size = 50
        num_workers = 8  # Safe concurrency well within TMDB rate limits
        processed = 0

        for i in range(0, len(pending), batch_size):
            batch = pending[i : i + batch_size]
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(process_single_movie, item, client): item[0]
                    for item in batch
                }
                for fut in concurrent.futures.as_completed(futures):
                    mid, entry, is_transient = fut.result()
                    if is_transient:
                        logger.warning(f"Movie {mid} experienced transient failure; skipping cache update to retry later.")
                    else:
                        cache[str(mid)] = entry

            processed += len(batch)
            save_posters_cache(cache)

            matched_posters = sum(
                1 for v in cache.values() if v and v.get("poster_path") and not str(v.get("poster_path")).startswith("http")
            )
            pct = (matched_posters / len(cache) * 100) if cache else 0
            logger.info(
                f"Batch {processed}/{len(pending)} done | Total cached: {len(cache)} | TMDB posters: {matched_posters} ({pct:.1f}%)"
            )

        save_posters_cache(cache)

    # 4. Final summary report
    matched_entries = [v for v in cache.values() if v and v.get("tmdb_id")]
    with_poster = [v for v in cache.values() if v and v.get("poster_path")]
    total_processed = len(cache)
    match_rate = (len(with_poster) / total_processed * 100) if total_processed > 0 else 0

    print("\n" + "=" * 60)
    print("TMDB MOVIE POSTERS MATCH-RATE REPORT")
    print("=" * 60)
    print(f"Total Movies in Catalog : {total_catalog}")
    print(f"Total Movies in Cache   : {total_processed}")
    print(f"Matched on TMDB         : {len(matched_entries)}")
    print(f"Matches With Poster     : {len(with_poster)}")
    print(f"Match Rate              : {match_rate:.2f}%")
    print("=" * 60)

    print("\nSample Key Titles Inspection:")
    test_ids = [1, 588, 1358, 260, 527, 858, 1196, 1210, 2987, 3114]
    for mid in test_ids:
        entry = cache.get(str(mid))
        if entry:
            t = entry.get("clean_title")
            yr = entry.get("release_year") or entry.get("year")
            tid = entry.get("tmdb_id")
            p = entry.get("poster_path")
            p_url = entry.get("poster_url")
            print(f"  [{mid:>4}] {t} ({yr}) -> TMDB #{tid} | Path: {p} | URL: {p_url}")
        else:
            print(f"  [{mid:>4}] Not in cache")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
