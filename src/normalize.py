"""Country-agnostic text normalisation for business names and addresses.

Only hand-written rules (abbreviation / legal-suffix / state-name maps) are used: no external
lookups, gazetteers or APIs, in line with the challenge's fair-play rules.

Design notes
- Accents are removed with Unicode NFKD (stdlib), so French "Société" == "Societe".
- Abbreviations are mapped to ONE canonical short form on both sides
  (Corporation/Corp -> corp, Street/St -> st, ...), so exact comparisons work after mapping.
- Legal-suffix tokens (pvt, ltd, llc, inc, sarl, sas, ...) are stripped to form the "core" name,
  and kept separately as a legal signature feature.
- DBA / trading-as names are split into variants so either part can be matched.
"""
import re
import unicodedata

# --------------------------------------------------------------------------- generic cleaning
_RE_SPACES = re.compile(r"\s+")
_RE_ACR_DOTS = re.compile(r"\b((?:[a-z]\.)+[a-z])\.?(?=\W|$)")        # p.v.t. -> pvt, u.s.a -> usa
_RE_APOS = re.compile(r"(?<=[a-z])['\u2019`\u00b4](?=[a-z])")          # mcdonald's -> mcdonalds
_RE_ORD = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b")                      # 21st -> 21
_RE_NONWORD = re.compile(r"[^\w\s]|_")
_RE_LETTER_DIGIT = re.compile(r"(?<=[^\W\d_])(?=\d)|(?<=\d)(?=[^\W\d_])")  # b12 -> b 12
_RE_PIN_SPLIT = re.compile(r"\b([1-9]\d{2}) (\d{3})\b")                # "226 010" -> "226010"
_SLASH_RULES = [
    (re.compile(r"\bd\s*/\s*b\s*/\s*a\b"), " dba "),
    (re.compile(r"\bt\s*/\s*a\b"), " dba "),
    (re.compile(r"\ba\s*/\s*k\s*/\s*a\b"), " aka "),
    (re.compile(r"\bm\s*/\s*s\b"), " "),            # Indian "M/s." (Messrs) prefix
    (re.compile(r"\bc\s*/\s*o\b"), " careof "),
    (re.compile(r"\b[sdw]\s*/\s*o\b"), " "),        # s/o, d/o, w/o
]


def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def clean_base(s) -> str:
    """Lower-case, strip accents, unify punctuation; returns space-separated tokens."""
    if s is None:
        return ""
    s = str(s)
    if not s or s.lower() in ("nan", "none", "null"):
        return ""
    s = strip_accents(s).lower()
    s = s.replace("&", " and ").replace("@", " at ")
    s = _RE_ACR_DOTS.sub(lambda m: m.group(1).replace(".", ""), s)
    s = _RE_APOS.sub("", s)
    for rx, rep in _SLASH_RULES:
        s = rx.sub(rep, s)
    s = _RE_ORD.sub(r"\1", s)
    s = _RE_NONWORD.sub(" ", s)
    s = _RE_LETTER_DIGIT.sub(" ", s)
    return _RE_SPACES.sub(" ", s).strip()


def _compile_phrases(d):
    keys = sorted(d, key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(re.escape(k) for k in keys) + r")\b")


def _apply_phrases(s, rx, d):
    return _RE_SPACES.sub(" ", rx.sub(lambda m: " " + d[m.group(0)] + " ", s)).strip()


# --------------------------------------------------------------------------- names
NAME_PHRASES = {
    "private limited": "pvt ltd",
    "limited liability company": "llc",
    "limited liability co": "llc",
    "limited liability partnership": "llp",
    "public limited company": "plc",
    "one person company": "opc",
    "doing business as": "dba",
    "trading as": "dba",
    "also known as": "aka",
    "formerly known as": "aka",
    "societe anonyme": "sa",
    "societe par actions simplifiee unipersonnelle": "sasu",
    "societe par actions simplifiee": "sas",
    "societe a responsabilite limitee": "sarl",
    "entreprise unipersonnelle a responsabilite limitee": "eurl",
    "societe en nom collectif": "snc",
    "societe civile immobiliere": "sci",
    "et compagnie": "cie",
    "et cie": "cie",
    "and company": "co",
}
NAME_TOKENS = {
    "corporation": "corp", "corpn": "corp", "corp": "corp",
    "incorporated": "inc", "incorp": "inc",
    "limited": "ltd", "lmtd": "ltd", "ltda": "ltd", "ltd": "ltd",
    "private": "pvt", "pvt": "pvt", "pte": "pvt", "prv": "pvt", "pvtltd": "pvt ltd",
    "company": "co", "compagnie": "cie",
    "et": "and",
    "brothers": "bros", "bro": "bros",
    "international": "intl", "intnl": "intl",
    "manufacturing": "mfg", "manufacturers": "mfg", "mfrs": "mfg",
    "enterprises": "ent", "enterprise": "ent", "entp": "ent", "ents": "ent",
    "industries": "ind", "industry": "ind", "inds": "ind", "indus": "ind",
    "associates": "assoc", "association": "assn",
    "technologies": "tech", "technology": "tech", "technologie": "tech",
    "services": "svc", "service": "svc", "svcs": "svc", "srvcs": "svc",
    "solutions": "soln", "solution": "soln",
    "systems": "sys", "system": "sys",
    "department": "dept", "national": "natl", "center": "ctr", "centre": "ctr",
    "saint": "st", "sainte": "ste", "mount": "mt",
    "pharmaceuticals": "pharma", "pharmaceutical": "pharma",
    "laboratories": "labs", "laboratory": "labs", "lab": "labs",
    "hospital": "hosp", "university": "univ", "institute": "inst",
    "consultants": "consult", "consultancy": "consult", "consulting": "consult",
    "communications": "comm", "communication": "comm",
    "electricals": "elec", "electrical": "elec", "electronics": "elect",
    "products": "prod", "product": "prod",
    "trading": "trdg", "traders": "trdrs",
}
LEGAL = {"pvt", "ltd", "llc", "llp", "inc", "corp", "co", "plc", "lp", "lllp", "pllc", "pc", "opc",
         "sa", "sas", "sasu", "sarl", "eurl", "snc", "sci", "scp", "selarl", "sca", "cie", "gmbh"}
NAME_STOP = {"the", "messrs", "le", "la", "les"}
_NAME_PH_RX = _compile_phrases(NAME_PHRASES)


def _core(toks):
    c = [t for t in toks if t not in LEGAL and t not in NAME_STOP]
    while c and c[0] == "and":
        c.pop(0)
    while c and c[-1] == "and":
        c.pop()
    return c if c else list(toks)


def process_name(raw):
    """Returns (norm, core, variants, legal_signature, acronym)."""
    s = clean_base(raw)
    s = _apply_phrases(s, _NAME_PH_RX, NAME_PHRASES)
    toks = " ".join(NAME_TOKENS.get(t, t) for t in s.split()).split()
    norm = " ".join(toks)
    parts, cur = [], []
    for t in toks:
        if t in ("dba", "aka"):
            if cur:
                parts.append(cur)
            cur = []
        else:
            cur.append(t)
    if cur:
        parts.append(cur)
    all_toks = [t for t in toks if t not in ("dba", "aka")]
    core_toks = _core(all_toks)
    core = " ".join(core_toks)
    variants = "|".join(" ".join(_core(p)) for p in parts) if len(parts) > 1 else ""
    legal = " ".join(sorted({t for t in all_toks if t in LEGAL}))
    acr = "".join(t[0] for t in core_toks if t[:1].isalpha()) if len(core_toks) >= 2 else ""
    return norm, core, variants, legal, acr


# --------------------------------------------------------------------------- addresses
US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca", "colorado": "co",
    "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la",
    "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
    "rhode island": "ri", "south carolina": "sc", "south dakota": "sd", "tennessee": "tn", "texas": "tx",
    "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa", "west virginia": "wv",
    "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc",
}
INDIA_STATES = {
    "uttar pradesh": "up", "madhya pradesh": "mp", "andhra pradesh": "ap", "himachal pradesh": "hp",
    "arunachal pradesh": "arp", "tamil nadu": "tn", "tamilnadu": "tn", "west bengal": "wb",
    "jammu and kashmir": "jk", "jammu kashmir": "jk", "maharashtra": "mh", "karnataka": "ka",
    "gujarat": "gj", "rajasthan": "rj", "punjab": "pb", "haryana": "hr", "bihar": "br", "odisha": "od",
    "orissa": "od", "kerala": "kl", "telangana": "tg", "jharkhand": "jh", "chhattisgarh": "cg",
    "chattisgarh": "cg", "uttarakhand": "uk", "uttaranchal": "uk", "new delhi": "delhi",
    "nct of delhi": "delhi", "pondicherry": "puducherry",
    # common transliteration / old-name variants of large cities
    "bangalore": "bengaluru", "bombay": "mumbai", "madras": "chennai", "calcutta": "kolkata",
    "gurgaon": "gurugram", "poona": "pune", "baroda": "vadodara", "trivandrum": "thiruvananthapuram",
    "allahabad": "prayagraj", "benaras": "varanasi", "banaras": "varanasi", "cawnpore": "kanpur",
}
ADDR_PHRASES = {
    **US_STATES, **INDIA_STATES,
    "post office": "po", "p o box": "pobox", "po box": "pobox", "post box": "pobox", "boite postale": "pobox",
    "united states of america": "", "united states": "", "republic of india": "",
    "in front of": "opp", "next to": "nr", "close to": "nr", "near by": "nr", "near to": "nr",
}
ADDR_TOKENS = {
    "street": "st", "str": "st", "strt": "st", "road": "rd", "raod": "rd", "avenue": "ave", "av": "ave",
    "avn": "ave", "boulevard": "blvd", "bd": "blvd", "boul": "blvd", "blvrd": "blvd", "drive": "dr",
    "drv": "dr", "lane": "ln", "court": "ct", "crt": "ct", "place": "pl", "plaza": "plz", "square": "sq",
    "highway": "hwy", "hiway": "hwy", "expressway": "expy", "parkway": "pkwy", "terrace": "ter",
    "circle": "cir", "trail": "trl", "suite": "ste", "apartment": "apt", "appt": "apt", "floor": "fl",
    "flr": "fl", "building": "bldg", "bldng": "bldg", "bld": "bldg", "room": "rm", "number": "no",
    "num": "no", "north": "n", "south": "s", "east": "e", "west": "w", "northeast": "ne",
    "northwest": "nw", "southeast": "se", "southwest": "sw", "near": "nr", "nearby": "nr",
    "opposite": "opp", "opp": "opp", "opps": "opp", "behind": "bhd", "beside": "bsd", "sector": "sec",
    "sect": "sec", "phase": "ph", "block": "blk", "colony": "col", "extension": "extn", "ext": "extn",
    "enclave": "encl", "industrial": "indl", "estate": "est", "district": "dist", "distt": "dist",
    "taluka": "tal", "taluk": "tal", "tehsil": "teh", "village": "vill", "vil": "vill", "post": "po",
    "nagr": "nagar", "ngr": "nagar", "saint": "st", "sainte": "ste", "mount": "mt", "fort": "ft",
    "point": "pt", "centre": "ctr", "center": "ctr", "chemin": "ch", "route": "rte", "impasse": "imp",
    "allee": "all", "quai": "qu", "faubourg": "fbg", "residence": "res", "batiment": "bat", "etage": "etg",
    "galli": "gali", "mohalla": "mohalla", "mohlla": "mohalla", "marg": "marg", "mrg": "marg",
    "chowk": "chowk", "chauraha": "chowk", "bazaar": "bazar", "bazar": "bazar", "market": "mkt",
    "mkt": "mkt", "cedex": "cedex",
}
ADDR_DROP = {"no", "usa", "us", "india", "france", "bharat", "careof"}
LANDMARK = {"nr", "opp", "bhd", "bsd", "landmark", "adjacent", "adj"}
_ADDR_PH_RX = _compile_phrases(ADDR_PHRASES)


def process_address(raw):
    """Returns (norm, numbers, postal, first_number, tail, landmark_flag)."""
    s = clean_base(raw)
    s = _RE_PIN_SPLIT.sub(r"\1\2", s)
    s = _apply_phrases(s, _ADDR_PH_RX, ADDR_PHRASES)
    toks = [ADDR_TOKENS.get(t, t) for t in s.split()]
    toks = [t for t in toks if t and t not in ADDR_DROP]
    norm = " ".join(toks)
    nums = [t for t in toks if any(ch.isdigit() for ch in t)]
    postal = sorted({t for t in nums if t.isdigit() and 5 <= len(t) <= 6})
    first_num = next((t for t in nums if t not in postal), "")
    words = [t for t in toks if not any(ch.isdigit() for ch in t) and t not in LANDMARK]
    tail = " ".join(words[-3:])
    landmark = int(any(t in LANDMARK for t in toks))
    return norm, " ".join(dict.fromkeys(nums)), " ".join(postal), first_num, tail, landmark


# --------------------------------------------------------------------------- transliteration skeleton
_SKEL_SUBS = [("ph", "f"), ("sh", "s"), ("kh", "k"), ("gh", "g"), ("bh", "b"), ("dh", "d"), ("th", "t"),
              ("ch", "c"), ("ck", "k"), ("q", "k"), ("w", "v"), ("z", "s"), ("x", "ks"), ("y", "i")]
_RE_REPEAT = re.compile(r"(.)\1+")
_RE_VOWEL = re.compile(r"[aeiou]")


def skeleton(text: str) -> str:
    """Rough phonetic key robust to Indian/French transliteration variants (Laxmi~Lakshmi)."""
    out = []
    for tok in text.split():
        t = tok
        for a, b in _SKEL_SUBS:
            t = t.replace(a, b)
        t = _RE_REPEAT.sub(r"\1", t)
        if len(t) > 1:
            t = t[0] + _RE_VOWEL.sub("", t[1:])
        out.append(t)
    return " ".join(out)


COUNTRY_ALIASES = {"USA": "US", "UNITED STATES": "US", "UNITED STATES OF AMERICA": "US", "U S": "US",
                   "IN": "INDIA", "IND": "INDIA", "BHARAT": "INDIA", "FR": "FRANCE", "FRA": "FRANCE"}


def country_key(raw) -> str:
    s = clean_base(raw).upper()
    return COUNTRY_ALIASES.get(s, s)
