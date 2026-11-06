"""
YOUR PIPELINE · the only file you edit today
=============================================
The app (challenge/app.py) calls these functions. You build them level by level, with Claude Code
or Kiro, and press "Reload pipeline" in the app after every change. The status box at the top of
the app shows which levels work.

  LEVEL 1 · Ingest   load_cvs, chunk          → the 📄 Data tab shows your CVs and chunks
  LEVEL 2 · Index    build_index (Chroma)     → the status shows how many chunks are stored
  LEVEL 3 · Answer   retrieve, build_prompt   → the 💬 Chat works: this is NAIVE RAG.
                                                Run ✅ Verify: that is your first score.
  LEVEL 4 · Features make_query, retrieve, should_refuse, build_graph … (see challenge/README.md)

Given to you (helpers.py):  embed(texts) -> normalized vectors · llm(system, user) -> answer text
Stuck or behind?  cp challenge/checkpoints/level3.py challenge/pipeline.py   (naive RAG, done)
"""
import contextlib
import io
import pathlib
import re
import time

import chromadb
import networkx as nx
from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph
from pypdf import PdfReader

from helpers import embed, llm
from ragkit.chunking import Chunk
from ragkit.retrieval import BM25Index, Reranker, tokenize

HERE = pathlib.Path(__file__).resolve().parent
K = 3                     # chunks per prompt
REFUSAL = "I don't know based on the CVs."
SYSTEM_PROMPT = ("You are a recruiting assistant. Answer ONLY from the numbered CV passages. "
                 "Each passage belongs to the candidate named in its header: state a fact about a candidate only "
                 "if it is in that candidate's own passage. If passages disagree, or it is unclear which candidate "
                 "is meant, say so instead of guessing. "
                 "Cite every fact like [1]. If the passages do not contain the answer, reply exactly: "
                 f"\"{REFUSAL}\" Be brief.")
BASE_SYSTEM_PROMPT = SYSTEM_PROMPT
# app.py reads pipeline.SYSTEM_PROMPT when it calls the LLM, right after build_prompt(): for an ambiguous
# follow-up build_prompt switches to this one (the base prompt makes a small model refuse instead of asking);
# make_query() and build_prompt() switch back at the start of every turn.
CLARIFY_SYSTEM_PROMPT = ("You are a recruiting assistant. The user's question is ambiguous. "
                         "Reply with exactly the sentence given after 'Reply:' and nothing else.")


# ================================================================= LEVEL 1 · INGEST
# Prompt idea: "In challenge/pipeline.py implement load_cvs and chunk as their docstrings say.
#  Use pypdf for PDF and python-docx for DOCX. Start with fixed-size chunks of 800 characters
#  with 100 overlap. Don't change other files."
CV_TYPES = (".pdf", ".docx", ".txt", ".md")


def docx_lines(parent, doc):
    """The text of paragraphs AND tables under `parent`, in document order (tables inside cells too)."""
    lines = []
    for el in parent.iterchildren():
        if el.tag == qn("w:p"):
            lines.append(Paragraph(el, doc).text)
        elif el.tag == qn("w:tbl"):
            seen = set()                              # a merged cell comes back once per column/row it spans
            for row in Table(el, doc).rows:
                cells = []
                for cell in row.cells:
                    if cell._tc in seen:
                        continue
                    seen.add(cell._tc)
                    text = " ".join(l for l in docx_lines(cell._tc, doc) if l.strip())
                    if text:
                        cells.append(text)
                lines.append(" | ".join(cells))
    return lines


SPACED_RUN = re.compile(r"(?<!\S)\w(?: \w){2,}(?!\S)")   # 3+ single letters with ONE space between them


def join_spaced_letters(text):
    """Undo PDF letter-spacing: "P R O F I L E  S U M M A R Y" -> "PROFILE SUMMARY".
    Only lines that are mostly single letters are touched; every other line stays exactly as it was."""
    lines = []
    for line in text.split("\n"):
        tokens = line.split()
        singles = sum(len(t) == 1 and t.isalnum() for t in tokens)
        if singles >= 4 and singles / len(tokens) > 0.6:
            line = SPACED_RUN.sub(lambda m: m.group().replace(" ", ""), line)
            line = re.sub(r" {2,}", " ", line)            # the wider gap between words becomes one space
        lines.append(line)
    return "\n".join(lines)


def read_file(path):
    """The plain text of one PDF, DOCX, TXT or MD file."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return join_spaced_letters("\n".join(page.extract_text() or "" for page in PdfReader(path).pages))
    if suffix == ".docx":
        doc = Document(path)
        return "\n".join(docx_lines(doc.element.body, doc))
    return path.read_text(encoding="utf-8", errors="replace")


def load_cvs(folder):
    """Read every CV in `folder` (PDF, DOCX, TXT, MD).
    Return a list of documents: [{"source": "<file name>", "text": "<all the text>"}, ...]
    `source` must be the FILE NAME: Verify checks retrieval against it."""
    docs = []
    for path in sorted(pathlib.Path(folder).iterdir()):
        if path.suffix.lower() in CV_TYPES and not path.name.startswith((".", "~$")):
            docs.append({"source": path.name, "text": read_file(path)})
    return docs


NAME_NOISE = re.compile(r"cv|resume|résumé|curriculum|vitae|updated|final|new|copy|draft|v\d+|\d+|"
                        r"jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec", re.I)


def candidate_name(source):
    """The candidate's name from the file name: "Test_Person_CV_Aug_2026.docx" -> "Test Person".
    Returns "" when the file name is ambiguous (fewer than 2 or more than 3 name words): we never guess."""
    words = [w for w in re.split(r"[\s_\-.()]+", pathlib.Path(source).stem) if w and not NAME_NOISE.fullmatch(w)]
    if not 2 <= len(words) <= 3 or not all(w.isalpha() for w in words):
        return ""
    return " ".join(w.capitalize() if w.islower() or w.isupper() else w for w in words)


def name_prefix(name):
    """The line chunk() puts in front of every chunk of a CV with a verified name ("" without one)."""
    return f"Candidate: {name}\n" if name else ""


def embedding_text(c):
    """The chunk text without the name prefix: the prefix shifted the vectors and hurt Verify,
    so we embed the original CV text and keep the name only in the stored text and metadata."""
    prefix = name_prefix(c.get("name", ""))
    return c["text"][len(prefix):] if prefix and c["text"].startswith(prefix) else c["text"]


# ---------------------------------------------------------------- LEVEL 4 step 5 · metadata: city, years
# A general gazetteer (written by hand, not taken from the CVs): canonical city -> spellings.
CITY_ALIASES = {
    "haifa": ["haifa"], "tel aviv": ["tel aviv", "tel-aviv", "tel aviv-yafo", "tel aviv-jaffa"],
    "jerusalem": ["jerusalem"], "nazareth": ["nazareth", "nasra"], "nof hagalil": ["nof hagalil", "nazareth illit"],
    "beersheba": ["beersheba", "beer sheva", "be'er sheva", "beer-sheva", "be'er-sheva"], "acre": ["acre", "akko", "akka"],
    "umm al-fahm": ["umm al-fahm", "umm el-fahem", "umm al fahm"], "shefa-amr": ["shefa-amr", "shefa amr", "shfaram", "shefaram"],
    "tamra": ["tamra"], "sakhnin": ["sakhnin"], "rahat": ["rahat"], "tayibe": ["tayibe", "taibe", "tayibeh"],
    "kafr qasim": ["kafr qasim", "kfar qasem", "kafr kassem"], "baqa al-gharbiyye": ["baqa al-gharbiyye", "baqa el-gharbia"],
    "tira": ["tira"], "ramla": ["ramla", "ramle"], "lod": ["lod", "lydda"], "jaffa": ["jaffa", "yafo"],
    "netanya": ["netanya"], "herzliya": ["herzliya", "herzlia"], "raanana": ["raanana", "ra'anana"],
    "petah tikva": ["petah tikva", "petach tikva", "petah tiqva"], "rishon lezion": ["rishon lezion", "rishon le zion"],
    "holon": ["holon"], "bat yam": ["bat yam"], "ramat gan": ["ramat gan"], "rehovot": ["rehovot"], "ashdod": ["ashdod"],
    "ashkelon": ["ashkelon"], "eilat": ["eilat"], "tiberias": ["tiberias"], "safed": ["safed", "tzfat", "zefat"],
    "karmiel": ["karmiel", "carmiel"], "afula": ["afula"], "nahariya": ["nahariya", "nahariyya"],
    "kiryat shmona": ["kiryat shmona"], "yokneam": ["yokneam", "yoqneam"], "kafr kanna": ["kafr kanna", "kfar kana"],
    "daliyat al-karmel": ["daliyat al-karmel", "daliyat el-carmel"], "isfiya": ["isfiya", "usfiya"], "maghar": ["maghar"],
    "arraba": ["arraba", "arrabe"], "majd al-krum": ["majd al-krum"], "ibillin": ["ibillin", "i'billin"],
    "kafr yasif": ["kafr yasif", "kfar yasif"], "jish": ["jish"], "kiryat ata": ["kiryat ata"], "kiryat bialik": ["kiryat bialik"],
    "kiryat motzkin": ["kiryat motzkin"], "kiryat yam": ["kiryat yam"], "nesher": ["nesher"], "tirat carmel": ["tirat carmel"],
    "hadera": ["hadera"], "kfar saba": ["kfar saba"], "hod hasharon": ["hod hasharon"], "modiin": ["modiin", "modi'in"],
    "ramallah": ["ramallah"], "bethlehem": ["bethlehem"], "jenin": ["jenin"], "nablus": ["nablus"],
    # LEVEL 4 step 5, more general localities (general knowledge, not from the CVs). Left out on purpose:
    # spellings that are also common first names or surnames (Ariel, Elad, Shoham, Arad, Yafa, Turan, Paris).
    "rosh haayin": ["rosh haayin", "rosh ha'ayin"], "or yehuda": ["or yehuda"], "yehud": ["yehud"],
    "kiryat ono": ["kiryat ono"], "givatayim": ["givatayim", "giv'atayim"], "bnei brak": ["bnei brak", "bne brak"],
    "ramat hasharon": ["ramat hasharon"], "kiryat gat": ["kiryat gat"], "kiryat malakhi": ["kiryat malakhi"],
    "sderot": ["sderot"], "netivot": ["netivot"], "ofakim": ["ofakim"], "dimona": ["dimona"], "yeruham": ["yeruham"],
    "beit shemesh": ["beit shemesh"], "mevaseret zion": ["mevaseret zion"], "maale adumim": ["maale adumim", "ma'ale adumim"],
    "yavne": ["yavne", "yavneh"], "ness ziona": ["ness ziona", "nes ziona"], "gedera": ["gedera"],
    "kiryat tivon": ["kiryat tivon", "tivon"], "zichron yaakov": ["zichron yaakov", "zikhron ya'akov", "zichron ya'akov"],
    "binyamina": ["binyamina"], "pardes hanna-karkur": ["pardes hanna", "pardes hanna-karkur"], "or akiva": ["or akiva"],
    "migdal haemek": ["migdal haemek", "migdal ha'emek"], "beit shean": ["beit shean", "beit she'an"],
    "kfar yona": ["kfar yona"], "even yehuda": ["even yehuda"], "tel mond": ["tel mond"], "rosh pinna": ["rosh pinna"],
    "katzrin": ["katzrin", "qatzrin"], "maalot-tarshiha": ["maalot", "ma'alot", "maalot-tarshiha", "tarshiha"],
    "kafr manda": ["kafr manda", "kfar manda"], "kabul": ["kabul"], "turan": ["tur'an"], "reineh": ["reineh"],
    "mashhad": ["mashhad"], "iksal": ["iksal"], "yafa an-naseriyye": ["yafa an-naseriyye", "yafia"],
    "daburiyya": ["daburiyya", "dabburiya"], "ein mahil": ["ein mahil"], "deir hanna": ["deir hanna"],
    "eilabun": ["eilabun"], "deir al-asad": ["deir al-asad"], "biina": ["bi'ina", "biina"], "nahf": ["nahf"],
    "rameh": ["rameh"], "peki'in": ["peki'in", "pekiin", "buqei'a"], "kisra-sumei": ["kisra-sumei", "kisra"],
    "yarka": ["yarka"], "julis": ["julis"], "abu snan": ["abu snan"], "jadeidi-makr": ["jadeidi-makr", "judeida-makr", "jdeideh"],
    "shibli": ["shibli"], "basmat tabun": ["basmat tab'un", "basmat tabun"], "zarzir": ["zarzir"],
    "fureidis": ["fureidis"], "jisr az-zarqa": ["jisr az-zarqa"], "arara": ["ar'ara", "arara"], "kafr qara": ["kafr qara"],
    "jatt": ["jatt"], "qalansawe": ["qalansawe", "qalansuwa"], "kafr bara": ["kafr bara"], "jaljulia": ["jaljulia"],
    "hurfeish": ["hurfeish"], "beit jann": ["beit jann"], "majdal shams": ["majdal shams"], "masade": ["mas'ade", "masade"],
    "ein qiniyye": ["ein qiniyye"], "lakiya": ["lakiya"], "hura": ["hura"], "tel sheva": ["tel sheva"],
    "kuseife": ["kuseife"], "segev shalom": ["segev shalom"], "ilut": ["ilut"], "ein rafa": ["ein rafa"],
    "hebron": ["hebron", "al-khalil"], "tulkarm": ["tulkarm", "tulkarem"], "qalqilya": ["qalqilya", "qalqiliya"],
    "jericho": ["jericho"], "gaza": ["gaza"], "birzeit": ["birzeit", "bir zeit"], "beit jala": ["beit jala"],
    "beit sahour": ["beit sahour"], "al-bireh": ["al-bireh", "el-bireh"], "tubas": ["tubas"], "salfit": ["salfit"],
    "amman": ["amman"], "irbid": ["irbid"], "zarqa": ["zarqa"], "aqaba": ["aqaba"], "madaba": ["madaba"],
    "london": ["london"], "berlin": ["berlin"], "new york": ["new york", "nyc"], "dubai": ["dubai"],
    "cairo": ["cairo"], "istanbul": ["istanbul"], "toronto": ["toronto"],
}
ALIAS_TO_CITY = {a: city for city, aliases in CITY_ALIASES.items() for a in aliases}
CITY_ALTERNATION = "|".join(re.escape(a) for a in sorted(ALIAS_TO_CITY, key=len, reverse=True))   # longest first
CITY_RE = re.compile(rf"\b({CITY_ALTERNATION})\b", re.I)
# An explicit residence cue. ", Israel" is NOT one: "Acme, Haifa, Israel 2019-2021" is a job line;
# "Haifa, Israel" in the header is found by header_city_segments instead.
RESIDENCE_RE = re.compile(r"\b(?:address|location|lives? in|living in|based in|resid\w*|home ?town)\b", re.I)
CONTACT_RE = re.compile(r"@|\+?\d[\d\s\-()]{7,}\d|\b(?:phone|mobile|linkedin)\b", re.I)
NOT_HOME_RE = re.compile(r"universit|college|institute|technion|school|academy|\bltd\b|\binc\b|company|\bat\b", re.I)
HEADER_CHARS = 400


def city_key(spelling):
    """Any spelling of a city -> its canonical name ("Akko" -> "acre"), or None."""
    return ALIAS_TO_CITY.get(spelling.lower().replace("’", "'"))


SEGMENT_SPLIT = re.compile(r"\s*(?:[|•·;,/–—]|\s-\s|\s{2,})\s*")       # "Name · Haifa, Israel | phone"
COUNTRY_TAIL = re.compile(r"\s+(?:israel|palestine|jordan)$", re.I)


def header_lines(text):
    """The CV's header: the lines in the first HEADER_CHARS characters, before the first section heading."""
    lines = []
    for line in text[:HEADER_CHARS].split("\n"):
        head = HEAD_DECORATION.sub("", line.split("|")[0].strip())
        if head and (EXPERIENCE_HEAD.fullmatch(head) or OTHER_HEAD.fullmatch(head)):
            break
        lines.append(line)
    return lines


def header_city_segments(line):
    """Cities that are a WHOLE segment of a header line ("Haifa" in "Name · Haifa · phone"), never part of
    a longer phrase ("Tel Aviv University", "Ariel Cohen")."""
    out = []
    for seg in SEGMENT_SPLIT.split(line):
        seg = COUNTRY_TAIL.sub("", re.sub(r"^[\W\d_]+|[\W\d_]+$", "", seg))    # emoji, zip code, "Israel"
        if seg and city_key(seg):
            out.append(seg)
    return out


def extract_city(text):
    """(city, provenance) of where the candidate LIVES. Evidence, in two kinds:
      "address_line"   a line with an address/location/"based in" cue, outside the Experience and Education sections;
      "header_segment" a city that is a whole segment of a header line (before the first section heading).
    A line naming a university, college, company or "at" is never evidence (a study or work place).
    No evidence -> ("", "none"); two different cities -> ("", "ambiguous"): we never pick one."""
    found, section = {}, None
    for line in text.split("\n"):
        head = HEAD_DECORATION.sub("", line.split("|")[0].strip())
        if EXPERIENCE_HEAD.fullmatch(head):
            section = "work"
        elif OTHER_HEAD.fullmatch(head):
            section = "study" if re.match(r"education|academic", head, re.I) else "other"
        if section in ("work", "study"):
            continue                                   # "Location: Haifa" under a job or a degree = that place
        if RESIDENCE_RE.search(line) and not NOT_HOME_RE.search(line):
            for m in CITY_RE.findall(line):
                found.setdefault(city_key(m), "address_line")
    for line in header_lines(text):
        if not NOT_HOME_RE.search(line):
            for seg in header_city_segments(line):
                found.setdefault(city_key(seg), "header_segment")
    if len(found) == 1:
        return next(iter(found.items()))
    return "", "ambiguous" if found else "none"


STATED_YEARS = re.compile(r"\b(\d{1,2})\s*\+?\s*(?:years?|yrs?)\s+of\s+(?:professional\s+|industry\s+|work\s+|relevant\s+)?"
                          r"experience\b(?!\s+(?:in|with|using|on)\b)", re.I)      # "with Python" = one skill, not total
EXPERIENCE_HEAD = re.compile(r"(?:work |professional |employment |relevant |career )?"
                             r"(?:experience|employment(?: history)?|work history|career history)", re.I)
OTHER_HEAD = re.compile(r"education|academic background|(?:technical )?skills|projects|certifications?|courses|languages|"
                        r"volunteer(?:ing)?|military service|personal (?:details|information)|summary|profile|about me|"
                        r"awards|publications|references|interests|hobbies|training", re.I)
# Lines under an Experience heading that are still not paid work. A university or college alone is NOT here:
# it can be the employer ("Teaching Assistant, X University"); a degree line still has a degree word.
NOT_WORK_RE = re.compile(r"\bb\.?sc\b|\bm\.?sc\b|\bb\.a\b|\bm\.a\b|bachelor|master'?s? degree|degree|diploma|\bstudent\b|"
                         r"\bph\.?d\b|school|\bcourses?\b|\bborn\b|birth|\bage\b|military|\barmy\b|\bidf\b|national service|"
                         r"volunteer|\bintern(?:ship)?s?\b", re.I)
MONTHS = {m: i for i, m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split(), 1)}
MONTH = r"(?:(\d{1,2})\s*[/.]\s*|(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*)?"   # "03/", "Sep.", "March "
DATE_RANGE = re.compile(rf"\b{MONTH}((?:19|20)\d{{2}})\s*(?:-|–|—|to|until)\s*{MONTH}((?:19|20)\d{{2}}|present|current|now|today)\b",
                        re.I)
THIS_YEAR = time.localtime().tm_year
NOW_MONTH = THIS_YEAR * 12 + time.localtime().tm_mon            # months since year 0, this month included
HEAD_DECORATION = re.compile(r"[\s\-—–_=•·:|#*~.]+$")           # "Experience ————", "EXPERIENCE:"


def month_index(num, name, year, end=False):
    """A date as months since year 0. A start month counts from its beginning, an end month to its end;
    a year without a month = January (so "2019 – 2021" = 2 years, as before). None when the month is not 1-12."""
    month = MONTHS[name[:3].lower()] if name else int(num) if num else 1
    if not 1 <= month <= 12:
        return None
    return int(year) * 12 + month - 1 + (1 if end and (num or name) else 0)


def experience_spans(text):
    """(start, end) month spans of the dated lines under an Experience heading (not education, intern,
    military, volunteer, birth lines). Impossible ranges (end before start, in the future, before 1970) dropped."""
    spans, in_experience = [], False
    for line in text.split("\n"):
        head = HEAD_DECORATION.sub("", line.split("|")[0].strip())
        if EXPERIENCE_HEAD.fullmatch(head):
            in_experience = True
        elif OTHER_HEAD.fullmatch(head):
            in_experience = False
        if not in_experience or NOT_WORK_RE.search(line):
            continue
        for sm, sname, sy, em, ename, ey in DATE_RANGE.findall(line):
            start = month_index(sm, sname, sy)
            end = NOW_MONTH if not ey[:1].isdigit() else month_index(em, ename, ey, end=True)
            if start is not None and end is not None and 1970 * 12 <= start <= end <= NOW_MONTH + 1:
                spans.append((start, end))
    return spans


AT_LEAST_BEFORE = re.compile(r"\b(?:over|more than|at least|minimum(?: of)?)\s*$", re.I)
ABOUT_BEFORE = re.compile(r"(?:\b(?:almost|nearly|about|around|approximately|roughly|close to)|~)\s*$", re.I)


def stated_kind(text, m):
    """How exact one "N years of experience" is: "stated" (exact), "stated_min" ("5+", "over 5": at least N)
    or "stated_approx" ("about 5", "nearly 5": could be a bit less or more)."""
    before = text[max(0, m.start() - 20):m.start()]
    if ABOUT_BEFORE.search(before):
        return "stated_approx"
    return "stated_min" if "+" in m.group(0) or AT_LEAST_BEFORE.search(before) else "stated"


def extract_years(text):
    """(years of professional experience, provenance). 1) one explicit "N years of experience" ->
    "stated" / "stated_min" / "stated_approx" (see stated_kind); 2) else dated jobs under an Experience heading,
    overlapping jobs merged, rounded DOWN -> "experience_dates" (a LOWER bound: undated jobs are not counted);
    3) else (-1, "none"). Never uses birth dates or age."""
    matches = list(STATED_YEARS.finditer(text))
    if len({int(m.group(1)) for m in matches}) == 1:
        kinds = {stated_kind(text, m) for m in matches}
        kind = "stated" if "stated" in kinds else "stated_approx" if "stated_approx" in kinds else "stated_min"
        return int(matches[0].group(1)), kind
    spans = sorted(experience_spans(text))
    if not spans:
        return -1, "none"
    total, (cur_start, cur_end) = 0, spans[0]
    for start, end in spans[1:]:                         # merge overlapping jobs: count each month once
        if start > cur_end:
            total, cur_start = total + cur_end - cur_start, start
        cur_end = max(cur_end, end)
    return (total + cur_end - cur_start) // 12, "experience_dates"


def chunk(doc, size=800, overlap=100):
    """Split ONE document into chunks.
    Return [{"id": "<unique id>", "source": doc["source"], "text": "<chunk text>"}, ...]
    You may add more keys (they become metadata), e.g. "name", "section", "city"."""
    text = doc["text"]
    if not text.strip():
        return []
    name = candidate_name(doc["source"])
    if name and not all(re.search(rf"\b{re.escape(w)}\b", text, re.I) for w in name.split()):
        name = ""                                 # the file name does not match the CV text: don't guess
    prefix = name_prefix(name)
    city, city_source = extract_city(text)              # once per CV, copied onto every chunk
    years, years_source = extract_years(text)
    step = size - overlap
    starts = range(0, max(len(text) - overlap, 1), step)
    return [{"id": f"{doc['source']}#{i}", "source": doc["source"], "name": name,
             "city": city, "city_source": city_source, "years": years, "years_source": years_source,
             "text": prefix + text[start:start + size]}
            for i, start in enumerate(starts)]


# ================================================================= LEVEL 2 · INDEX (the vector DB)
# Prompt idea: "Implement build_index with chromadb: a PersistentClient in HERE / 'chroma_db',
#  delete and recreate a collection 'cvs' with cosine distance, add every chunk with its id,
#  text, metadata and embed(texts) as embeddings. Return the collection count."
COLLECTION = None
BM25 = None               # LEVEL 4 step 3: the keyword index next to Chroma
CHUNKS_BY_ID = {}         # chunk id -> chunk, to return chunks that only BM25 found
POOL = 20                 # results taken from EACH search before fusion
RRF_K = 60                # RRF: score = sum of 1 / (RRF_K + rank)
RERANK_ENABLED = True     # LEVEL 4 step 4: on, Verify went from 6/10 to 7/10 with it
RERANK_POOL = 20          # RRF results the cross-encoder re-scores before we keep K
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
RERANKER = None           # the loaded ragkit Reranker (or False after a failed load)


def get_reranker():
    """The cross-encoder, loaded once. None when reranking is off or the model did not load.
    ragkit's Reranker silently scores by word overlap without a model: we never use that fallback."""
    global RERANKER
    if not RERANK_ENABLED:
        return None
    if RERANKER is None:
        RERANKER = Reranker(RERANK_MODEL)
        if RERANKER.model is None:
            print(f"WARNING: reranker model {RERANK_MODEL} did not load: keeping the RRF ranking (no reranking)")
            RERANKER = False
    return RERANKER or None


FILTERS_ENABLED = False   # LEVEL 4 step 5: an experiment, off until Verify shows it does no harm
LAST_FILTER = None        # what the last retrieve() filtered on, and how many CVs were left out as unknown
CITY_IN_QUESTION = re.compile(rf"\b(?:in|from)\s+(?:the\s+city\s+of\s+)?({CITY_ALTERNATION})\b", re.I)
NEGATED = re.compile(r"\b(?:not|outside(?: of)?|except|excluding|other than)\s+(?:\w+\s+){0,2}$", re.I)
MIN_YEARS = [(re.compile(r"\b(\d{1,2})\s*\+\s*(?:years?|yrs?)\b", re.I), 0),
             (re.compile(r"\bat least\s+(\d{1,2})\s+(?:years?|yrs?)\b", re.I), 0),
             (re.compile(r"\b(\d{1,2})\s+or more\s+(?:years?|yrs?)\b", re.I), 0),
             (re.compile(r"\bminimum(?: of)?\s+(\d{1,2})\s+(?:years?|yrs?)\b", re.I), 0),
             (re.compile(r"\b(?:more than|over)\s+(\d{1,2})\s+(?:years?|yrs?)\b", re.I), 1)]   # "more than 4" = 5+


def parse_filters(query):
    """What the question explicitly requires: {"city": "haifa"}, {"min_years": 5}, both, or None.
    Only positive phrasings ("in Haifa", "5+ years", "at least 5 years"); "not in Haifa" is not a filter."""
    found = {}
    for m in CITY_IN_QUESTION.finditer(query):
        if not NEGATED.search(query[:m.start()]):
            found["city"] = city_key(m.group(1))
            break
    for pattern, extra in MIN_YEARS:
        m = pattern.search(query)
        if m:
            found["min_years"] = int(m.group(1)) + extra
            break
    return found or None


FILTER_MODE = "lenient"   # "lenient": drop only CONFIRMED nonmatches, keep unknowns (ranked after the matches)
                          # "strict":  keep only CONFIRMED matches (unknowns are dropped too)
LOWER_BOUND_YEARS = ("stated_min", "experience_dates")   # the real number may be higher


def field_status(c, field, want):
    """"match", "nonmatch" or "unknown" for ONE condition on one CV's metadata.
    city: no city -> unknown. years: an exact stated number decides both ways; a lower bound ("5+", dated jobs)
    can only confirm a match; "about N" confirms only above N; no number -> unknown."""
    if field == "city":
        city = c.get("city") or ""
        return "unknown" if not city else "match" if city == want else "nonmatch"
    years, how = c.get("years", -1), c.get("years_source", "none")
    if years is None or years < 0:
        return "unknown"
    if how == "stated":
        return "match" if years >= want else "nonmatch"
    if how == "stated_approx":
        return "match" if years > want else "unknown"
    return "match" if years >= want else "unknown"          # a lower bound below `want` proves nothing


def filter_status(c, filters):
    """All conditions together: any confirmed nonmatch -> "nonmatch"; else any unknown -> "unknown"; else "match"."""
    statuses = [field_status(c, f, filters[k]) for f, k in (("city", "city"), ("years", "min_years")) if k in filters]
    return "nonmatch" if "nonmatch" in statuses else "unknown" if "unknown" in statuses else "match"


def passes(c, filters):
    """Keep this CV? THE one decision for BM25 and the graph; where_clause says the same to Chroma."""
    status = filter_status(c, filters)
    return status == "match" or (status == "unknown" and FILTER_MODE == "lenient")


def where_clause(filters):
    """passes() as a Chroma where= filter on the chunk metadata (same result: tests/compare in the audit)."""
    lenient = FILTER_MODE == "lenient"
    conds = []
    if "city" in filters:
        conds.append({"city": {"$in": [filters["city"], ""]}} if lenient else {"city": filters["city"]})
    if "min_years" in filters:
        n = filters["min_years"]
        if lenient:      # dropped only when confirmed: an exact "stated" number below n
            conds.append({"$or": [{"years": {"$gte": n}}, {"years_source": {"$ne": "stated"}}]})
        else:            # a confirmed match: >= n, except "about N" which must be > n
            conds.append({"$or": [{"$and": [{"years": {"$gte": n}}, {"years_source": {"$ne": "stated_approx"}}]},
                                  {"$and": [{"years": {"$gt": n}}, {"years_source": "stated_approx"}]}]})
    return conds[0] if len(conds) == 1 else {"$and": conds}


def filter_fields(hit):
    """The filter fields this hit's CV does not CONFIRM ("city", "years of experience", or both): not stated,
    ambiguous (two cities), or only a lower bound / an "about N" below the requirement."""
    filters = {k: v for k, v in (hit.get("filter") or {}).items() if k in ("city", "min_years")}
    c = next((c for c in CHUNKS_BY_ID.values() if c["source"] == hit["source"]), {})
    names = [label for f, k, label in (("city", "city", "city"), ("years", "min_years", "years of experience"))
             if k in filters and field_status(c, f, filters[k]) == "unknown"]
    return " and ".join(names) or "this requirement"


def filter_label(hit):
    """For the passage header in build_prompt: "" without a filter, else confirmed or not stated."""
    status = hit.get("filter_status")
    if not status:
        return ""
    return "; filter: confirmed match" if status == "match" else f"; {filter_fields(hit)}: not confirmed by the CV"


def filter_report(filters):
    """How many CVs are confirmed matches, unknown (kept in lenient mode, dropped in strict) and confirmed nonmatches."""
    cvs = {c["source"]: c for c in CHUNKS_BY_ID.values()}.values()
    counts = {s: 0 for s in ("match", "unknown", "nonmatch")}
    for c in cvs:
        counts[filter_status(c, filters)] += 1
    return {**filters, "mode": FILTER_MODE, "confirmed_cvs": counts["match"], "unknown_cvs": counts["unknown"],
            "unknown_kept": FILTER_MODE == "lenient", "nonmatch_excluded": counts["nonmatch"]}


def bm25_text(c):
    """What BM25 indexes: the verified name (an exact word to match) + the original CV text.
    No "Candidate:" label: a word in only some chunks would favor them (we saw that with embeddings)."""
    return f"{c['name']}\n{embedding_text(c)}" if c.get("name") else embedding_text(c)


def build_index(chunks):
    """Store every chunk with its vector in a Chroma collection. Return how many are stored."""
    global COLLECTION, BM25, CHUNKS_BY_ID
    if not chunks:
        raise ValueError("build_index() got no chunks: check load_cvs() and chunk()")
    texts = [c["text"] for c in chunks]
    vectors = embed([embedding_text(c) for c in chunks])   # before touching the DB: if this fails, the old index stays
    client = chromadb.PersistentClient(path=str(HERE / "chroma_db"))
    if "cvs" in [c.name for c in client.list_collections()]:
        old = client.get_collection("cvs", embedding_function=None).count()
        print(f"build_index: replacing collection 'cvs' in {HERE / 'chroma_db'} ({old} old chunks)")
        client.delete_collection("cvs")
    collection = client.create_collection("cvs", configuration={"hnsw": {"space": "cosine"}},
                                          embedding_function=None)
    collection.add(ids=[c["id"] for c in chunks], documents=texts, embeddings=vectors.tolist(),
                   metadatas=[{k: v for k, v in c.items() if k not in ("id", "text")} for c in chunks])
    stored = collection.count()
    if stored != len(chunks):
        raise RuntimeError(f"build_index() stored {stored} of {len(chunks)} chunks")
    COLLECTION = collection
    BM25 = BM25Index([Chunk(c["id"], c["source"], bm25_text(c)) for c in chunks], contextual=False)
    CHUNKS_BY_ID = {c["id"]: c for c in chunks}
    if get_reranker():                            # load + warm up now, not inside the first question
        get_reranker().model.predict([("warm up", "warm up")], show_progress_bar=False)
    return stored


# ================================================================= LEVEL 3 · RETRIEVE AND PROMPT
# Prompt idea: "Implement retrieve: embed the query, query the Chroma collection for k results,
#  and return them as dicts with rank, source, text, score (1 - distance) and via='vector'.
#  Implement build_prompt: number the passages [1], [2]… with their source, then the question."
def retrieve(query, k=K):
    """Return the k best chunks: [{"rank": 1, "source": ..., "text": ..., "score": ..., "via": "vector"}, ...]
    Hybrid: up to POOL chunks from Chroma and POOL from BM25, merged with Reciprocal Rank Fusion.
    score = rrf_score = the RRF score; vector_score = cosine similarity (None when only BM25 found the chunk).
    With RERANK_ENABLED the top RERANK_POOL are re-ordered by the cross-encoder: rerank_score, rerank_ms, and
    rrf_rank = the rank before reranking. Without it (or without a model) the RRF order is kept."""
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError(f"retrieve(): k must be a whole number >= 1, got {k!r}")
    if COLLECTION is None:
        raise RuntimeError("retrieve(): no Chroma collection yet: build_index() (LEVEL 2) must run first")
    if not query or not query.strip():
        return []
    if PENDING and PENDING[0] == query and not _INTERNAL[0]:   # a follow-up about the previous answer's candidate(s)
        target, clarify = PENDING[1], PENDING[2]
        if len(target) == 1 and not clarify:
            return hybrid_retrieve(query, k, source=target[0])
        hits = [h for src in target for h in hybrid_retrieve(query, 1, source=src)]   # each possible referent
        for rank, h in enumerate(hits, 1):
            h["rank"] = rank
        return hits                                   # [] when the previous turn named nobody -> refusal
    scope = candidate_scope(query)                    # a question about ONE named candidate: search only that CV
    if scope:
        return hybrid_retrieve(query, k, source=scope)
    hits = hybrid_retrieve(query, k)
    entities = graph_entities(query)                  # STEP 7c: "which candidates know X?" -> all of them
    graph_hits = graph_retrieve(query, entities) if entities else []
    if not graph_hits:
        return hits
    # safety net: keep a text-search hit the graph missed, if its chunk names every skill explicitly.
    # Not for cities: a city named in a chunk may be a university or employer, not where the person lives.
    seen = {h["source"] for h in graph_hits}
    terms = [key_terms(e[1]) for e in entities]
    city_asked = any(e[0] == "city" for e in entities)
    extra = [h for h in hits if not city_asked and h["source"] not in seen and all(t <= key_terms(h["text"]) for t in terms)]
    extra = list({h["source"]: h for h in reversed(extra)}.values())[::-1]       # one per CV, best first
    for rank, h in enumerate(graph_hits + extra, 1):
        h["rank"] = rank
    return graph_hits + extra


def hybrid_retrieve(query, k, source=None):
    """LEVEL 4 steps 3-5: vector + BM25 -> RRF -> optional rerank, optional metadata filter.
    `source` limits the search to one CV. Technical paraphrases add a second vector list and BM25 terms."""
    pool = max(POOL, k)
    n = min(pool, COLLECTION.count())
    if n == 0:
        return []
    global LAST_FILTER
    filters = parse_filters(query) if FILTERS_ENABLED else None
    LAST_FILTER = filter_report(filters) if filters else None
    if LAST_FILTER:
        print(f"filter: {LAST_FILTER}")
    conds = ([where_clause(filters)] if filters else []) + ([{"source": source}] if source else [])
    where = None if not conds else conds[0] if len(conds) == 1 else {"$and": conds}
    expansion = expand_query(query)
    texts = [query] + ([expansion] if expansion else [])
    res = COLLECTION.query(query_embeddings=embed(texts).tolist(), n_results=n, where=where,
                           include=["documents", "metadatas", "distances"])
    stored = {cid: {"source": meta["source"], "text": text} for ids, docs, metas in zip(res["ids"], res["documents"],
              res["metadatas"]) for cid, text, meta in zip(ids, docs, metas)}
    vector = {cid: (rank, 1 - dist) for rank, (cid, dist) in enumerate(zip(res["ids"][0], res["distances"][0]), 1)}
    concept = {cid: rank for rank, cid in enumerate(res["ids"][1], 1)} if expansion else {}
    bm25_filter = (lambda ch: (source is None or CHUNKS_BY_ID[ch.id]["source"] == source)
                   and (not filters or passes(CHUNKS_BY_ID[ch.id], filters))) if (filters or source) else None
    keyword = ({h.chunk.id: h.rank for h in BM25.search(f"{query} {expansion}".strip(), pool, where=bm25_filter)}
               if BM25 else {})

    fused = {}
    for cid, (rank, _) in vector.items():
        fused[cid] = 1 / (RRF_K + rank)
    for ranks in (keyword, concept):
        for cid, rank in ranks.items():
            fused[cid] = fused.get(cid, 0) + 1 / (RRF_K + rank)
    ranked = sorted(fused, key=lambda cid: -fused[cid])        # stable: ties keep the vector order
    rrf_rank = {cid: r for r, cid in enumerate(ranked, 1)}

    reranker, rerank, rerank_ms = get_reranker(), {}, None
    if reranker:
        candidates = ranked[:max(RERANK_POOL, k)]
        t0 = time.perf_counter()
        rq = f"{query} ({expansion})" if expansion else query
        scores = reranker.model.predict([(rq, bm25_text(CHUNKS_BY_ID[cid])) for cid in candidates],
                                        show_progress_bar=False)
        rerank_ms = (time.perf_counter() - t0) * 1000
        rerank = dict(zip(candidates, map(float, scores)))
        ranked = sorted(candidates, key=lambda cid: -rerank[cid])
        print(f"rerank: {len(candidates)} chunks in {rerank_ms:.0f} ms")
    status = {cid: filter_status(CHUNKS_BY_ID[cid], filters) for cid in ranked} if filters else {}
    if filters:                                         # confirmed matches before CVs that do not state the field
        ranked = sorted(ranked, key=lambda cid: status[cid] != "match")

    hits = []
    for rank, cid in enumerate(ranked[:k], 1):
        c = stored.get(cid) or CHUNKS_BY_ID[cid]
        via = "+".join(name for name, found in (("vector", vector), ("bm25", keyword), ("concept", concept)) if cid in found)
        hits.append({"rank": rank, "source": c["source"], "text": c["text"], "score": fused[cid],
                     "rrf_score": fused[cid], "rrf_rank": rrf_rank[cid], "rerank_score": rerank.get(cid),
                     "rerank_ms": rerank_ms, "vector_score": vector[cid][1] if cid in vector else None, "via": via,
                     "filter": LAST_FILTER, "filter_status": status.get(cid)})
    return hits


CONTACT_DETAILS = re.compile(r"\S+@\S+|\+?\d[\d\s\-()]{7,}\d|https?://\S+|linkedin\S*", re.I)


def graph_evidence(hit, entity):
    """A short quote from the passage that names the graph skill(s), contact details removed.
    Cities get "lives in <city>" (from metadata), never the raw address line."""
    skills = [SKILL_PATTERNS[s.strip()] for s in entity.split("+") if s.strip() in SKILL_PATTERNS]
    cities = [s.strip() for s in entity.split("+") if s.strip() in CITY_ALIASES]
    parts = []
    if skills:
        text = embedding_text({"text": hit["text"], "name": candidate_name(hit["source"])}).replace("\n", " ")
        for sent in re.split(r"(?<=[.;!?])\s+|\s+[|•·]\s+", text):
            if all(s.search(sent) for s in skills):
                sent = CONTACT_DETAILS.sub("", sent).strip(" ,;-")
                parts.append(sent if len(sent) <= 120 else sent[:117].rsplit(" ", 1)[0] + "…")
                break
        else:
            parts.append(f"{', '.join(e for e in entity.split(' + ') if e not in cities)} (in this CV)")
    parts += [f"lives in {c.title()}" for c in cities]
    return "; ".join(parts)


def clarify_prompt(question, hits):
    """An ambiguous "this candidate": ask which one instead of answering. The candidates are listed only
    when the previous answer was a graph list (then we know it named exactly them); otherwise the pipeline
    cannot see what the previous answer said, so it asks for the name without suggesting anyone."""
    if PENDING[3]:
        names = list(dict.fromkeys(verified_name(h["source"]) or pathlib.Path(h["source"]).stem for h in hits))
        ask = f"Which candidate do you mean: {', '.join(names[:-1])} or {names[-1]}?" if len(names) > 1 else \
              f"Do you mean {names[0]}?"
    else:
        ask = "Which candidate do you mean? Please give their name."
    global SYSTEM_PROMPT
    SYSTEM_PROMPT = CLARIFY_SYSTEM_PROMPT              # for THIS call only: build_prompt/make_query restore it
    return f"Context:\n(no passages: the question is ambiguous)\n\nQuestion: {question}\nReply: {ask}"


def build_prompt(question, hits):
    """The text the LLM receives: the numbered passages, then the question."""
    global SYSTEM_PROMPT
    SYSTEM_PROMPT = BASE_SYSTEM_PROMPT
    if not hits:
        passages = "(no CV passages were found)"
    else:
        passages = "\n\n".join(f"[{i}] (candidate: {verified_name(h['source']) or 'unnamed'}; source: {h['source']}"
                               f"{filter_label(h)})\n{h['text'].strip()}" for i, h in enumerate(hits, 1))
    graph = [h for h in hits if str(h.get("via", "")).startswith("graph:")]
    listed = len(graph) >= 2 and len(graph) == len(hits)
    remember_turn(question, dict.fromkeys(h["source"] for h in hits), listed)     # who this answer can name
    if PENDING and PENDING[2] and LAST_REWRITE[0] == question and PENDING[0] == LAST_REWRITE[1]:
        return clarify_prompt(question, hits)          # "this candidate" is ambiguous: ask, do not answer
    if listed:                                                    # STEP 7: a list answer; 1 hit = nothing to drop
        entity = graph[0]["via"].split(":", 1)[1].split("→")[0].strip()
        lines = "\n".join(f"- {h['via'].split('→', 1)[1].strip()} [{i}]: {graph_evidence(h, entity)}"
                          + (f" ({filter_fields(h)} not confirmed by the CV)" if h.get("filter_status") == "unknown" else "")
                          for i, h in enumerate(hits, 1))
        report = graph[0].get("filter") or {}
        if "city" in report and report.get("unknown_cvs"):      # the graph lists confirmed residents only
            lines += f"\n- ({report['unknown_cvs']} other CVs do not state where the candidate lives: not included)"
        return (f"Context:\n{passages}\n\nQuestion: {question}\n\n"
                f"The knowledge graph found exactly {len(hits)} candidates for \"{entity}\", one per passage, "
                f"each with the evidence from its passage. Answer by copying ALL {len(hits)} lines below exactly, "
                f"in this order, nothing before or after:\n{lines}")
    follow_up = LAST_REWRITE[0] == question
    if follow_up and LAST_REWRITE[1] != question:                       # STEP 1: say what a follow-up meant
        question = f"{question}\n(In this conversation this means: {LAST_REWRITE[1]})"
    unknown = [h for h in hits if h.get("filter_status") == "unknown"]
    if unknown:                                                         # STEP 5: never present unknown as a match
        question += ("\n(Passages marked \"not confirmed by the CV\" come from CVs that do not clearly show that "
                     "requirement: do not say that candidate meets it; say the CV does not confirm it.)")
    if len(hits) > 1 and len({h["source"] for h in hits}) == 1:          # one CV: surface its contradictions
        question += ("\n(All passages are from the same CV. Only if two passages state DIFFERENT values for the "
                     "same fact, e.g. two different universities for the same degree, mention both. A passage that "
                     "does not mention the fact is not a contradiction: then just answer from the passage that does.)")
    return f"Context:\n{passages}\n\nQuestion: {question}"     # "Context:" is what ragkit's LLM backends expect


# ================================================================= LEVEL 4 · FEATURES
# These already work in their simplest form. Each step of the menu replaces one of them
# (or retrieve / chunk above). One step at a time, then Verify.
FOLLOW_UP = re.compile(r"\b(?:he|she|him|his|her|hers|they|them|their|theirs|(?:this|that|the same|the) "
                       r"(?:candidate|person|one|applicant)|(?:of|among) them)\b", re.I)
CONTINUATION = re.compile(r"^\s*(?:and|also|what about|how about|and what about)\b", re.I)
POSSESSIVE = re.compile(r"\b(?:his|their|hers)\b|\bher\b(?=\s+[a-z])", re.I)          # "her skills" -> Name's skills
PERSON = re.compile(r"\b(?:he|she|him|her|they|them|(?:this|that|the same|the) (?:candidate|person|one|applicant))\b", re.I)
OTHERS = re.compile(r"\b(?:else|other|others|besides|except|apart from|compared?|than|versus|vs)\b", re.I)
LAST_REWRITE = (None, None)   # (question, rewritten query): build_prompt tells the LLM what a follow-up meant


def candidate_index():
    """source -> candidate name, for every indexed CV that has one (verified name, else from the file name)."""
    names = {}
    for c in CHUNKS_BY_ID.values():
        names.setdefault(c["source"], c.get("name") or candidate_name(c["source"]))
    return {src: n for src, n in names.items() if n}


def mentioned_candidates(text):
    """The CVs whose candidate `text` names: a full name, or a first/last name that only ONE candidate has."""
    names = candidate_index()
    full = {src for src, n in names.items() if re.search(rf"\b{re.escape(n)}\b", text, re.I)}
    if full:
        return full
    owners = {}
    for src, n in names.items():
        for w in n.split():
            if len(w) >= 3:
                owners.setdefault(w.lower(), set()).add(src)
    return {src for w, srcs in owners.items() if len(srcs) == 1 and re.search(rf"\b{re.escape(w)}\b", text, re.I)
            for src in srcs}


def candidate_scope(query):
    """The one CV a question is about ("Where did Dana Levi study?"), or None (no name, several, or "who else…")."""
    srcs = mentioned_candidates(query)
    return next(iter(srcs)) if len(srcs) == 1 and not OTHERS.search(query) else None


RESOLVE_MARGIN = 3.0      # cross-encoder logits the best CV must lead the second-best CV by
RESOLVE_MARGIN_VECTOR = 0.10                         # the same, in cosine, when reranking is off


def verified_name(source):
    """The candidate name of a CV only if it was verified against the CV text (chunk() did that), else None."""
    return next((c["name"] for c in CHUNKS_BY_ID.values() if c["source"] == source and c.get("name")), None)


def answer_candidate(question):
    """The ONE candidate the answer to an earlier question was about, re-found by retrieval, or None.
    Clear only when: the graph found exactly one candidate; or the search found evidence (not a refusal)
    and the best CV leads the second-best CV by a clear margin. Several plausible candidates -> None."""
    with contextlib.redirect_stdout(io.StringIO()):
        hits = retrieve(question, 10)
        if not hits or should_refuse(question, hits):
            return None
    graph = {h["source"] for h in hits if str(h.get("via", "")).startswith("graph:")}
    if graph:
        return next(iter(graph)) if len(graph) == 1 else None
    by_cv, use_rerank = {}, hits[0].get("rerank_score") is not None
    for h in hits:
        s = h["rerank_score"] if use_rerank else (h.get("vector_score") or 0)
        by_cv[h["source"]] = max(by_cv.get(h["source"], s), s)
    ranked = sorted(by_cv.values(), reverse=True)
    margin = RESOLVE_MARGIN if use_rerank else RESOLVE_MARGIN_VECTOR
    if len(ranked) > 1 and ranked[0] - ranked[1] < margin:
        return None
    top = max(by_cv, key=by_cv.get)
    key = (key_terms(question) - GENERIC_TERMS) | key_terms(expand_query(question))
    owners = term_owners()
    distinctive = {t for t in key if owners.get(t) == {top}}
    return top if distinctive else None                # a question word that ONLY this CV (in the whole index) has


_TERM_OWNERS = (None, {})


def term_owners():
    """key term -> the set of CVs (sources) whose text contains it, for the current index (cached)."""
    global _TERM_OWNERS
    if _TERM_OWNERS[0] is not CHUNKS_BY_ID:
        owners = {}
        for c in CHUNKS_BY_ID.values():
            for t in key_terms(embedding_text(c)):
                owners.setdefault(t, set()).add(c["source"])
        _TERM_OWNERS = (CHUNKS_BY_ID, owners)
    return _TERM_OWNERS[1]


LAST_TURN = {"key": None}  # the LAST turn the app answered: its key, its CVs ([] = refused), listed?, resolved?
PENDING = None            # (rewritten follow-up, [CVs], clarify?, listed?) for retrieve / should_refuse / build_prompt
CONV_KEY = None           # the conversation key of the turn being answered now (set by make_query)
_INTERNAL = [0]           # > 0 while make_query re-runs retrieval for itself: those are not answered turns
MAX_REFERENTS = 6


def turn_key(question):
    return " ".join(str(question).lower().split())


def conv_key(earlier, question):
    """A turn is identified by the questions before it AND its text: the same words in another chat are another turn."""
    return tuple(turn_key(t) for t in earlier), turn_key(question)


def remember_turn(question, cvs, listed=False):
    """Record the turn the app is answering now: which CVs its answer was generated from ([] = refused).
    Only ONE turn is kept: the pipeline has no conversation id, so anything older could belong to another chat."""
    global LAST_TURN
    if _INTERNAL[0]:
        return
    key = CONV_KEY if CONV_KEY and CONV_KEY[1] == turn_key(question) else conv_key([], question)
    unresolved = bool(PENDING) and PENDING[0] == LAST_REWRITE[1] and (PENDING[2] or not PENDING[1])
    LAST_TURN = {"key": key, "cvs": list(cvs), "listed": listed, "unresolved": unresolved}


def referent(history, past):
    """Which CV(s) a singular follow-up ("this candidate", "she") can mean, and how sure we are:
    ([cv], "named")       the user named the candidate, or the last assistant message names exactly one;
    ([cvs], "named")      several named -> clarify;
    ([], "answered")      the previous answer refused / found nobody -> refuse, never invent;
    ([cvs], "answered")   the previous turn WAS answered just now, but its text is not visible to the pipeline
                          (app.py passes only user questions) -> ask which candidate is meant;
    ([cvs], "inferred")   the previous question was not answered here (e.g. Verify history) -> re-run it;
    (None, None)          nothing to anchor to. Returns (cvs, how, listed)."""
    msgs = [h for h in (history or []) if isinstance(h, dict) and h.get("role") == "assistant"]
    if msgs:                                               # the app passed the assistant's answers: read the last one
        text = as_text(msgs[-1])
        if REFUSAL.lower().rstrip(".") in text.lower():
            return [], "answered", False
        named = mentioned_candidates(text)
        return (sorted(named)[:MAX_REFERENTS], "named", len(named) > 1) if named else ([], "answered", False)
    for i in range(len(past) - 1, max(len(past) - 5, -1), -1):
        turn = past[i]
        named = mentioned_candidates(turn)
        if named:                                          # the user named the candidate(s) earlier
            return sorted(named)[:MAX_REFERENTS], "named", False
        just_answered = conv_key(past[:i], turn) == LAST_TURN["key"]
        if FOLLOW_UP.search(turn):
            if just_answered and LAST_TURN["unresolved"]:  # we just asked "which one?" / refused: still unclear
                return LAST_TURN["cvs"][:MAX_REFERENTS], "answered", LAST_TURN["listed"]
            continue                                       # an earlier resolved follow-up: look further back
        if just_answered:                                  # answered just now: we cannot see what it said
            return LAST_TURN["cvs"][:MAX_REFERENTS], "answered", LAST_TURN["listed"]
        _INTERNAL[0] += 1                                  # not answered here: re-run that retrieval
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                hits = retrieve(turn, K)
                used = [] if not hits or should_refuse(turn, hits) else list(dict.fromkeys(h["source"] for h in hits))
            if len(used) > 1:
                one = answer_candidate(turn)               # one CV holds the only distinctive evidence?
                if one in used:
                    used = [one]
        finally:
            _INTERNAL[0] -= 1
        return used[:MAX_REFERENTS], "inferred", False
    return None, None, False


def as_text(turn):
    """History items may be strings or chat messages ({"role", "content"})."""
    return turn.get("content") or turn.get("question") or "" if isinstance(turn, dict) else str(turn)


# ---------------------------------------------------------------- STEP 1 · LLM rewrite, the rules as fallback
# The RULES decide WHO a follow-up is about (referent / PENDING: clear, ambiguous -> clarify, nobody -> refuse).
# The LLM only rephrases, and only when that is already settled; every rewrite is checked, else the rule wins.
LLM_REWRITE_ENABLED = True
LAST_REWRITE_BY = "rules"     # who wrote the last follow-up query: "llm" or "rules"
_REWRITE_CACHE = {}
REWRITE_SYSTEM = ("You rewrite the last question of a recruiting chat into ONE complete question that can be "
                  "understood without the chat. Use only words from the chat. Do not add any person's name unless "
                  "you are told which name to use. Do not answer the question. Reply with the question only.")
REWRITE_FILLER = ("and also about tell more do did does he she him his her hers "
                  "they them their theirs this that same one candidate person applicant")   # words a rewrite may drop
QUESTION_WORD = re.compile(r"\b(?:where|when|what|who|whom|which|why|how)\b", re.I)   # stopwords: checked apart


def similar_term(t, terms):
    """`t` is in `terms`, allowing a different ending ("study" ~ "studied", "build" ~ "built" is not caught)."""
    return any(u == t or (len(t) >= 5 and len(u) >= 5 and u[:5] == t[:5]) for u in terms)


def rewrite_ok(out, question, past, name=None, target=None, drop_skills=()):
    """Accept an LLM rewrite only if it is one short question that keeps the follow-up's own words and skills,
    keeps half of the earlier question's words (when it was combined with it), names no candidate the user
    did not name (with `name`: names that one candidate and nobody else), and holds no contact details."""
    if not out or len(out) > 220 or CONTACT_RE.search(out) or "know based on" in out.lower():
        return False
    named = mentioned_candidates(out)
    if name:
        if not re.search(rf"\b{re.escape(name)}\b", out, re.I) or named - {target} or POSSESSIVE.search(out) \
                or PERSON.search(out):
            return False                               # not that name, another name too, or "he" still there
    elif named - mentioned_candidates(" ".join(past + [question])):
        return False                                   # the LLM brought in a candidate nobody named
    core = CONTINUATION.sub("", question)              # "And where did he study?" -> "where did he study?"
    asked, wrote = ({w.lower() for w in QUESTION_WORD.findall(t)} for t in (core, out))
    if asked - wrote or wrote - asked - {w.lower() for w in QUESTION_WORD.findall(past[-1])}:
        return False                                   # "where" became "what", or a question nobody asked
    words, filler = key_terms(out), key_terms(REWRITE_FILLER)
    if not all(similar_term(t, words) for t in key_terms(core) - filler):
        return False
    skills = extract_skills(out)
    if not extract_skills(question) <= skills or skills & set(drop_skills):
        return False
    if not name:                                       # combined with the earlier question: keep its topic
        before = key_terms(past[-1]) - GENERIC_TERMS - filler
        if before and sum(similar_term(t, words) for t in before) < len(before) / 2:
            return False
    return True


def llm_rewrite(question, past, rule, name=None, target=None, drop_skills=()):
    """STEP 1: ask llm() for the full question; return it if rewrite_ok accepts it, else the rule's `rule`."""
    global LAST_REWRITE_BY
    LAST_REWRITE_BY = "rules"
    if not LLM_REWRITE_ENABLED:
        return rule
    chat = "\n".join(f"User: {t}" for t in past[-3:])
    if name:
        task = f"The follow-up is about the candidate {name}: use that name for the person it refers to."
    elif drop_skills:
        task = (f"The follow-up asks the earlier question again, with {', '.join(sorted(extract_skills(question)))} "
                f"instead of {', '.join(sorted(drop_skills))}.")
    else:
        task = "Keep every word of the follow-up; take the missing part from the earlier questions."
    user = f"Chat so far:\n{chat}\n\nFollow-up: {question}\n{task}\nComplete question:"
    if user not in _REWRITE_CACHE:
        try:
            out = llm(REWRITE_SYSTEM, user, 60)
        except Exception as e:                         # no model, no server, timeout: the rules still work
            print(f"follow-up: llm() failed ({type(e).__name__}), rules used")
            return rule
        out = next((ln for ln in str(out).strip().splitlines() if ln.strip()), "")
        out = re.sub(r"^\s*(?:complete|rewritten|full)?\s*question\s*:\s*", "", out, flags=re.I).strip(" \"'`")
        if len(_REWRITE_CACHE) > 500:
            _REWRITE_CACHE.clear()
        _REWRITE_CACHE[user] = out
    out = _REWRITE_CACHE[user]
    if rewrite_ok(out, question, past, name, target, drop_skills):
        LAST_REWRITE_BY = "llm"
        return out
    print(f"follow-up: LLM rewrite rejected ({out[:80]!r}), rules used")
    return rule


def make_query(question, history):
    """STEP 1 · Follow-ups: rewrite "and where did he study?" using `history` (earlier questions).
    Grounded: a pronoun becomes a candidate NAMED in the history (one that exists in the index, never a guess);
    "and Python?" swaps the skill of the previous question; "which of them…" is combined with the previous
    question (so the graph intersects both). A question that is not a follow-up is returned unchanged."""
    global LAST_REWRITE, PENDING, CONV_KEY, SYSTEM_PROMPT, LAST_REWRITE_BY
    PENDING = None                                     # never carry a follow-up target into another request
    LAST_REWRITE_BY = "rules"
    SYSTEM_PROMPT = BASE_SYSTEM_PROMPT
    users = [h for h in (history or []) if not (isinstance(h, dict) and h.get("role") == "assistant")]
    past = [t for t in (as_text(h).strip() for h in users) if t]
    CONV_KEY = conv_key(past, question)
    short =(len(key_terms(question) - GENERIC_TERMS) <= 1 and not extract_skills(question)
             and not LIST_QUESTION.search(question))       # "Education?" yes; "Who knows Kafka?" is standalone
    if not past or not (FOLLOW_UP.search(question) or CONTINUATION.search(question) or short) \
            or mentioned_candidates(question):
        LAST_REWRITE = (question, question)
        return question
    rewritten = None
    if FOLLOW_UP.search(question) and not re.search(r"\b(?:of|among) them\b", question, re.I):
        target, how, listed = referent(history, past)      # the CV(s) "this candidate" can mean, and how sure
        if target is not None:
            rewritten = question
            clear = len(target) == 1 and how in ("named", "inferred")   # an unseen answer is never "clear"
            if clear:
                name = verified_name(target[0])
                if not name and how == "named":
                    name = candidate_index().get(target[0])       # the user typed that name themselves
                if name:
                    rewritten = PERSON.sub(name, POSSESSIVE.sub(f"{name}'s", question, count=1), count=1)
                    rewritten = llm_rewrite(question, past, rewritten, name=name, target=target[0])
            PENDING = (rewritten, target, bool(target) and not clear, listed)   # [] -> refuse; else clarify
    if rewritten is None:
        new, old = extract_skills(question), extract_skills(past[-1])
        if CONTINUATION.search(question) and new and old and not new & old:
            rewritten = SKILL_PATTERNS[sorted(old)[0]].sub(", ".join(sorted(new)), past[-1], count=1)
            rewritten = llm_rewrite(question, past, rewritten, drop_skills=old)
        elif re.search(r"\b(?:of|among) them\b", question, re.I):
            rewritten = f"{question} {past[-1]}"       # rules only: the graph must see both questions' entities
        else:
            rewritten = llm_rewrite(question, past, f"{question} {past[-1]}")
    LAST_REWRITE = (question, rewritten)
    print(f"follow-up: rewritten for search by {LAST_REWRITE_BY}: {rewritten!r}")
    return rewritten


# ---------------------------------------------------------------- technical paraphrases -> canonical terms
# General tech knowledge (written by hand, not from any evaluation): the stems must ALL start words of the
# question (stems of 3 letters or less must match a whole word); the terms are added to the search.
CONCEPTS = [
    (("container", "orchestrat"), "kubernetes docker"), (("containeri",), "docker kubernetes"),
    (("event", "stream"), "kafka streaming"), (("stream", "process"), "kafka spark streaming"),
    (("real", "time", "data"), "kafka streaming"), (("message", "broker"), "kafka rabbitmq"),
    (("message", "queu"), "kafka rabbitmq"), (("pub", "sub"), "kafka"),
    (("dashboard",), "power bi tableau looker dashboards"), (("business", "intelligence"), "power bi tableau looker"),
    (("reporting", "tool"), "power bi tableau excel"), (("data", "visuali"), "power bi tableau visualization"),
    (("infrastructure", "code"), "terraform ansible"), (("continuous", "integrat"), "ci/cd jenkins"),
    (("continuous", "deliver"), "ci/cd jenkins"), (("continuous", "deploy"), "ci/cd jenkins"),
    (("cloud",), "aws azure gcp"), (("relational", "database"), "sql postgresql mysql"),
    (("non", "relational"), "nosql mongodb"), (("document", "database"), "mongodb nosql"),
    (("data", "warehous"), "snowflake bigquery databricks"), (("big", "data"), "spark hadoop hive databricks"),
    (("distributed", "comput"), "spark hadoop"), (("distributed", "process"), "spark hadoop"),
    (("workflow", "orchestrat"), "airflow"), (("pipeline", "orchestrat"), "airflow"), (("schedul", "pipeline"), "airflow"),
    (("extract", "transform"), "etl"), (("data", "pipeline"), "etl airflow"), (("data", "integrat"), "etl"),
    (("machine", "learn"), "machine learning scikit-learn"), (("predictive", "model"), "machine learning scikit-learn"),
    (("deep", "learn"), "deep learning pytorch tensorflow"), (("neural", "net"), "deep learning pytorch tensorflow"),
    (("language", "model"), "llm nlp openai"), (("generative", "ai"), "llm openai rag"), (("genai",), "llm openai rag"),
    (("chatbot",), "llm nlp rag"), (("text", "analy"), "nlp"), (("natural", "language"), "nlp"),
    (("image", "recogni"), "computer vision opencv"), (("image", "process"), "computer vision opencv"),
    (("version", "control"), "git github"), (("source", "control"), "git github"),
    (("automat", "test"), "pytest selenium"), (("unit", "test"), "pytest"), (("test", "automat"), "selenium pytest"),
    (("frontend",), "react angular vue javascript typescript"), (("front", "end"), "react angular vue javascript"),
    (("user", "interface"), "react angular vue figma"), (("backend",), "fastapi flask django node.js rest api"),
    (("back", "end"), "fastapi flask django node.js"), (("web", "framework"), "django flask fastapi react"),
    (("web", "servic"), "rest api fastapi flask"), (("api",), "rest api fastapi"), (("spreadsheet",), "excel"),
    (("statistic",), "statistics spss"), (("no", "code"), "n8n zapier power automate"),
    (("low", "code"), "n8n zapier power automate"), (("workflow", "automat"), "n8n zapier power automate airflow"),
    (("search", "engine"), "elasticsearch"), (("full", "text", "search"), "elasticsearch"), (("cach",), "redis"),
    (("devops",), "docker kubernetes ci/cd terraform jenkins"), (("mlops",), "mlflow docker kubernetes"),
    (("model", "deploy"), "mlflow docker fastapi"), (("microservic",), "microservices docker kubernetes rest api"),
    (("ui", "design"), "figma"), (("ux",), "figma"), (("crm",), "salesforce"), (("erp",), "sap"),
    (("object", "orient"), "java c# c++ python"), (("scripting",), "python bash"),
]


def expand_query(query):
    """The canonical tech terms of the concepts a question paraphrases ("container orchestration" ->
    "kubernetes docker"), minus words already in the question; "" when none."""
    words = tokenize(query)

    def has(stem):
        return any(w == stem if len(stem) <= 3 else w.startswith(stem) for w in words)

    terms = []
    for stems, extra in CONCEPTS:
        if all(has(s) for s in stems):
            terms += [t for t in extra.split() if t not in words and t not in terms]
    return " ".join(terms)


# Calibrated on the practice questions + 59 automatic "which candidates know <word>?" probes + 15 invented
# off-topic questions: answerable ones always had >= 1 key word in the top chunks; off-topic ones mostly 0
# and a cross-encoder score below -7. Both signals must agree before we refuse.
REFUSE_RERANK = -5.0      # cross-encoder logit of the best chunk
REFUSE_VECTOR = 0.30      # cosine of the best chunk, used only when reranking is off
QUESTION_WORDS = ("candidate candidates who which whom person people experience experienced know knows knowing "
                  "have has had work worked working skill skills year years anyone someone any list all show find "
                  "give tell does did using use used good strong background familiar most least more less best top "
                  "many much other also like just only")


def key_terms(text):
    """Lowercase content words, a plural "s" removed ("dashboards" == "dashboard")."""
    return {t[:-1] if len(t) > 3 and t.endswith("s") else t for t in tokenize(text)}


GENERIC_TERMS = key_terms(QUESTION_WORDS)


def should_refuse(query, hits):
    """STEP 6 · Return True when nothing relevant came back: the app answers REFUSAL without the LLM.
    Refuse only when BOTH say "no evidence": none of the question's key words is in the retrieved chunks,
    AND the best chunk scores low. Graph answers (exact entity matches) are never refused."""
    if PENDING and PENDING[0] == query and PENDING[2] and hits:
        return False                                   # an ambiguous follow-up: build_prompt asks which candidate
    if not hits:
        remember_turn(LAST_REWRITE[0] if LAST_REWRITE[1] == query else query, [])
        return True
    if any(str(h.get("via", "")).startswith("graph") for h in hits):
        return False
    key = key_terms(query) - GENERIC_TERMS
    if not key:
        return False                                   # nothing specific to check: let the LLM decide
    text_terms = set().union(*(key_terms(h["text"]) for h in hits))
    found = (key | key_terms(expand_query(query))) & text_terms      # a paraphrase's canonical terms count too
    best = hits[0].get("rerank_score")
    low = best < REFUSE_RERANK if best is not None else max(h.get("vector_score") or 0 for h in hits) < REFUSE_VECTOR
    print(f"refusal check: key words found {len(found)}/{len(key)}, best score {best if best is not None else 'n/a'}")
    refuse = not found and low
    if refuse:                                         # an answered turn that named nobody
        remember_turn(LAST_REWRITE[0] if LAST_REWRITE[1] == query else query, [])
    return refuse


# ================================================================= LEVEL 4 step 7 · GRAPH RAG
# A general skill list (written by hand, not from the CVs): canonical skill -> spellings.
SKILL_ALIASES = {
    "python": ["python"], "java": ["java"], "javascript": ["javascript", "js"], "typescript": ["typescript"],
    "c++": ["c++", "cpp"], "c#": ["c#"], ".net": [".net", "dotnet"], "golang": ["golang"], "rust": ["Rust"],
    "scala": ["scala"], "kotlin": ["kotlin"], "swift": ["Swift"], "php": ["php"], "ruby": ["ruby"], "matlab": ["matlab"],
    "sql": ["sql"], "nosql": ["nosql"], "postgresql": ["postgresql", "postgres"], "mysql": ["mysql"],
    "mongodb": ["mongodb", "mongo"], "redis": ["redis"], "elasticsearch": ["elasticsearch", "elastic search"],
    "oracle": ["oracle db", "oracle database", "pl/sql"], "kafka": ["kafka"], "spark": ["Spark", "PySpark", "pyspark"],
    "hadoop": ["hadoop"], "hive": ["hive"], "airflow": ["airflow"], "dbt": ["dbt"], "snowflake": ["snowflake"],
    "databricks": ["databricks"], "bigquery": ["bigquery"], "etl": ["etl"], "power bi": ["power bi", "powerbi"],
    "tableau": ["tableau"], "looker": ["looker"], "excel": ["Excel", "MS Excel"], "spss": ["spss"], "sas": ["SAS"],
    "pandas": ["pandas"], "numpy": ["numpy"], "scikit-learn": ["scikit-learn", "sklearn", "scikit learn"],
    "pytorch": ["pytorch"], "tensorflow": ["tensorflow"], "keras": ["keras"], "hugging face": ["hugging face", "huggingface"],
    "langchain": ["langchain"], "llm": ["llm", "llms", "large language models"], "rag": ["RAG", "retrieval-augmented"],
    "nlp": ["nlp", "natural language processing"], "computer vision": ["computer vision", "opencv"],
    "machine learning": ["machine learning"], "deep learning": ["deep learning"], "statistics": ["statistics"],
    "mlflow": ["mlflow"], "openai": ["openai"], "prompt engineering": ["prompt engineering"],
    "fastapi": ["fastapi"], "flask": ["flask"], "django": ["django"], "spring": ["spring boot"], "node.js": ["node.js", "nodejs"],
    "react": ["React", "react.js", "reactjs"], "angular": ["angular"], "vue": ["vue", "vue.js"], "html": ["html"], "css": ["css"],
    "rest api": ["rest api", "restful", "rest apis"], "graphql": ["graphql"], "microservices": ["microservices"],
    "aws": ["aws", "amazon web services"], "azure": ["azure"], "gcp": ["gcp", "google cloud"], "docker": ["docker"],
    "kubernetes": ["kubernetes", "k8s"], "terraform": ["terraform"], "ansible": ["ansible"], "jenkins": ["jenkins"],
    "ci/cd": ["ci/cd", "cicd"], "linux": ["linux"], "git": ["git", "github", "gitlab"], "jira": ["jira"],
    "selenium": ["selenium"], "pytest": ["pytest"], "n8n": ["n8n"], "zapier": ["zapier"], "power automate": ["power automate"],
    "salesforce": ["salesforce"], "sap": ["SAP"], "figma": ["figma"], "sharepoint": ["sharepoint"],
}
def alias_pattern(alias):
    """A whole-word pattern; a space or hyphen inside the alias also matches none/newline ("PowerBI", "Power-BI")."""
    body = re.sub(r"\\[ \-]", r"[\\s\\-]*", re.escape(alias))
    return rf"(?<![\w+#.]){body}(?![\w+#])"


SKILL_PATTERNS = {skill: re.compile("|".join(alias_pattern(a) for a in aliases),
                                    0 if any(a != a.lower() for a in aliases) else re.I)       # "React" vs "react to"
                  for skill, aliases in SKILL_ALIASES.items()}
GRAPH = nx.Graph()
LIST_QUESTION = re.compile(r"^\s*(?:who|which|list|name|find|show)\b|\b(?:all|every|any)\s+(?:the\s+)?candidates?\b", re.I)


def extract_skills(text):
    """Every skill of SKILL_ALIASES named in `text`, matched as a whole word."""
    return {skill for skill, pattern in SKILL_PATTERNS.items() if pattern.search(text)}


def extract(text):
    """STEP 7a · The entities of one CV: its skills, and its city (only with residence evidence, see extract_city)."""
    return {"skills": extract_skills(text), "city": extract_city(text)[0]}


def build_graph(chunks):
    """STEP 7 · Graph RAG: return a networkx graph with (type, name) nodes, e.g.
    ("candidate", "Lina Haddad") —has_skill— ("skill", "kubernetes"). Give candidate nodes
    source=<file name>. The 🕸 Graph tab draws it; use it in retrieve for "which candidates…" questions."""
    global GRAPH
    by_source = {}
    for c in chunks:
        by_source.setdefault(c["source"], []).append(c)
    g = nx.Graph()
    for source, cs in by_source.items():
        label = cs[0].get("name") or pathlib.Path(source).stem       # no verified name: the file name
        node = ("candidate", label if ("candidate", label) not in g else f"{label} ({source})")
        g.add_node(node, source=source)
        facts = extract("\n".join(embedding_text(c) for c in cs))
        for skill in facts["skills"]:
            g.add_edge(node, ("skill", skill), rel="has_skill")
        city = cs[0].get("city") or facts["city"]
        if city:
            g.add_edge(node, ("city", city), rel="lives_in")
    GRAPH = g
    return g


def graph_entities(query):
    """The skill and city nodes of GRAPH that a list question names, or [] (then: normal search)."""
    if not LIST_QUESTION.search(query) or not GRAPH.number_of_nodes():
        return []
    nodes = [("skill", s) for s in extract_skills(query)]
    city = parse_filters(query) or {}
    if city.get("city"):
        nodes.append(("city", city["city"]))
    return [n for n in nodes if n in GRAPH]


def graph_retrieve(query, entities):
    """STEP 7c · Every candidate connected to ALL the named entities (no top-K limit), each with its
    best chunk from Chroma (where source = that CV) as evidence, ordered by the cross-encoder if on."""
    people = set.intersection(*(set(GRAPH.neighbors(e)) for e in entities))
    people = [n for n in people if n[0] == "candidate"]
    qvec = embed([query]).tolist()
    global LAST_FILTER
    filters = parse_filters(query) if FILTERS_ENABLED else None
    LAST_FILTER = filter_report(filters) if filters else None
    hits = []
    for person in people:
        source = GRAPH.nodes[person]["source"]
        res = COLLECTION.query(query_embeddings=qvec, n_results=5, where={"source": source},
                               include=["documents", "distances"])
        if not res["ids"][0] or (filters and not passes(CHUNKS_BY_ID[res["ids"][0][0]], filters)):
            continue                                   # the same decision as hybrid_retrieve (passes)
        st = filter_status(CHUNKS_BY_ID[res["ids"][0][0]], filters) if filters else None
        rows = list(zip(res["ids"][0], res["documents"][0], res["distances"][0]))
        skills = [SKILL_PATTERNS[e[1]] for e in entities if e[0] == "skill"] if len(people) >= 2 else []
        # a list: evidence = the most similar chunk that really names the skill(s);
        # one candidate: the chunk most similar to the question (it may ask about another detail)
        cid, doc, dist = next((r for r in rows if all(s.search(r[1]) for s in skills)), rows[0])
        hits.append({"source": source, "text": doc, "vector_score": 1 - dist,
                     "rrf_score": None, "rrf_rank": None, "rerank_score": None, "rerank_ms": None,
                     "filter": LAST_FILTER, "filter_status": st,
                     "via": f"graph: {' + '.join(e[1] for e in entities)} → {person[1]}", "id": cid})
    reranker = get_reranker()
    if reranker and hits:
        scores = reranker.model.predict([(query, bm25_text(CHUNKS_BY_ID[h["id"]])) for h in hits], show_progress_bar=False)
        for h, s in zip(hits, scores):
            h["rerank_score"] = float(s)
    hits.sort(key=lambda h: (h["filter_status"] not in (None, "match"),       # confirmed matches first
                             -(h["rerank_score"] if h["rerank_score"] is not None else h["vector_score"])))
    for rank, h in enumerate(hits, 1):
        h["rank"] = rank
        h["score"] = h["rerank_score"] if h["rerank_score"] is not None else h["vector_score"]
        del h["id"]
    print(f"graph: {[e[1] for e in entities]} -> {len(hits)} candidates")
    return hits
