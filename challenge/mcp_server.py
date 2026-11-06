"""
STEP 8 · The CV chatbot as an MCP server (for Claude Code, Claude Desktop or Kiro)

Two read-only tools on top of challenge/pipeline.py:
  search_cvs(question)  the hybrid / graph retrieval of the pipeline, with its refusal check
  get_cv(file_name)     one CV's text, by the file name search_cvs returns

The assistant's own model writes the answer; this server only finds evidence. Contact details
(emails, phone numbers, links, address lines) are removed from everything it returns.

Run it by hand to check it starts:   .venv/bin/python challenge/mcp_server.py   (waits for a client; Ctrl+C)
Connect it:                          .mcp.json → "cv-chatbot", then /mcp in Claude Code

Privacy: everything runs locally. The Chroma index is built in a temporary folder that is deleted
when the server stops, so the Gradio app's challenge/chroma_db is never touched and no CV data is
written into the repo. CV_DIR (default challenge/cvs) is the only folder get_cv can read.
"""
import atexit
import contextlib
import os
import pathlib
import re
import shutil
import sys
import tempfile

os.environ.setdefault("HF_HUB_OFFLINE", "1")             # use the cached models only, never download
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from mcp.server.fastmcp import FastMCP  # noqa: E402

with contextlib.redirect_stdout(sys.stderr):               # stdout is reserved for the MCP protocol
    import pipeline  # noqa: E402

CV_DIR = pathlib.Path(os.getenv("CV_DIR", HERE / "cvs")).resolve()
MAX_QUESTION = 500
MAX_TEXT = 1200                                           # characters of each passage search_cvs returns
CONTACT = [
    re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),                                    # emails
    re.compile(r"(?:https?://|www\.)\S+|\b(?:linkedin|github)\.com/\S*", re.I),   # links / profiles
    re.compile(r"\+?\d[\d\s\-().]{7,}\d"),                                        # phone numbers
    re.compile(r"(?im)^\s*(?:address|home address|street)\s*[:\-].*$"),           # address lines
]


def redact(text):
    """Remove contact details before anything leaves the server."""
    for pattern in CONTACT:
        text = pattern.sub("[redacted]", text)
    return text


def setup():
    """Read the CVs, chunk them, index them in a temporary Chroma folder, build the graph."""
    tmp = tempfile.mkdtemp(prefix="cv-chatbot-mcp-")
    atexit.register(shutil.rmtree, tmp, ignore_errors=True)
    pipeline.HERE = pathlib.Path(tmp)                     # chroma_db goes here, not into challenge/
    with contextlib.redirect_stdout(sys.stderr):
        docs = [d for d in pipeline.load_cvs(CV_DIR)
                if (CV_DIR / d["source"]).resolve().parent == CV_DIR]     # skip links that point out of CV_DIR
        chunks =[c for d in docs for c in pipeline.chunk(d)]
        if chunks:
            pipeline.build_index(chunks)
            pipeline.build_graph(chunks)
    print(f"cv-chatbot: {len(docs)} CVs, {len(chunks)} chunks from {CV_DIR}", file=sys.stderr)
    return {d["source"]: d for d in docs}


DOCS = setup()
mcp = FastMCP("cv-chatbot")


@mcp.tool()
def search_cvs(question: str) -> dict:
    """Search the candidate CVs with the challenge pipeline (hybrid search, reranking, knowledge graph).
    Returns ranked passages, each with the CV file name, the candidate name (when verified) and how it was
    found. "Which candidates know X?" returns EVERY matching candidate from the knowledge graph.
    refused=true means no CV contains evidence for the question: do not answer from memory."""
    if not isinstance(question, str) or not question.strip():
        return {"error": "question must be a non-empty string"}
    question = question.strip()[:MAX_QUESTION]
    if not DOCS:
        return {"error": f"no CVs found in {CV_DIR.name}/"}
    with contextlib.redirect_stdout(sys.stderr):
        query = pipeline.make_query(question, [])
        hits = pipeline.retrieve(query, pipeline.K)
        refused = bool(pipeline.should_refuse(query, hits))
    if refused:
        return {"question": question, "refused": True, "message": pipeline.REFUSAL, "results": []}
    results = [{"rank": h["rank"], "file_name": h["source"], "candidate": pipeline.verified_name(h["source"]) or "",
                "found_by": h.get("via", ""), "text": redact(pipeline.embedding_text(
                    {"text": h["text"], "name": pipeline.verified_name(h["source"]) or ""}))[:MAX_TEXT]}
               for h in hits]
    return {"question": question, "refused": False, "count": len(results), "results": results}


@mcp.tool()
def get_cv(file_name: str) -> dict:
    """Return the text of ONE CV by its exact file name (as given by search_cvs), contact details removed.
    Only files of the CV folder can be read; anything else returns an error."""
    name = file_name.strip() if isinstance(file_name, str) else ""
    if not name or len(name) > 255 or name.startswith(".") or any(ch in name for ch in "/\\\x00"):
        return {"error": "invalid file name: give a plain CV file name, e.g. from search_cvs"}
    path = (CV_DIR / name).resolve()                      # follows symlinks: a link out of CV_DIR fails below
    if path.parent != CV_DIR or name not in DOCS:          # no traversal, only indexed CV files
        return {"error": f"no CV named {name!r}: use search_cvs to find file names"}
    doc = DOCS[name]
    chunk = next((c for c in pipeline.CHUNKS_BY_ID.values() if c["source"] == name), {})
    return {"file_name": name, "candidate": pipeline.verified_name(name) or "",
            "text": redact(doc["text"]), "characters": len(doc["text"]),
            "years_of_experience": chunk.get("years", -1), "years_source": chunk.get("years_source", "none")}


if __name__ == "__main__":
    mcp.run()                                             # stdio transport
