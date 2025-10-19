# tools/build_stopwords_eu.py

from pathlib import Path
import pickle

EU_LANGS = [
    "bg","cs","da","de","el","en","es","et","fi","fr","ga","hr",
    "hu","it","lt","lv","mt","nl","pl","pt","ro","sk","sl","sv"
]

# ---- Loaders ----
def load_iso(lang: str) -> set[str]:
    from stopwordsiso import stopwords as iso_sw
    try:
        return set(iso_sw(lang) or [])
    except Exception:
        return set()

def load_nltk(lang: str) -> set[str]:
    # map ISO -> NLTK language names where available
    MAP = {
        "da":"danish","nl":"dutch","en":"english","fi":"finnish","fr":"french",
        "de":"german","hu":"hungarian","it":"italian","no":"norwegian",
        "pt":"portuguese","ro":"romanian","ru":"russian","es":"spanish","sv":"swedish",
        "el":"greek"
    }
    name = MAP.get(lang)
    if not name:
        return set()
    try:
        from nltk.corpus import stopwords as sw
        return set(sw.words(name))
    except Exception:
        return set()

def domain_adjust(sw: set[str]) -> set[str]:
    # Keep negations and some function words useful for queries
    for keep in ("no", "not", "nor", "via", "per"):
        if keep in sw:
            sw.remove(keep)
    # Optional: add any project-specific stop words:
    sw.update({"etc"})  # example
    return sw

def main():
    out_path = Path("artifacts/stopwords_eu.pkl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = {}
    all_union = set()
    for lang in EU_LANGS:
        iso = load_iso(lang)
        nk  = load_nltk(lang)
        merged = domain_adjust(iso.union(nk))
        bundle[lang] = merged
        all_union |= merged

    bundle["all_union"] = all_union
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)

    print(f"[stopwords] Saved {out_path} with {len(bundle)-1} languages, "
          f"union size={len(all_union)}")

if __name__ == "__main__":
    main()
