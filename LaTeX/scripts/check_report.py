"""Check report scope and assets; the ratio is an approximate prose-word count."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def prose_count(text):
    # Exclude code, displayed equations, tables, figures, labels and citations.
    # Keep prose, subsection names and inline non-mathematical formatting.
    for environment in ["lstlisting", "equation", "align", "table"]:
        text = re.sub(r"\\begin\{" + environment + r"\}.*?\\end\{" + environment + r"\}",
                      "", text, flags=re.S)
    text = re.sub(r"\\reportfigure\{[^}]+\}.*?\{fig:[^}]+\}", "", text, flags=re.S)
    text = re.sub(r"\\reservedfigure\{[^}]*\}\{[^}]*\}\{[^}]*\}\{[^}]*\}", "", text)
    text = re.sub(r"\\(?:cite|label|cref|Cref)\{[^}]*\}", "", text)
    text = re.sub(r"\$[^$]*\$", "", text)
    text = re.sub(r"\\[A-Za-z]+\*?(?:\[[^\]]*\])?", "", text)
    return len(re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", text))


def main():
    source = (ROOT / "main.tex").read_text()
    body = source.split(r"\section{Introduction}", 1)[1].split("\\clearpage\n\\appendix", 1)[0]
    primary, rest = body.split(r"\section{Evolution Beyond YOLOv1}", 1)
    evolution, conclusion = rest.split(r"\section{Conclusion}", 1)
    primary_words = prose_count(primary) + prose_count(conclusion)
    evolution_words = prose_count(evolution)
    share = primary_words / (primary_words + evolution_words)
    figure_paths = re.findall(r"\\reportfigure\{([^}]+)\}", source)
    assert len(figure_paths) == 5
    assert all((ROOT / "figures" / name).exists() for name in figure_paths)
    reserved = re.findall(r"\\reservedfigure\{[^}]*\}\{[^}]*\}\{[^}]*\}\{([^}]+)\}", source)
    assert set(reserved) == {"fig:architecture", "fig:demo"}
    demo = source.split(r"\section{Simple Demonstration / Implementation Example}", 1)[1].split(
        r"\section{Evolution Beyond YOLOv1}", 1)[0]
    assert "The implementation has not yet been performed." in demo
    assert not re.search(r"synthetic|results.json|executed implementation|lstlisting|tab:demo", source, re.I)
    bib = (ROOT / "references.bib").read_text()
    keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    citations = {key.strip() for group in re.findall(r"\\cite\{([^}]+)\}", source)
                 for key in group.split(",")}
    assert citations <= keys, f"Missing bibliography keys: {citations - keys}"
    assert 0.79 <= share <= 0.82, f"Review scope ratio: {share:.1%}"
    print(f"YOLOv1-focused main prose: {primary_words} words; evolution: {evolution_words} words.")
    print(f"Approximate scope balance: {share:.1%} / {1-share:.1%}; appendix/references excluded.")
    print(f"Verified {len(figure_paths)} active figures, two reserved figures, {len(citations)} citation keys, and pending Section 7.")


if __name__ == "__main__":
    main()
