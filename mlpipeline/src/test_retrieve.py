from retrieve import reinsert_citations


def test_reinsert_citations():
    data = {
        "txt": "This is a text with @cit@ tags",
        "citation_texts": ["38 C.F.R. §38.1"]
    }
    res = reinsert_citations(data)
    assert res == "This is a text with 38 C.F.R. §38.1 tags", res