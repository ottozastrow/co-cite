import re

def book_from_statute(statute) -> str:
    """
    returns the book from a statute
    38 U.S.C.A. §§ 5100, 5102, 5103,  -> 38u.s.c.a.
    if only u.s.c. is given, return u.s.c.a.
    # some statutes are missing their §. e.g. 38 C.F.R. 3.321(b)(1)
    """
    statute = statute.replace(" ", "").lower()

    if len(statute.split("§")) == 1:
        statute = statute.replace("38u.s.c.a.", "38u.s.c.a.§")
        statute = statute.replace("38cfr", "38cfr§")
        statute = statute.replace("38c.f.r.", "38c.f.r.§")

    book = statute.split("§")[0]
    if book == "38u.s.c.":
        book = "38u.s.c.a."
    book_no_dot = book.replace(".", "")
    if "cfr" in book_no_dot:
        book = "38 C.F.R."
    elif "usca" in book_no_dot:
        book = "38 U.S.C.A."
    return book


def remove_last_numeric_brackets(x):
    """
    removes last bracket from inputs, if the content is numeric
    """
    if len(x) > 2 and x[-1] == ")" and x[-2].isnumeric():
        x = x.rsplit("(", 1)[0]
    return x


def sections_from_statute(
        statute,
        remove_subsubsections,
        remove_subsections=True
        ) -> list[str]:
    """
    returns the book from a statute
    38 U.S.C.A. §§ 5100, 5102, 5103A,  -> [5100, 5102, 5103A]
    """
    statute = statute.replace(" ", "").lower()

    if len(statute.split("§")) == 1:
        statute = statute.replace("38u.s.c.a.", "38u.s.c.a.§")
        statute = statute.replace("38cfr", "38cfr§")
        statute = statute.replace("38c.f.r.", "38c.f.r.§")

    sections = statute.split("§")[-1]
    sections = sections.split(",")
    if remove_subsubsections:
        # remove trailing brackets from sections if the content is numeric
        sections = [remove_last_numeric_brackets(section)
                    for section in sections]

    # remove leftover brackets (but leave content)
    sections = [section.replace("(", "").replace(")", "")
                for section in sections
                if section != ""]

    if remove_subsections:
        new_sections = []
        for section in sections:
            # if last letter not numeric
            if section[-1] not in "0123456789":
                section = section[:-1]
            new_sections.append(section)
        sections = new_sections

    assert len(sections) > 0, "no sections found in statute"
    for section in sections:
        assert(len(section)) > 0, "section is empty"
    return sections


def normalize_section(section):
    return section.replace(" ", "").replace("(", "").replace(")", "").lower()


def normalize_case(inputs_orig):
    # remove trailing brackets from sections if the content is numeric
    # e.g. (1998)
    inputs = remove_last_numeric_brackets(inputs_orig)

    # check if inputs begins with see
    lowered_inputs = inputs.lower()

    law_categories = [
        ("Vet. App.", r"vet\.? ?app\.?"),
        ("F.3d", r"f\.? ?3d?"),
        ("F.2d", r"f\.? ?2d?"),
        ("F.d", r"f\.? ?d"),
        ("F. supp", r"f\.? ?supp.?"),
        ("Fed. Cir.", r"fed\.? ?cir\.?"),
    ]
    patterns = []
    for category, regex in law_categories:
        patterns.append((category, re.compile(regex)))

    """
    See Combee v. Brown, 34 F.3d 1039, 1042
    See Combee v. Brown, 34 F. 3
    normalized [combeevbrown34 f3]
    there are many such examples, where F.3d and F.3 are used interchangebly
    """
    # TODO support multiple categories per citation (always use the first one)

    for category, pattern in patterns:
        span = pattern.search(lowered_inputs)
        if span:
            participants = inputs[:span.start()].strip()

            details = inputs[span.end():].strip()
            details = details.replace(" ", "")
            details = details.lower()

            details = details.split("-")[0]
            details = details.split("(")[0]
            details = details.split(",")[0]

            # remove trailing symbols
            details = details.replace("(", "")
            details = details.replace(")", "")
            details = details.replace("-", "")
            details = details.replace(".", "")
            details = details.replace(",", "")

            inputs = participants + " " + category + " " + details
            break

    return inputs


def split_citation_segments(
        inputs,
        remove_subsubsections=True,
        remove_subsections=False) -> list[str]:
    """
    splits a citation into segments
    38 U.S.C.A. §§ 5100A, 5102(a)(1) becomes
    [38 U.S.C.A. § 5100a, 38 U.S.C.A. § 5102a]

    or if its not a statute

    transforms from
    See Moreau v. Brown, 9 Vet. App. 389, 396 (1996)
    to
    [Moreau v. Brown, 9 Vet. App. 389, 396]
    explanations:
    - "see" or trailing commas are not part
    of the citation but often in the data
    - year numbers (1997) are not always present
    - in Vet. App. 389, 396  we have the appendix
    389 but page number 396. the page number is often not present
    """

    # regex pattern for if "see" or "eg" "in" is at beginning of string
    # has to work for "See, " "e.g. ", "See in ", ...
    useless_prefix = re.compile(r"((see|e\.g\.|in)\.?\,? )+")

    useless_prefix_span = useless_prefix.search(inputs.lower())
    if useless_prefix_span:
        if useless_prefix_span.start() == 0:
            inputs = inputs[useless_prefix_span.end():]
    try:
        x = inputs.lower()
        is_case = "§" not in x\
            and "c.f.r" not in x\
            and "u.s.c.a" not in x\
            and "u.s.c." not in x
        if not is_case:
            book = book_from_statute(inputs)
            sections = sections_from_statute(
                inputs,
                remove_subsubsections,
                remove_subsections=remove_subsections)
            segments = [book + " § " + section for section in sections]
            return segments
        else:
            inputs = normalize_case(inputs)
            return [inputs]
    except Exception as e:
        print(e)
        print("inputs:", inputs)
        return [inputs]
