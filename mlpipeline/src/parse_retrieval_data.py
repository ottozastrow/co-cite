import json
import random
import os
from pkg_resources import working_set
import tqdm
import re
import numpy as np
from fuzzysearch import find_near_matches
import pandas as pd


def generate_squad_json(questions, answers, contexts, filepath):
    """
    squad format:
        FeaturesDict({
        'answers': Sequence({
            'answer_start': tf.int32,
            'text': Text(shape=(), dtype=tf.string),
        }),
        'context': Text(shape=(), dtype=tf.string),
        'id': tf.string,
        'is_impossible': tf.bool,
        'plausible_answers': Sequence({
            'answer_start': tf.int32,
            'text': Text(shape=(), dtype=tf.string),
        }),
        'question': Text(shape=(), dtype=tf.string),
        'title': Text(shape=(), dtype=tf.string),
    })

    save jsonl where each line is a json dict
    """
    # if file exists delete it
    if os.path.exists(filepath):
        os.remove(filepath)
    
    # create file
    # generate random int
    random_int = random.randint(0, 100000)
    all_dicts = {"data":[]}
    for i in range(len(questions)):
        question = questions[i]
        answer = answers[i]
        context = contexts[i]
        docdict = {
            "paragraphs":[{
                "context": context, 
                "id": str(i + random_int), 
                "title": str(i),
                "is_impossible": False,
                "qas":[{
                    "id": str(i),
                    "question": question,
                    "answers": [{
                        "text": answer,
                        "answer_start": context.find(answer)
                    }], 
                }]
            }],
        }
        import pdb
        all_dicts["data"].append(docdict)

    with open(filepath, "w") as f:
        f.write(json.dumps(all_dicts))


def build_dpr_dataset(args):
    """
    every sample in the output has this format:
    {
        "dataset": str,
        "question": str,
        "answers": list of str
        "positive_ctxs": list of dictionaries of format {'title': str, 'text': str, 'score': int, 'title_score': int, 'passage_id': str}
        "negative_ctxs": []
        "hard_negative_ctxs": list of dictionaries of format {'title': str, 'text': str, 'score': int, 'title_score': int, 'passage_id': str}
    }

    input data is parquet format
    for every context-citation pair we need a list of all document sections which contain the correct answer
    because some contexts have thousands of correct answers we instead focus on the source documents.
    for that we require a mapping between citation string and document id

    in the case of statutes:
    - create a function that generates citation strings from raw statues and a DB which indexes by citation string
    - for every context-citation pair we lookup the document id or fulltext of the citation

    """

def parse_uscode():
    """ return dictionary with paragraphs as index and text as value """
    filedir = "data/USCODE-2020-title38.htm"
    return_dict = {}
    
    # read file into string
    with open(filedir, "r") as f:
        fulltext = f.read()

    p_start_indexes = {0:0}
    last_index = 0
    missing_paragraphs = []
    last_found_p = 0
    for p in tqdm.tqdm(range(1, 10000)):
        
        # find the index of the first occurrence of § + str(p) + non-number
        # compute pi_start via regex pattern
        # write a regex pattern that finds § + str(p) + non-number
        pattern = re.compile("§" + str(p) + r".")

        # find first occurence of pattern in fulltext starting at last_index
        span = pattern.search(fulltext, pos=last_index)
        
        if span is None:  # if p was not found
            print("could not find paragraph " + str(p))
            missing_paragraphs.append(p)
        else:  # if p is found
            pi_start = span.start()
            last_index = pi_start
            p_start_indexes[p] = pi_start
    
            # add last iterations paragraph to dictionary
            if p>1:
                pi_start = p_start_indexes[last_found_p]
                pi_end = p_start_indexes[p]
                
                return_dict[last_found_p] = fulltext[pi_start:pi_end]
            last_found_p = p
    print("num of paragraphs: ", len(return_dict))
    return return_dict

def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub) # use start += 1 to find overlapping matches


def str_to_sections(
        text:str, offset:int, citation_prefix:str, 
        citation_postfix:str,
        regex_splittext:str, regex_prefix:str,
        regex_postfix:str, is_numerical=False,
        child_of=None,
        chapter_prefix="",
    ):
    python_splittext = regex_splittext.replace('\\',"")
    # find indices of splittext in text
    split_offsets = list(find_all(text, python_splittext))

    split_offsets = [0] + split_offsets + [len(text)]
    splits = text.split(python_splittext)
    splits = [splits[0]] + [regex_splittext + split for split in splits[1:]]
    skipping_p_list = []
    results = {}
    counter = 102 if is_numerical else "a"  # counts current paragraph
    # print("splits len", len(splits))
    split_index = 0
    latest_found_pargraph = 0

    while True:
        if split_index >= len(splits):
            break
        split = splits[split_index]
        # print(split_index, counter, latest_found_pargraph)
        # find first occurence of regex_index in split
        search_string = regex_prefix + str(counter) + regex_postfix
        regex_pattern = re.compile(search_string)
        span = regex_pattern.search(text, split_offsets[split_index], split_offsets[split_index+1])
        
        if span is not None:
            results[counter] = {
                "start": span.start() + offset, 
                "end": split_offsets[split_index+1] + offset,
                "child_of": child_of,
            }
            split_index += 1
            if is_numerical:
                counter += 1
            else:
                counter = chr(ord(str(counter)) + 1)
            latest_found_pargraph = counter

        else:
            if is_numerical:
                assert type(counter) == int
                if counter > 10000:  # type: ignore  # TODO: remove this hard limit
                    split_index += 1
                    counter = latest_found_pargraph
                else:
                    skipping_p_list.append(counter)
                    counter = counter + 1  # type: ignore
            else:
                if counter == "z":
                    counter = latest_found_pargraph
                    split_index += 1
                    counter = "a"
                else:
                    skipping_p_list.append(counter)  # type: ignore
                    # next character in alphabet
                    latest_found_pargraph = counter
                    counter = chr(ord(str(counter)) + 1)
    
    # change dictionary keys to citation string
    citations = {}
    for p in results.keys():
        citations[citation_prefix + chapter_prefix + str(p) + citation_postfix] = results[p]

    return citations

def parse_uscode_v2(fulltext:str, chapter_prefix:str):
    """ return dictionary with paragraphs as index and text as value """
    
    spans_paragraphs = str_to_sections(
        fulltext,
        offset=0,
        citation_prefix="38 U.S.C.A. § ",
        chapter_prefix=chapter_prefix,
        citation_postfix="",
        regex_splittext='<h3 class="section-head">§',
        regex_prefix='<h3 class="section-head">§',
        regex_postfix=r".",
        is_numerical=True,
        child_of=None,
    )
    spans_sections = {}
    for p in spans_paragraphs.keys():
        span = spans_paragraphs[p]
        
        spans_sections.update(str_to_sections(
            fulltext[span["start"]: span["end"]],
            offset=span["start"],
            citation_prefix=p + "(",
            citation_postfix = ")",
            regex_splittext=r'<p class="statutory-body">\(',
            regex_prefix=r'<p class="statutory-body">\(',
            regex_postfix=r"\)",
            is_numerical=False,
            child_of=str(p),
        ))
    print("number of paragraphs: ", len(spans_paragraphs))
    print("number of subsections", len(spans_sections))
    
    spans_paragraphs.update(spans_sections)
    # print(len(spans_sections))
    return spans_paragraphs


def statutory_kb_lookup(fulltext: str, statutory_kb:dict, citation:str) -> str:
    """ return text of statutelkb for citation """
    if citation in statutory_kb.keys():
        span = statutory_kb[citation]
        text = fulltext[span["start"]:span["end"]]
        
        return text
    else:
        return None


def normalize_section(text:str) -> dict:
    string_editorial_notes = '<h4 class="note-head"><strong>Editorial Notes</strong></h4>'
    string_source_credit = '<p class="source-credit">'
    string_amendments = '<h4 class="note-head">Amendments</h4>'
    text = text.split(string_editorial_notes)[0]
    text = text.split(string_source_credit)[0]
    text = text.split(string_amendments)[0]
    text = re.sub(r' class=".*?"', "", text)
    return text

def normalize_document(fulltext):
    # remove all html comments
    fulltext = re.sub(r"<!--.*?-->", "", fulltext)

    # remove multiple empty lines
    fulltext = re.sub(r"\n\n\n+", "\n\n", fulltext)
    fulltext = re.sub(r"\n\n", "\n", fulltext)
    return fulltext

def parse_title(text:str, child_of, statutory_kb) -> str:
    if child_of is not None:
        try:
            return statutory_kb[child_of]["title"]
        except:
            return "<no title parsed wtg>"
    else:
        # use regex to find string between '<h3> and </h3>'
        try:
            title = re.search(r'<h3 class="section-head">(.*?)</h3>', text).group(1)
        except:
            print("problem with title")
            title ="<no title parsed>"
            # remove all html class tags
        return title

def fuzzy_citation_search(search_str, sections, keys):
    keys = " "*20 + keys + " "*10
    results = find_near_matches(search_str, keys, max_l_dist=0)
    citations = [keys[k.start-20:k.end+4] for k in results]
    citations = [s.split(";")[1].strip() for s in citations]

    # get title
    titles = [sections[s]["title"] for s in citations]

    # zip titles and citations
    results = list(zip(citations, titles))
    return results


def normalize_to_parsed_citation(key):
    """the citations encountered in the dataset need to be transformed to match those from the parsed statutes"""
    key = key.replace("38 U.S.C.A. §§", "")
    key = key.replace("38 U.S.C.A. §", "")
    key = key.replace("38 u.s.c.a. §", "")
    key = key.strip()
    # remove content of brackets from string with re, only if the brackets contain numbers
    key = re.sub(r'\([0-9]*\)', "", key)
    # useful for citations like "38 U.S.C.A. § 552 (a)(1)"

    # if last character is a letter, remove it
    if len(key) > 0 and key[-1].isalpha():
        section = key[-1]
        key = key[:-1] + "(" + section.lower() + ")"
    # useful for citations like "38 U.S.C.A. § 552A"
    key = "38 U.S.C.A. § " + key
    return key


def retrieve_usca(sample, sections: pd.DataFrame):
    key = sample["label"]
    # imagine a sample like this: "38 U.S.C.A. §§ 552(a), 523"

    sample["title"] = ""
    sample["sourcetext"] = ""
    segments = key.split(",")
    at_least_one_found = False
    for segment_orig in segments:
        
        segment = normalize_to_parsed_citation(segment_orig)
        found_source = segment in sections.keys()
        if found_source:
            at_least_one_found = True
            retrieved_groundtruth = sections[segment]
            sample["title"] += retrieved_groundtruth["title"] + "[SEP]"
            sample["sourcetext"] += retrieved_groundtruth["sourcetext"] + "[SEP]"

    sample["found_source"] = at_least_one_found
    return sample


def read_files(data_dir):
    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith(".htm")]
    files = sorted(files)
    print("joining", files)
    fulltexts = ["" for i in range(len(files))]
    for i in range(len(files)):
        f = files[i]
        with open(data_dir + f, "r") as file:
            fulltexts[i] = file.read()
    return fulltexts


def load_sources_kb(args):
    """ if available load cache dataset, else rebuild dataset"""
    # check if sourcedatapath ends with /
    assert args.source_data_path[-1] == "/"
    cached_dataset_json_path = args.source_data_path[:-1] + "_cached_dataset.json"

    if args.rebuild_source_kb or not os.path.exists(cached_dataset_json_path):
        # delete cached_dataset_json_path
        if os.path.exists(cached_dataset_json_path):
            os.remove(cached_dataset_json_path)
        
        retrieval_kb = build_sources_kb(args)
        with open(cached_dataset_json_path, "w") as f:
            json.dump(retrieval_kb, f)
        print("saved cache dataset to", cached_dataset_json_path)
    
    else:
        with open(cached_dataset_json_path, "r") as file:
            retrieval_kb = json.load(file)
    return retrieval_kb


def build_sources_kb(args, has_chapter_prefix=False):
    print()
    print("building retrieval dataset")
    fulltexts = read_files(args.source_data_path)

    sections = {}
    paragraph_lengths =  []
    for i in tqdm.tqdm(range(len(fulltexts))):
        fulltext = normalize_document(fulltexts[i])
        prefix = "38 U.S.C.A. § "
        if has_chapter_prefix:
            assert False, "not supported yet"
            chapter_prefix = str(i) + "."
            prefix += str(i+1) + "."
        else:
            chapter_prefix = ""
        statutory_kb = parse_uscode_v2(fulltext, chapter_prefix)

        for p in statutory_kb.keys():
            span = statutory_kb[p]
            text = fulltext[span["start"]:span["end"]]
            normalized = normalize_section(text)
            sections[p] = {
                "sourcetext": normalized,
                "citation": prefix + str(p),
                "child_of": statutory_kb[p]["child_of"],
                "title": parse_title(text, statutory_kb[p]["child_of"], sections)
            }
            
            paragraph_lengths.append(len(sections[p]["text"]))
        
    print(len(sections))
    print("mean paragraph length: ", np.mean(paragraph_lengths))
    print()
    keys = "; ".join(list(sections.keys()))

    # res = fuzzy_citation_search("103", sections, keys)
    return sections

