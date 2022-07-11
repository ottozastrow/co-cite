import json
import random
import os
from pkg_resources import working_set
import tqdm
import re


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
    for p in tqdm.tqdm(range(1, 2000)):
        
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


def search_string(
        text:str, offset:int, citation_prefix:str, 
        splittext:str, regex_prefix:str,
        regex_postfix:str, is_numerical=False
    ):
    # find indices of splittext in text
    split_offsets = list(find_all(text, splittext))

    split_offsets = [0] + split_offsets + [len(text)]
    splits = text.split(splittext)
    splits = [splits[0]] + [splittext + split for split in splits[1:]]
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
        regex_pattern = re.compile(regex_prefix + str(counter) + regex_postfix)
        span = regex_pattern.search(text, split_offsets[split_index], split_offsets[split_index+1])
        
        if span is not None:
            results[counter] = {
                "start": span.start() + offset, 
                "end": split_offsets[split_index+1] + offset,
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
                if counter > 2000:  # type: ignore
                    split_index += 1
                    counter = latest_found_pargraph
                else:
                    skipping_p_list.append(counter)
                    counter = counter + 1  # type: ignore
            else:
                if ord(str(counter)) > 25:
                    counter = latest_found_pargraph
                    split_index += 1
                    counter = "a"
                else:
                    skipping_p_list.append(counter)  # type: ignore
                    # next character in alphabet
                    latest_found_pargraph = counter
                    counter = chr(ord(str(counter)) + 1)
    # print("skipped: ", skipping_p_list)
    
    # change dictionary keys to citation string
    citations = {}
    for p in results.keys():
        citations[citation_prefix + str(p)] = results[p]
    return citations

def parse_uscode_v2(fulltext:str):
    """ return dictionary with paragraphs as index and text as value """

    spans_paragraphs = search_string(
        fulltext,
        offset=0,
        citation_prefix="U.S.C.A. 38 §",
        splittext='<h3 class="section-head">§',
        regex_prefix='<h3 class="section-head">§',
        regex_postfix=r".",
        is_numerical=True
    )
    spans_sections = {}
    for p in spans_paragraphs.keys():
        span = spans_paragraphs[p]
        
        # spans_sections.update(search_string(
        #     fulltext[span["start"]: span["end"]],
        #     offset=span["start"],
        #     citation_prefix=" (" + str(p) + ")",
        #     splittext=r'<p class="statutory-body">(',
        #     regex_prefix=r'<p class="statutory-body">(',
        #     regex_postfix=r")",
        #     is_numerical=False
        # ))
        # print("lens", len(spans_sections))
    
    print("num of paragraphs: ", len(spans_paragraphs))

    for current in spans_paragraphs.keys():
        span = spans_paragraphs[current]
        break
        print(fulltext[span["start"]:span["end"]])
    print(spans_paragraphs.keys())
    # print(len(spans_sections))
    return spans_paragraphs


def statutory_kb_lookup(fulltext: str, statutory_kb:dict, citation:str) -> str:
    """ return text of statutelkb for citation """
    if citation in statutory_kb.keys():
        span = statutory_kb[citation]
        text = fulltext[span["start"]:span["end"]]
        # remove all html class tags
        text = re.sub(r' class=".*?"', "", text)
        return text
    else:
        return None


# print(statutory_kb_lookup(fulltext, statutory_kb, "U.S.C.A. 38 §103"))
# parse_uscode()


def build_dataset():
    filedir = "data/USCODE-2020-title38.htm"

    paragraph_lengths = []
    fulltext = ""
    # read file into string
    with open(filedir, "r") as f:
        fulltext = f.read()

    # remove all html comments
    fulltext = re.sub(r"<!--.*?-->", "", fulltext)

    # remove multiple empty lines
    fulltext = re.sub(r"\n\n\n+", "\n\n", fulltext)
    fulltext = re.sub(r"\n\n", "\n", fulltext)


    print(fulltext[:600])
    statutory_kb = parse_uscode_v2(fulltext)
    prefix = "U.S.C.A. 38 §"
    for i in range(103,1500):
        res = statutory_kb_lookup(fulltext, statutory_kb, prefix + str(i))
        if res:
            print(i, len(res))
            paragraph_lengths.append(len(res))
            print(res)
            break
build_dataset()