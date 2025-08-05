import os


def file_iterator(corpus, endat, ext):
    cnt = 0
    for dirname, dirs, files in os.walk(corpus):
        for f in sorted(files):
            if cnt >= endat and endat > 0:
                return
            elif f.split('.')[-1] == ext:
                cnt += 1
                yield (cnt, dirname, f)


def file_read(path):
    if not os.path.isfile(path):
        return None
    with open(path, 'r') as fh:
        return fh.read()


def corpus_length__arqmath3_rawxml(xml_file, max_items):
    from xmlr import xmliter
    cnt = 0
    for attrs in xmliter(xml_file, 'row'):
        if cnt + 1 > max_items and max_items > 0:
            return max_items
        cnt += 1
    return cnt


def corpus_reader__arqmath3_rawxml(xml_file, preserve_formula_ids=False):
    from xmlr import xmliter
    from bs4 import BeautifulSoup
    from maxsim.gen_topic import replace_dollar_tex
    def html2text(html, preserve):
        soup = BeautifulSoup(html, "html.parser")
        for elem in soup.select('span.math-container'):
            if not preserve:
                elem.replace_with('[imath]' + elem.text + '[/imath]')
            else:
                formula_id = elem.get('id')
                if formula_id is None:
                    elem.replace_with(' ')
                else:
                    elem.replace_with(
                        f'[imath id="{formula_id}"]' + elem.text + '[/imath]'
                    )
        return soup.text
    def comment2text(html):
        soup = BeautifulSoup(html, "html.parser")
        return replace_dollar_tex(soup.text)

    if 'Posts' in os.path.basename(xml_file):
        for attrs in xmliter(xml_file, 'row'):
            sign = 0
            if '@Body' not in attrs:
                body = None
            else:
                if "align" in attrs['@Body']:
                    sign = 1
                    print("original doc:", attrs['@Body'])
                body = html2text(attrs['@Body'], preserve_formula_ids)
            ID = attrs['@Id']
            vote = attrs['@Score']
            postType = attrs['@PostTypeId']
            if postType == "1": # Question
                title = html2text(attrs['@Title'], preserve_formula_ids)
                tags = attrs['@Tags']
                tags = tags.replace('-', '_')
                if '@AcceptedAnswerId' in attrs:
                    accept = attrs['@AcceptedAnswerId']
                else:
                    accept = None
                # YIELD (docid, doc_props), contents
                yield (ID, 'Q', title, body, vote, tags, accept), None
            else:
                assert postType == "2" # Answer
                parentID = attrs['@ParentId']
                # YIELD (docid, doc_props), contents
                if sign == 1:
                    print("processed doc is:", body)
                yield (ID, 'A', parentID, vote), body

    elif 'Comments' in os.path.basename(xml_file):
        for attrs in xmliter(xml_file, 'row'):
            if '@Text' not in attrs:
                comment = None
            else:
                comment = comment2text(attrs['@Text'])
            ID = attrs['@Id']
            answerID = attrs['@PostId']
            # YIELD (docid, doc_props), contents
            yield (answerID, 'C', ID, comment), None
    else:
        raise NotImplemented


def corpus_reader__jsonl(jsonl_path, fields):
    import json
    import pdb
    fields = eval(fields)
    with open(jsonl_path, 'r') as fh:
        for line in fh:
            line = line.rstrip()
            j = json.loads(line)
            values = [j[f] for f in fields]
            # YIELD (docid, doc_props), contents
            yield tuple(values[:-1]), values[-1]
