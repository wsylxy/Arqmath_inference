
import re
import os
import json

def replace_dollar_tex(s):
	l = len(s)
	i, j, stack = 0, 0, 0
	new_txt = ''
	while i < l:
		if s[i] == "\\" and (i + 1) < l:
			if s[i + 1] == '$':
				# skip if it is escaped dollar
				new_txt += '$'
				i += 1
			elif stack == 0:
				# otherwise just copy it
				new_txt += s[i]
		elif s[i] == '$':
			if stack == 0: # first open dollar
				stack = 1
				j = i + 1
			elif stack == 1: # second dollar
				if i == j:
					# consecutive dollar
					# (second open dollar)
					stack = 2
					j = i + 1
				else:
					# non-consecutive dollar
					# (close dollar)
					stack = 0
					# print('single: %s' % s[j:i])
					new_txt += '[imath]%s[/imath]' % s[j:i]
			else: # stack == 2
				# first close dollar
				stack = 0
				# print('double: %s' % s[j:i])
				new_txt += '[imath]%s[/imath]' % s[j:i]
				# skip the second close dollar
				i += 1
		elif stack == 0:
			# non-escaped and non enclosed characters
			new_txt += s[i]
		i += 1
	return new_txt

def replace_alignS_tex(s, prefix='align'):
	# replace '\begin{align*} * \end{align*}'
	regex = re.compile(r'\\begin{' + prefix + r'.*}(.+)\\end{' + prefix + r'.*}')
	return re.sub(regex, r"[imath]\1[/imath]", s)

def topic_process(xmlfile):
    from xmlr import xmliter
    from bs4 import BeautifulSoup
    print(xmlfile)
    for attrs in xmliter(xmlfile, 'Topic'):
        sign = 0
        qid = attrs['@number']
        title = attrs['Title']
        post_xml = title + '\n' + attrs['Question']
        s = BeautifulSoup(post_xml, "html.parser")
        if "align" in s.text:
            sign = 1
            print("original doc", s.text)
        post = replace_dollar_tex(s.text)
        post = replace_alignS_tex(post)
        if sign == 1:
              print("processed doc", post)
        query = [{
            'type': 'term',
            'str': post
        }]
        yield qid, query, None

def _topic_process(xmlfile):
    for qid, query, _ in topic_process(xmlfile):
        yield qid, query, None

def gen_topics_queries(collection, qfilter=None):
    # func_name = '_topic_process__' + collection.replace('-', '_')
    # handler = getattr(collection_driver, func_name)
    curdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prefix = f'{curdir}/data/topics.{collection}'
    print(f'Searching topics file at: {prefix} ...')
    found = False
    for src in [f'{prefix}.{ent}' for ent in ['txt', 'json', 'xml']]:
        if not os.path.exists(src):
            continue
        else:
            found = True
        ext = src.split('.')[-1]
        if ext == 'txt':
            with open(src, 'r') as fh:
                for i, line in enumerate(fh):
                    line = line.rstrip()
                    qid, query, args = _topic_process(i, line)
                    if qfilter:
                        query = list(filter(qfilter, query))
                    yield qid, query, args
        elif ext == 'json':
            with open(src, 'r') as fh:
                qlist = json.load(fh)
                for i, json_item in enumerate(qlist):
                    qid, query, args = _topic_process(i, json_item)
                    if qfilter:
                        query = list(filter(qfilter, query))
                    yield qid, query, args
        elif ext == 'xml':
            for qid, query, args in _topic_process(src):
                yield qid, query, args
    if not found:
        raise ValueError(f'Unrecognized index name {collection}')