from preprocess import preprocess_for_transformer_old as process
# from get_posts import get_posts
from custom_tokenize import LatexTokenizer
from custom_tokenize import *
from math_latex import *
import ARQMathCode.topic_file_reader as ARQ
import bs4 as bs

tok = LatexTokenizer()
reader = ARQ.TopicReader('./topic/Topics_Task1_2022_V0.1.xml')
longest_len = 99999

def get_posts(body):
    soup = bs.BeautifulSoup(body, "lxml")
    result = ""
    for element in soup.recursiveChildGenerator():
        if element.name == 'span' and 'math-container' in element.get('class', []):
            math_text = element.get_text(strip=True).strip().strip('.').strip('$')
            # if math_text.startswith('$') and math_text.endswith('$'):
            #     math_text = math_text[1:-1]
            result += f"[imath]{math_text}[/imath]"
        elif isinstance(element, str):
            parent = element.parent
            if parent.name == 'span' and 'math-container' in parent.get('class', []):
                continue
            result += element
        if hasattr(element, "tail") and element.tail:
            result += element.tail
    return result.strip()

# print(reader.map_topics.keys)
id_count = 0
for id in reader.map_topics.keys():
    #find title formulas
    title_p = get_posts(reader.get_topic(id).title)   
    standard_post = process(title_p)                   
    if len(standard_post) <= longest_len:
        processed_title = standard_post
    else:
        processed_title = standard_post[:longest_len-1]
    #find question formulas
    question_p = get_posts(reader.get_topic(id).question)
    standard_post = process(question_p)                   
    if len(standard_post) <= longest_len:
        processed_question = standard_post
    else:
        processed_question = standard_post[:longest_len-1]
    post = processed_title + processed_question
    with open(file='./data_processing/processed/topic_post_pya0_v2.txt', mode='a', encoding='utf-8') as file:
        # print(' '.join(post))
        file.write(post + '\t' + f'{id}' + '\n')
print(id_count)

