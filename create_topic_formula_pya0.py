from preprocess import preprocess_for_transformer_old as process
from custom_tokenize import LatexTokenizer
from custom_tokenize import *
from math_latex import *
import ARQMathCode.topic_file_reader as ARQ
import bs4 as bs

tok = LatexTokenizer()
reader = ARQ.TopicReader('./topic/Topics_Task1_2022_V0.1.xml')
longest_len = 200
invalid_ids = []

def get_formulas(body):
    soup = bs.BeautifulSoup(body, "lxml")
    formulas = []
    for math in soup.find_all('span', {'class': "math-container"}):
        formulas.append(math.text)
    return formulas

def check_formula(eq):
    if len(eq)<10:
        return False
    for op in customize_operator:
        if op in eq:
            return False
    return True

print(get_formulas(reader.get_topic('A.301').title))
# print(reader.map_topics.keys)
id_count = 0
for id in reader.map_topics.keys():
    formulas = []
    #find title formulas
    title_f = get_formulas(reader.get_topic(id).title)
    if len(title_f) > 0:
        for tf in title_f:
            standard_formula1 = process('[imath]' + tf.strip().strip('.').strip('$') + '[/imath]')
            standard_formula = re.findall(r"\$.*?\$", standard_formula1)    #convert str into a list
            if check_formula(standard_formula):                    
                formulas.append(standard_formula)
    #find question formulas
    question_f = get_formulas(reader.get_topic(id).question)
    if len(question_f) > 0:
        for qf in question_f:
            qf = qf.strip('$')
            standard_formula1 = process('[imath]' + qf.strip().strip('.').strip('$') + '[/imath]')
            standard_formula = re.findall(r"\$.*?\$", standard_formula1)    #convert str into a list
            if check_formula(standard_formula):
                formulas.append(standard_formula)
                    
    if len(formulas) > 0:
        id_count += 1
    else:
        invalid_ids.append(id)
    with open(file='./data_processing/processed/topic_formulas_pya0_v2.txt', mode='a', encoding='utf-8') as file:
        for p in formulas:
            print(' '.join(p) + f'{id}')
            file.write(' '.join(p) + '\t' + f'{id}' + '\n')
print(id_count)
print(invalid_ids)
invalid_ids = ['A.326', 'A.329', 'A.339', 'A.354', 'A.365', 'A.373', 'A.375']
