# one sentence = one formula
# one document = one formula chain?\
from preprocess import preprocess_for_transformer_old as process
from preprocess import count_textual_content_pya0
from collections import defaultdict
from custom_tokenize import LatexTokenizer
from custom_tokenize import *
from math_latex import *
from os import makedirs
import bs4 as bs
import pickle
import os
import json
import re
import csv

# from tex_lex
t_REL_CLASS = r'=|:=|\\[dD]oteq|\\dot=|\\approxeq|\\backsimeq|\\circeq|\\cong|\\backsim|\\curlyeqprec|\\curlyeqsucc|\\eqslantgtr|\\eqslantless|\\equiv|\\gnsim|\\triangleq|\\eqsim|\\thicksim|\\sim|\\simeq|\\nsim|\\neq|\\not(=|\\equiv)|\\frown|\\between|\\eqcirc|\\smallfrown|\\smallsmile|\\approx|\\asymp|\\ge|\\geq|\\geqq|\\geqslant|\\gg|\\gnapprox|\\gt|>|\\gtrapprox|\\gtrdot|\\gtreqless|\\gtreqqless|\\gtrless|\\gtrsim|\\le^[f]|\\leq|\\leqq|\\leqslant|\\lessapprox|\\lessdot|\\lesssim|\\ll|\\lnapprox|\\lneq|\\lneqq|\\lt|<|\\lvertneqq|\\ncong|\\ne|\\ngeq|\\ngeqq|\\ngeqslant|\\nleq|\\nleqq|\\nleqslant|\\nless|\\nprec|\\npreceq|\\nsucc|\\nsucceq|\\prec|\\preceq|\\succ|\\succapprox|\\succcurlyeq|\\thickapprox|\\trianglelefteq|\\trianglerighteq|\\succeq|\\succnapprox|\\succneqq|\\succnsim|\\succsim|\\unlhd|\\unrhd|\\gneq|\\gneqq|\\gvertneqq|\\ggg|\\gggtr|\\ngtr|\\precapprox|\\preccurlyeq|\\precnapprox|\\precneqq|\\precnsim|\\precsim|\\Cap|\\cap|\\Cup|\\cup|\\curlyvee|\\dashv|\\curlywedge|\\land|\\lor|\\sqcap|\\sqcup|\\vee|\\veebar|\\wedge|\\Join|\\bowtie|\\Subset|\\Supset|\\nsubseteq|\\nsupseteq|\\supseteq|\\subset|\\sqsubset|\\sqsubseteq|\\sqsupset|\\sqsupseteq|\\subseteq|\\subseteqq|\\subsetneq|\\subsetneqq|\\supset|\\supseteq|\\supseteqq|\\supsetneq|\\supsetneqq|\\varsubsetneq|\\varsubsetneqq|\\varsupsetneq|\\varsupsetneqq|\\in|\\ni|\\not\\in|\\owns|\\nparallel|\\parallel|\\propto'

rel_pattern = re.compile(t_REL_CLASS)


data_path = './data_processing'
posts_file = json.load(open(f'{data_path}/cleaned_posts.json', encoding='utf-8'))
out_path = f'{data_path}'
makedirs(out_path, exist_ok=True)

splits = 1
voc_list = defaultdict(int)
found_space_token = False
defined_tokens = []
original_formulas = []
corrected_formulas = []
filtered_formulas = []
tok = LatexTokenizer()

def check_post(post):
    matches = re.findall(r"(\$.*?\$)", post)
    if len(matches) >= 10:
        return True
    else:
        return False

def find_formulas(lst):
    result = []
    left, right = 0, len(lst)-1
    while left < right:
        if left == right:
            result.append(lst[left])
        else:
            result.append(lst[left])
            result.append(lst[right])
        left += 1
        right -= 1
    return result

def get_formulas(body):
    soup = bs.BeautifulSoup(body, "lxml")
    formulas = []
    for math in soup.find_all('span', {'class': "math-container"}):
        formulas.append(math.text)
    return formulas

with open("./data_processing/processed/mse-aops-2021-vocab-v3.pkl", "rb") as p:
    vocab = pickle.load(p)
reference_voclist = []
for i, (token, freq) in enumerate(vocab.items()):
    # print(f"{token}: {freq}")
    reference_voclist.append(token)

def write_block(block: list, post_type: str):
    if len(block) > 0:
        if post_type == 'question':
            with open(f'{out_path}/processed/fusion_blocks_post_pya0.txt', 'a', encoding='utf-8') as out_file:
                for row in block:  
                    out_file.write('[Q]' + '\t' + row + '\n')
        elif post_type == 'answer':
            with open(f'{out_path}/processed/fusion_blocks_post_pya0.txt', 'a', encoding='utf-8') as out_file:
                for row in block:  
                    out_file.write('[R]' + '\t' + row + '\n')    
                out_file.write('\n')

# def write_q_dict(block: list, post_id: int):
#     file_exists = os.path.isfile(f'{out_path}/processed/q_dict_post_pya0.csv')
#     if len(block) > 0:
#         with open(f'{out_path}/processed/q_dict_pya0.csv', 'a', newline='', encoding='utf-8') as f:
#             writer = csv.writer(f)
#             if not file_exists:
#                 writer.writerow(['formula', 'post_id'])
#             for f in block:
#                 writer.writerow([' '.join(f), post_id])

def write_a_dict(post: str, post_id: int):
    file_exists = os.path.isfile(f'{out_path}/processed/a_dict_pya0_fusion.csv')
    # if len(post) > longest_len:
    #     post = post[:longest_len-1]
    with open(f'{out_path}/processed/a_dict_pya0_fusion.csv', 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['doc', 'post_id'])
        writer.writerow([post, post_id])

def write_voc(token, original, processed, i):
    file = open(f'{out_path}/processed/voc_post_pya0.txt', 'a', encoding='utf-8')
    file.write(token + '\n')
    file.write(original + '\n')
    file.write(' , '.join(map(str, processed)))
    file.write('\n')
    file.write('index: ' + str(i) + '\n')
    file.close()

incorrect_count = 0
num_qf = 0
num_af = 0
longest_len = 512
num_clusters = 0
zero_formula_q = 0

for question in posts_file:
    question_ps = []
    question_id = question['post_id']
    question_fs = []
    question_f = get_formulas(question['question_posts'])
    standard_post = process(question['question_posts']) #processed post is a string
    if len(question_f) == 0:    #if the question has no formulas
        continue
    for f in question_f:
        standard_formula1 = process('[imath]' + p.strip().strip('.').strip('$') + '[/imath]')
        standard_formula = re.findall(r"\$.*?\$", standard_formula1)
        question_fs.append(standard_formula)    #each formula is a list
    longest_formula = max(question_fs, key=len)
    if len(longest_len) < 5:
        continue
    question_ps.append(f"{standard_post}\t{' '.join(longest_formula)}")
    print(standard_post)      

    #TODO: check is there formulas in the post, count the number of zero formula posts
    
    if 'answer_posts' in question:
        answer_count = 3
        answer_ps = []
        assert len(question['answer_posts']) == len(question['answer_ids']), "length of answer_list doesn't match length of answer_ids"
        for i, answer in enumerate(question['answer_posts']):
            single_answer_fs = []
            dict_fs = []
            answer_id = question['answer_ids'][i]
            if answer_count == 0:
                break
            standard_post = process(answer)
            answer_fs = []
            answer_f = get_formulas(answer)
            if len(answer_f) == 0:
                continue
            for f in answer_f:
                standard_formula1 = process('[imath]' + p.strip().strip('.').strip('$') + '[/imath]')
                standard_formula = re.findall(r"\$.*?\$", standard_formula1)
                answer_fs.append(standard_formula)    #each formula is a list
            longest_formula = max(answer_fs, key=len)
            if len(longest_len) < 5:
                continue
            answer_ps.append(f"{standard_post}\t{' '.join(longest_formula)}")
            answer_count -= 1
            write_a_dict(post=standard_post, post_id=answer_id)
    if question_ps and len(answer_ps) > 0:
        num_clusters += 1
        write_block(block=question_ps, post_type='question')
        write_block(block=answer_ps, post_type='answer')
        

print('num_clusters:', num_clusters)
# print(newcommand_id_list)
print('zero formula q is:', zero_formula_q)
print('bug formulas:', incorrect_count)
# print("num_qf:", num_qf)
# print("num_af:", num_af)
# print(f'Total len of formulas: {len(filtered_formulas)}')

# print(voc_list)

with open(f'{out_path}/processed/voc_list_pya0.txt', 'w', encoding='utf-8') as file:
    sorted_dict = dict(sorted(voc_list.items(), key=lambda item: item[1], reverse=True))
    for voc in sorted_dict:
        file.write(voc + '\t' + str(voc_list[voc]) + '\n')

# written_lines = 0
# with open(f'{out_path}/processed/rel_split_formulas.txt', 'w', encoding='utf-8') as out_file:
#     for formula in filtered_formulas:  
#         out_file.write(' '.join(formula) + '\n')
#         written_lines += 1
# print(f'Total number of non-empty files: {written_lines}')

# with open(f'{out_path}/processed/original_formulas.txt', 'w', encoding='utf-8') as ori_file:
#     for formula in original_formulas:  
#         ori_file.write(' '.join(formula) + '\n')

# with open(f'{out_path}/processed_small/corrected_formulas.txt', 'w', encoding='utf-8') as ori_file:
#     for formula in corrected_formulas:  
#         ori_file.write(' '.join(formula) + '\n')