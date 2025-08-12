# one sentence = one formula
# one document = one formula chain?\
from preprocess import preprocess_for_transformer_old as process
from preprocess import count_textual_content_pya0
from collections import defaultdict
from custom_tokenize import LatexTokenizer
from custom_tokenize import *
from math_latex import *
from os import makedirs
import pickle
import os
import json
import re
import csv

# from tex_lex
t_REL_CLASS = r'=|:=|\\[dD]oteq|\\dot=|\\approxeq|\\backsimeq|\\circeq|\\cong|\\backsim|\\curlyeqprec|\\curlyeqsucc|\\eqslantgtr|\\eqslantless|\\equiv|\\gnsim|\\triangleq|\\eqsim|\\thicksim|\\sim|\\simeq|\\nsim|\\neq|\\not(=|\\equiv)|\\frown|\\between|\\eqcirc|\\smallfrown|\\smallsmile|\\approx|\\asymp|\\ge|\\geq|\\geqq|\\geqslant|\\gg|\\gnapprox|\\gt|>|\\gtrapprox|\\gtrdot|\\gtreqless|\\gtreqqless|\\gtrless|\\gtrsim|\\le^[f]|\\leq|\\leqq|\\leqslant|\\lessapprox|\\lessdot|\\lesssim|\\ll|\\lnapprox|\\lneq|\\lneqq|\\lt|<|\\lvertneqq|\\ncong|\\ne|\\ngeq|\\ngeqq|\\ngeqslant|\\nleq|\\nleqq|\\nleqslant|\\nless|\\nprec|\\npreceq|\\nsucc|\\nsucceq|\\prec|\\preceq|\\succ|\\succapprox|\\succcurlyeq|\\thickapprox|\\trianglelefteq|\\trianglerighteq|\\succeq|\\succnapprox|\\succneqq|\\succnsim|\\succsim|\\unlhd|\\unrhd|\\gneq|\\gneqq|\\gvertneqq|\\ggg|\\gggtr|\\ngtr|\\precapprox|\\preccurlyeq|\\precnapprox|\\precneqq|\\precnsim|\\precsim|\\Cap|\\cap|\\Cup|\\cup|\\curlyvee|\\dashv|\\curlywedge|\\land|\\lor|\\sqcap|\\sqcup|\\vee|\\veebar|\\wedge|\\Join|\\bowtie|\\Subset|\\Supset|\\nsubseteq|\\nsupseteq|\\supseteq|\\subset|\\sqsubset|\\sqsubseteq|\\sqsupset|\\sqsupseteq|\\subseteq|\\subseteqq|\\subsetneq|\\subsetneqq|\\supset|\\supseteq|\\supseteqq|\\supsetneq|\\supsetneqq|\\varsubsetneq|\\varsubsetneqq|\\varsupsetneq|\\varsupsetneqq|\\in|\\ni|\\not\\in|\\owns|\\nparallel|\\parallel|\\propto'

rel_pattern = re.compile(t_REL_CLASS)


data_path = './data_processing'
posts_file = json.load(open(f'{data_path}/cleaned_posts_pya01.json', encoding='utf-8'))
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
    if len(post) >= 10:
        return True
    else:
        return False

def write_block(block: list, post_type: str):
    if len(block) > 0:
        if post_type == 'question':
            with open(f'{out_path}/processed/post_blocks_pya0_v2.txt', 'a', encoding='utf-8') as out_file:
                for post in block:  
                    out_file.write('[Q]' + '\t' + post + '\n')
        elif post_type == 'answer':
            with open(f'{out_path}/processed/post_blocks_pya0_v2.txt', 'a', encoding='utf-8') as out_file:
                for post in block:  
                    out_file.write('[R]' + '\t' + post + '\n')    
                out_file.write('\n')

def write_a_dict(post: str, post_id: int):
    file_exists = os.path.isfile(f'{out_path}/processed/a_dict_post_pya0_v2.csv')
    # if len(post) > longest_len:
    #     post = post[:longest_len-1]
    with open(f'{out_path}/processed/a_dict_post_pya0_v2.csv', 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['doc', 'post_id'])
        writer.writerow([post, post_id])

num_qf = 0
num_af = 0
longest_len = 512
num_clusters = 0

for question in posts_file:
    question_ps = []
    question_id = question['post_id']
    standard_post = process(question['question_posts'])
    if check_post(standard_post):
        question_ps.append(standard_post)
    print(standard_post)      
    if len(question_ps) == 0:
        continue
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
            if check_post(standard_post):
                answer_ps.append(standard_post)
            answer_count -= 1
            # write_a_dict(post=standard_post, post_id=answer_id)
    if len(question_ps) > 0 and len(answer_ps) > 0:
        num_clusters += 1
        write_block(block=question_ps, post_type='question')
        write_block(block=answer_ps, post_type='answer')
        
print('num_clusters:', num_clusters)
