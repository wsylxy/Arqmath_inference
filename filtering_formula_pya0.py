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
formulas_file = json.load(open(f'{data_path}/cleaned_formulas_pya0.json', encoding='utf-8'))
out_path = f'{data_path}'
makedirs(out_path, exist_ok=True)

splits = 1
voc_list = defaultdict(int)
found_space_token = False
defined_tokens = []
original_formulas = []
corrected_formulas = []
filtered_formulas = []
num_answer = 0
tok = LatexTokenizer()

def check_formula(eq, text_length):
    if len(eq)<10:
        return False
    # for op in customize_operator:
    #     if op in eq:
    #         return False
        
    # if (text_length/len(eq))>0.2:
    #     return False
    return True

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

with open("./data_processing/processed/mse-aops-2021-vocab-v3.pkl", "rb") as p:
    vocab = pickle.load(p)
reference_voclist = []
for i, (token, freq) in enumerate(vocab.items()):
    # print(f"{token}: {freq}")
    reference_voclist.append(token)

def write_block(block: list, formula_type: str):
    if len(block) > 0:
        if formula_type == 'question':
            with open(f'{out_path}/processed/formula_blocks_pya0_formula_v2.txt', 'a', encoding='utf-8') as out_file:
                for formula in block:  
                    out_file.write('[Q]' + '\t' + ' '.join(formula) + '\n')
        elif formula_type == 'answer':
            with open(f'{out_path}/processed/formula_blocks_pya0_formula_v2.txt', 'a', encoding='utf-8') as out_file:
                for formula in block:  
                    out_file.write('[R]' + '\t' + ' '.join(formula) + '\n')    
                out_file.write('\n')

# def write_q_dict(block: list, post_id: int):
#     file_exists = os.path.isfile(f'{out_path}/processed/q_dict_pya0.csv')
#     if len(block) > 0:
#         with open(f'{out_path}/processed/q_dict_pya0.csv', 'a', newline='', encoding='utf-8') as f:
#             writer = csv.writer(f)
#             if not file_exists:
#                 writer.writerow(['formula', 'post_id'])
#             for f in block:
#                 writer.writerow([' '.join(f), post_id])

def write_a_dict(block: list, post_id: int):
    file_exists = os.path.isfile(f'{out_path}/processed/a_dict_pya0_formula_v2.csv')
    if len(block) != 0:
        with open(f'{out_path}/processed/a_dict_pya0_formula_v2.csv', 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['formula', 'doc_id'])
            for f in block:
                writer.writerow([' '.join(f), post_id])

# def write_voc(token, original, processed, i):
#     file = open(f'{out_path}/processed/voc_pya0.txt', 'a', encoding='utf-8')
#     file.write(token + '\n')
#     file.write(original + '\n')
#     file.write(' , '.join(map(str, processed)))
#     file.write('\n')
#     file.write('index: ' + str(i) + '\n')
#     file.close()

incorrect_count = 0
num_f = 0
num_qf = 0
num_af = 0
# longest_len = 200
num_clusters = 0
newcommand_id_list = []
zero_formula_q = 0
not_pass_Check = 0
for question in formulas_file:
    question_id = question['post_id']
    question_and_title_formulas = []    #list that contains all formulas in a block
    if 'title_formulas' in question:
        title_fs = []   #list that contains all formulas in the title of a question
        for p in question['title_formulas']:
            num_f += 1
            standard_formula1 = process('[imath]' + p.strip().strip('.').strip('$') + '[/imath]')
            standard_formula = re.findall(r"\$.*?\$", standard_formula1)    #convert str into a list
            text_length = count_textual_content_pya0(standard_formula)  #count the length of text part in a formula
            if check_formula(standard_formula, text_length) and standard_formula not in title_fs:
                bug_formula = False
                num_qf += 1
                for i in range(len(standard_formula)):    #write voc_list
                    if standard_formula[i] in reference_voclist:
                        voc_list[standard_formula[i]] += 1
                    # else:
                    #     incorrect_count += 1
                    #     bug_formula = True
                    #     break
                # if bug_formula:
                #     continue
                title_fs.append(standard_formula)
                # if len(standard_formula) <= longest_len:
                #     title_fs.append(standard_formula)
                # else:
                #     title_fs.append(standard_formula[:longest_len-1])
            else:
                not_pass_Check += 1
        if len(title_fs) != 0:
            question_and_title_formulas.extend(title_fs)
    # print(question['post_id'])         
    question_formula_list = find_formulas(question['question_formulas'])
    question_formula_count = 6  #6 formulas per question at most
    question_fs = []
    for p in question_formula_list:
        num_f += 1
        if question_formula_count == 0:
            break
        standard_formula1 = process('[imath]' + p.strip().strip('.').strip('$') + '[/imath]')
        standard_formula = re.findall(r"\$.*?\$", standard_formula1)
        text_length = count_textual_content_pya0(standard_formula)
        if check_formula(standard_formula, text_length) and standard_formula not in question_fs and standard_formula not in question_and_title_formulas:
            bug_formula = False
            num_qf += 1
            for i in range(len(standard_formula)):    #write voc_list
                if standard_formula[i] in reference_voclist:
                    voc_list[standard_formula[i]] += 1
            #     else:
            #         incorrect_count += 1
            #         bug_formula = True
            #         break
            # if bug_formula:
            #     continue
            question_fs.append(standard_formula)
            question_formula_count -= 1 
            print(standard_formula)
        else:
            not_pass_Check += 1
    # print('questions:', question_fs) 
    if len(question_fs) != 0:
        question_and_title_formulas.extend(question_fs)    
    # print("q_len:", len(question_and_title_formulas))        

    # if len(question_and_title_formulas) == 0:    #check if title and question have no formulas
    #     zero_formula_q += 1
    #     continue
    
    if 'answer_formulas' in question:
        answer_count = 3
        answer_fs = []
        assert len(question['answer_formulas']) == len(question['answer_ids']), "length of answer_list doesn't match length of answer_ids"
        for i, answer in enumerate(question['answer_formulas']):
            single_answer_fs = []
            dict_fs = []
            answer_id = question['answer_ids'][i]
            
            if answer_count == 0:
                break
            answer_formula_list = find_formulas(answer)
            answer_formula_count = 6
            for p in answer_formula_list:
                num_f += 1
                if answer_formula_count == 0:
                    break
                standard_formula1 = process('[imath]' + p.strip().strip('.').strip('$') + '[/imath]')
                standard_formula = re.findall(r"\$.*?\$", standard_formula1)
                text_length = count_textual_content_pya0(standard_formula)
                if check_formula(standard_formula, text_length) and standard_formula not in answer_fs and standard_formula not in question_and_title_formulas:
                    bug_formula = False
                    num_af += 1
                    for i in range(len(standard_formula)):    #write voc_list
                        if standard_formula[i] in reference_voclist:
                            voc_list[standard_formula[i]] += 1
                    #     else:
                    #         incorrect_count += 1
                    #         bug_formula = True
                    #         break
                    # if bug_formula:
                    #     continue
                    single_answer_fs.append(standard_formula)
                    # if len(standard_formula) <= longest_len:
                    #     single_answer_fs.append(standard_formula)
                    # else:
                    #     single_answer_fs.append(standard_formula[:longest_len-1])
                    answer_formula_count -= 1  
                else:
                    not_pass_Check += 1         
            write_a_dict(single_answer_fs, post_id=answer_id)
            num_answer += 1
            answer_fs.extend(single_answer_fs)
            answer_count -= 1
        # print("a_len", len(answer_fs))
    # print(len(question_and_title_formulas)) 
    # print(len(answer_fs))
    if len(question_and_title_formulas) > 0 and len(answer_fs) > 0:
        num_clusters += 1
        write_block(block=question_and_title_formulas, formula_type='question')
        write_block(block=answer_fs, formula_type='answer')

print('num_clusters:', num_clusters)
print('zero formula q is:', zero_formula_q)
print('bug formulas:', incorrect_count)
print('not_pass_Check:', not_pass_Check)
print("# of answers", num_answer)
print("num qf:", num_qf)
print("num af", num_af)
print("num_f:", num_f)


# with open(f'{out_path}/processed/voc_list_pya0_formula.txt', 'w', encoding='utf-8') as file:
#     sorted_dict = dict(sorted(voc_list.items(), key=lambda item: item[1], reverse=True))
#     for voc in sorted_dict:
#         file.write(voc + '\t' + str(voc_list[voc]) + '\n')
