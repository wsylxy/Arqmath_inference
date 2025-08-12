import random
from itertools import combinations, permutations

filepath = 'data_processing/processed/fusion_blocks_pya0_formula.txt'
outfile = '/mnt/d/data_process/train_set_pya0_formula.txt'
clusters = [] #list of all clusters
cluster = [] #list of each cluster
random.seed(42)
target_len = 80
longest_len = 200

block_file = open(file=filepath, mode='r', encoding='utf-8')
for line in block_file:
    if line.strip():
        line = line.strip().split('\t')
        # if len(line[1].split(' ')) <= 200:
        #     cluster.append(line)
        # else:
        #     formula = line[1].split(' ')
        #     cluster.append([line[0], ' '.join(formula[:longest_len-1])])    #each cluster is a list of sublist, sublist form is [[Q]/[R], formula]
        cluster.append(line)
    else:
        clusters.append(cluster)
        cluster = []
block_file.close()
num_neg = 1
print("num of clusters:", len(clusters))
    
def generate_pairs(cluster:list, target_len:int):
    question_formulas = []
    answer_formulas = []
    for line in cluster:
        if line[0] == '[Q]':
            question_formulas.append(line[1] + "\t" + line[2])
        elif line[0] == '[R]':
            answer_formulas.append(line[1] + "\t" + line[2])
    output = []
    # if len(question_formulas) >= 2:
    #     question_pairs = list(combinations(iterable=question_formulas, r=2))
    #     output.extend(question_pairs)
    for q in question_formulas:
        for a in answer_formulas:
            output.append((q, a))
    if len(output) > target_len:
        output = output[:target_len-1]
    return output

def write_trainset(clusters, num_neg, outfile, target_len):
    # print(1)
    total_output = []
    ids_all = [j for j in range(len(clusters))]
    for i in range(len(clusters)):
        pos = generate_pairs(cluster=clusters[i], target_len=target_len)
        # pos= generate_small_data(cluster=clusters[i])  #small dataset
        while True:
            ids = random.sample(ids_all, k=num_neg*len(pos))
            if i not in ids:
                break
        neg_samples = []
        # print(2)
        for idx in ids:
            # neg_samples.append(clusters[idx][0][1])
            # print(idx)
            while True:
                sample_tuple = random.sample(population=clusters[idx], k=1) #return of random.sample is a tuple, even thougth only sample 1 item
                if sample_tuple[0][0] == "[R]":
                    neg_samples.append(sample_tuple[0][1] + '\t' + sample_tuple[0][2])
                    break

        # print('neg length', len(neg_samples))
        out = []
        id_neg = 0
        assert len(neg_samples) == num_neg*len(pos)
        for j, p in enumerate(pos):
            line = [p[0], p[1]]
            id_neg = j*num_neg
            line.extend(neg_samples[id_neg:id_neg+num_neg])
            # for id_neg in range(len(neg_samples), num_neg):
            #     line.extend(neg_samples[id_neg:id_neg+num_neg])
            # print('neg length', len(negs))
            out.append(line)
        print(i)
        # print('line length:', len(line))
        for line in out:
            total_output.append(line)
    random.shuffle(total_output)
    # total_output = random.sample(population=total_output, k=10000000)

    # print("total:", len(total_output))
    write_file = open(file=outfile, mode='a', encoding='utf-8')
    for line in total_output:
        write_file.write('\t'.join(line))
        write_file.write('\n')
    write_file.close()

write_trainset(clusters, num_neg, outfile=outfile, target_len=target_len)