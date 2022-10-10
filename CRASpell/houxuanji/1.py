
a = './dict_data/vocab.txt'
with open(a, 'r', encoding='utf-8') as f:
    d = f.readlines()[:10]
    print(d)