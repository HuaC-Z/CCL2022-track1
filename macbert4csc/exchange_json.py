with open('./train_aug.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    output = []
    for i, line in enumerate(lines):
        l = {}
        arr = line.strip('\n').split('\t')
        a = arr[0].strip()
        b = arr[1].strip()
        l["original_text"] = a
        l["correct_text"] = b
        l["id"] = "-"

        tmp = []
        for i, char in enumerate(arr[0]):
            if char != arr[1][i]:
                if char == '"' or arr[1][i] == '"':
                    continue
                # print(i, arr[1][i])
                tmp.append(i+1)
        l["wrong_ids"] = tmp
        output.append(l)

import json
json_str = json.dumps(output, ensure_ascii=False, indent=4)
with open('train_aug.json', 'w', encoding='utf-8') as json_file:
    json_file.write(json_str)
