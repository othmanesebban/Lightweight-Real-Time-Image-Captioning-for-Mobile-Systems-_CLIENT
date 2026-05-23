import json
with open("Dataset/captions_val2014.json", 'r') as f:
    data = json.load(f)

images = data['images']
captions = data['annotations']

prefix = "COCO_train2014_"

for cap in captions:
    image_id = str(cap['image_id'])
    len_id = len(image_id)
    zeros = '0'*(12-len_id)
    image_id = prefix+zeros+image_id
    cap['image_id'] = image_id
    cap['caption'] = cap['caption'].replace('\n', '').replace(',', ' ,').replace('.', '').replace('"', '" ')

captions = sorted(captions, key=lambda k: k['image_id'])

with open("captions_generations.txt",'w') as f:
    for i, cap in enumerate(captions):
        f.write(cap['image_id']+'#'+str(i%5)+'\t'+cap['caption']+'\n')

