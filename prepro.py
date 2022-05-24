"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

preprocess NLVR annotations into LMDB
"""
import argparse
import json
import pickle
import os
from os.path import exists

from nltk.tokenize import word_tokenize, sent_tokenize
from cytoolz import curry
from tqdm import tqdm
from pytorch_pretrained_bert import BertTokenizer

from data.data import open_lmdb

LABEL2ANS = {
    "Single Family": 0,
    "Condo": 1,
    "Multiple Occupancy": 2,
    "Vacant Land": 3,
    "Townhouse": 4,
    "Recreational": 5,
    "Country House": 6,
    "Villa": 0,
    "Ejerlejlighed": 1,
    "Kollektiv": 2,
    "Helårsgrund": 3,
    "Fritidsgrund": 3,
    "Rækkehus": 4,
    "Fritidsbolig": 5,
    "Landejendom": 6
}

@curry
def bert_tokenize(tokenizer, text):
    ids = []
    for word in text.strip().split():
        ws = tokenizer.tokenize(word)
        if not ws:
            # some special char
            continue
        ids.extend(tokenizer.convert_tokens_to_ids(ws))
    return ids


def process_nlvr2(jsonl, db, tokenizer, missing=None):
    id2len = {}
    txt2img = {}  # not sure if useful
    for line in tqdm(jsonl, desc='processing NLVR2'):
        example = json.loads(line)
        id_ = example['identifier']
        img_id = '-'.join(id_.split('-')[:-1])
        img_fname = (f'nlvr2_{img_id}-img0.npz', f'nlvr2_{img_id}-img1.npz')
        if missing and (img_fname[0] in missing or img_fname[1] in missing):
            continue
        input_ids = tokenizer(example['sentence'])
        if 'label' in example:
            target = 1 if example['label'] == 'True' else 0
        else:
            target = None
        txt2img[id_] = img_fname
        id2len[id_] = len(input_ids)
        example['input_ids'] = input_ids
        example['img_fname'] = img_fname
        example['target'] = target
        db[id_] = example
    return id2len, txt2img


def process_sbu(captions, found_ids, db, tokenizer):
    id2len = {}
    txt2img = {} 
    img2txt = {}
    for idx, caption in tqdm(enumerate(captions), desc='processing SBU'):
        if idx in found_ids:
            idx = str(idx)
            img_fname = f"SBU_{idx}.npz"
            example = {}
            example['id'] = idx
            example['image_id'] = idx
            example['img_fname'] = img_fname
            example['sent'] = caption[1]
            tokens = word_tokenize(caption[1])
            example['toked_caption'] = tokens
            example['dataset'] = "SBU"
            example['split'] = "train"
            input_ids = tokenizer(' '.join(tokens))
            example['input_ids'] = input_ids
            example['dataset_image_id'] = idx
            txt2img[idx] = img_fname
            img2txt[img_fname] = idx
            id2len[idx] = len(input_ids)
            db[idx] = example
    
    return id2len, txt2img, img2txt

def process_wit(captions, found_ids, db, tokenizer):
    id2len = {}
    txt2img = {} 
    img2txt = {}
    for idx, caption in tqdm(enumerate(captions), desc='processing WIT'):
        if idx in found_ids:
            idx = str(idx)
            img_fname = f"WIT_{idx}.npz"
            example = {}
            example['id'] = idx
            example['image_id'] = idx
            example['img_fname'] = img_fname
            example['sent'] = caption["caption_reference_description"]
            tokens = word_tokenize(caption["caption_reference_description"])
            example['toked_caption'] = tokens
            example['dataset'] = "SBU"
            example['split'] = "train"
            input_ids = tokenizer(' '.join(tokens))
            example['input_ids'] = input_ids
            example['dataset_image_id'] = idx
            txt2img[idx] = img_fname
            img2txt[img_fname] = idx
            id2len[idx] = len(input_ids)
            db[idx] = example
    
    return id2len, txt2img, img2txt

def process_dagw(captions, db, tokenizer):
    id2len = {}
    for idx, caption in tqdm(enumerate(captions), desc='processing DAGW'):
        idx = str(idx)
        example = {}
        example['id'] = idx
        example['sent'] = caption
        tokens = word_tokenize(caption)
        example['toked_caption'] = tokens
        example['dataset'] = "DAGW"
        example['split'] = "train"
        input_ids = tokenizer(' '.join(tokens))
        example['input_ids'] = input_ids
        id2len[idx] = len(input_ids)
        db[idx] = example
    
    return [id2len]

def process_amhomes(captions, found_ids, db, tokenizer, split):
    id2len = {}
    txt2img = {} 
    img2txt = {}
    missing = 0
    for idx, home in tqdm(captions.items(), desc=f"Processing Amhomes {split}"):
        if idx in found_ids:
            example = {}
            img_fname = f"{idx}.npz"
            example['id'] = idx
            example['image_id'] = idx
            example['img_fname'] = img_fname
            sent = sent_tokenize(home["description"], language='english')
            if len(sent) == 0:
                sent = ["Beautiful house!"]
                missing += 1
            example['sent'] = sent[0]
            tokens = word_tokenize(sent[0])
            example['toked_caption'] = tokens
            example['dataset'] = "SmallH"
            example['split'] = split
            input_ids = tokenizer(' '.join(tokens))
            example['input_ids'] = input_ids
            example['dataset_image_id'] = idx
            example['target'] = LABEL2ANS[home["homeType"]]
            txt2img[idx] = img_fname
            img2txt[img_fname] = idx
            id2len[idx] = len(input_ids)
            db[idx] = example
    
    print(f"For {split} there were a total of {missing} missing descriptions...")
    
    return id2len, txt2img, img2txt

def process_danhomes(captions, found_ids, db, tokenizer, split):
    id2len = {}
    txt2img = {} 
    img2txt = {}
    for idx, home in tqdm(captions.items(), desc=f"Processing DanHomes {split}"):
        if idx in found_ids:
            example = {}
            img_fname = f"danhomes_{idx}.npz"
            example['id'] = idx
            example['image_id'] = idx
            example['img_fname'] = img_fname
            sent = sent_tokenize(home["DescriptionPlain"], language='danish')
            if len(sent) == 0:
                continue
            example['sent'] = sent[0]
            tokens = word_tokenize(sent[0])
            example['toked_caption'] = tokens
            example['dataset'] = "DanHomes"
            example['split'] = split
            input_ids = tokenizer(' '.join(tokens))
            example['input_ids'] = input_ids
            example['dataset_image_id'] = idx
            example['target'] = LABEL2ANS[home["Type"]]
            txt2img[idx] = img_fname
            img2txt[img_fname] = idx
            id2len[idx] = len(input_ids)
            db[idx] = example
    
    return id2len, txt2img, img2txt

def process_referring_expressions(refs, instances, iid_to_ann_ids,
                                  db, tokenizer, split):
    """
    Inputs:
    - refs: [ref_id, ann_id, image_id, split, sent_ids, sentences]
    - instances: {images, annotations, categories}
    - iid_to_ann_ids: image_id -> ann_ids ordered by extracted butd features
    Return:
    - id2len : sent_id -> tokenized question length
    - images : [{id, file_name, ann_ids, height, width} ]
    - annotations: [{id, area, bbox, image_id, category_id, iscrowd}]
    - categories : [{id, name, supercategory}]
    """
    # images within split
    image_set = set([ref['image_id'] for ref in refs if ref['split'] == split])
    images = []
    for img in instances['images']:
        if img['id'] in image_set:
            images.append({
                'id': img['id'], 'file_name': img['file_name'],
                'ann_ids': iid_to_ann_ids[str(img['id'])],
                'height': img['height'], 'width': img['width']})
    # Images = {img['id']: img for img in images}
    # anns within split
    annotations = []
    for ann in instances['annotations']:
        if ann['image_id'] in image_set:
            annotations.append({
                'id': ann['id'], 'area': ann['area'], 'bbox': ann['bbox'],
                'image_id': ann['image_id'],
                'category_id': ann['category_id'],
                'iscrowd': ann['iscrowd']
            })
    Anns = {ann['id']: ann for ann in annotations}
    # category info
    categories = instances['categories']
    # refs within split
    refs = [ref for ref in refs if ref['split'] == split]
    print(f"Processing {len(refs)} annotations...")
    id2len = {}
    for ref in tqdm(refs, desc='processing referring expressions'):
        ref_id = ref['ref_id']
        ann_id = ref['ann_id']
        image_id = ref['image_id']
        img_fname = f"visual_grounding_coco_gt_{int(image_id):012}.npz"
        for sent in ref['sentences']:
            sent_id = sent['sent_id']
            input_ids = tokenizer(sent['sent'])
            id2len[str(sent_id)] = len(input_ids)
            db[str(sent_id)] = {
                'sent_id': sent_id, 'sent': sent['sent'],
                'ref_id': ref_id, 'ann_id': ann_id,
                'image_id': image_id, 'bbox': Anns[ann_id]['bbox'],
                'input_ids': input_ids,
                'img_fname': img_fname
                }
    return id2len, images, annotations, categories, refs


def main(opts):
    if not exists(opts.output):
        os.makedirs(opts.output)
    else:
        raise ValueError('Found existing DB. Please explicitly remove '
                         'for re-processing')
    meta = vars(opts)
    meta['tokenizer'] = opts.toker
    toker = BertTokenizer.from_pretrained(
        opts.toker, do_lower_case='uncased' in opts.toker)
    tokenizer = bert_tokenize(toker)
    meta['UNK'] = toker.convert_tokens_to_ids(['[UNK]'])[0]
    meta['CLS'] = toker.convert_tokens_to_ids(['[CLS]'])[0]
    meta['SEP'] = toker.convert_tokens_to_ids(['[SEP]'])[0]
    meta['MASK'] = toker.convert_tokens_to_ids(['[MASK]'])[0]
    meta['v_range'] = (toker.convert_tokens_to_ids('!')[0],
                       len(toker.vocab))
    with open(f'{opts.output}/meta.json', 'w') as f:
        json.dump(vars(opts), f, indent=4)

    open_db = curry(open_lmdb, opts.output, readonly=False)
    output_field_name = ['id2len', 'txt2img']
    with open_db() as db:
        if opts.task == 'nlvr':
            with open(opts.annotations[0]) as ann:
                if opts.missing_imgs is not None:
                    missing_imgs = set(json.load(open(opts.missing_imgs)))
                else:
                    missing_imgs = None
                jsons = process_nlvr2(
                    ann, db, tokenizer, missing_imgs)
        elif opts.task == "sbu":
            captions = json.load(open(opts.annotations[0]))
            found_ids = open(opts.missing_imgs).read().split("\n")
            found_ids = list(map(lambda x: int(x), found_ids[:-1]))
            jsons = process_sbu(captions, found_ids, db, tokenizer)
            output_field_name = ['id2len', 'txt2img', 'img2txt']
        elif opts.task == "wit":
            captions = json.load(open(opts.annotations[0]))
            found_ids = open(opts.missing_imgs).read().split("\n")
            found_ids = list(map(lambda x: int(x), found_ids[:-1]))
            jsons = process_wit(captions, found_ids, db, tokenizer)
            output_field_name = ['id2len', 'txt2img', 'img2txt']
        elif opts.task == "dagw":
            captions = json.load(open(opts.annotations[0]))
            jsons = process_dagw(captions, db, tokenizer)
            output_field_name = ['id2len']
        elif opts.task == "amhomes":
            captions = json.load(open(opts.annotations[0]))
            found_ids = open(opts.missing_imgs).read().split("\n")
            # found_ids = list(map(lambda x: int(x), found_ids[:-1]))
            jsons = process_amhomes(captions, found_ids, db, tokenizer, opts.split)
            output_field_name = ['id2len', 'txt2img', 'img2txt']
        elif opts.task == "danhomes":
            captions = json.load(open(opts.annotations[0]))
            found_ids = open(opts.missing_imgs).read().split("\n")
            # found_ids = list(map(lambda x: int(x), found_ids[:-1]))
            jsons = process_danhomes(captions, found_ids, db, tokenizer, opts.split)
            output_field_name = ['id2len', 'txt2img', 'img2txt']
        elif opts.task == 're':
            data = pickle.load(open(opts.annotations[0], 'rb'))
            instances = json.load(open(opts.annotations[1], 'r'))
            iid_to_ann_ids = json.load(
                open(opts.annotations[2], 'r'))['iid_to_ann_ids']
            # dirs/refcoco_testA_bert-base-cased.db -> testA
            img_split = opts.output.split('/')[-1].split('.')[0].split('_')[1]
            jsons = process_referring_expressions(
                data, instances, iid_to_ann_ids,
                db, tokenizer, img_split)
            output_field_name = [
                'id2len', 'images', 'annotations',
                'categories', 'refs']

    for dump, name in zip(jsons, output_field_name):
        with open(f'{opts.output}/{name}.json', 'w') as f:
            json.dump(dump, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations', required=True, nargs='+',
                        help='annotation JSON')
    parser.add_argument('--missing_imgs',
                        help='some training image features are corrupted')
    parser.add_argument('--output', required=True,
                        help='output dir of DB')
    parser.add_argument('--task', required=True, default='nlvr',
                        choices=['nlvr', 're', 'sbu', 'wit', 'dagw', 'amhomes', 'danhomes'])
    parser.add_argument('--split', required=True, default='train',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--toker', default='bert-base-cased',
                        help='which BERT tokenizer to used')
    args = parser.parse_args()
    if args.task == 'nlvr':
        assert len(args.annotations) == 1
    elif args.task == 're':
        assert len(args.annotations) == 3
    main(args)
