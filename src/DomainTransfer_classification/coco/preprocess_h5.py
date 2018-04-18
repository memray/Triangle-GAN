"""
Preprocess a raw json dataset into hdf5/json files for use in data_loader.lua

Input: json file that has the form
[{ file_path: 'path/img.jpg', captions: ['a caption', ...] }, ...]
example element in this list would look like
{'captions': [u'A man with a red helmet on a small moped on a dirt road. ', u'Man riding a motor bike on a dirt road on the countryside.', u'A man riding on the back of a motorcycle.', u'A dirt path with a young person on a motor bike rests to the foreground of a verdant area with a bridge and a background of cloud-wreathed mountains. ', u'A man in a red shirt and a red hat is on a motorcycle on a hill side.'], 'file_path': u'val2014/COCO_val2014_000000391895.jpg', 'id': 391895}

This script reads this json, does some basic preprocessing on the captions
(e.g. lowercase, etc.), creates a special UNK token, and encodes everything to arrays

Output: a json file and an hdf5 file
The hdf5 file contains several fields:
/images is (N,3,256,256) uint8 array of raw image data in RGB format
/labels is (M,max_length) uint32 array of encoded labels, zero padded
/caption_start_ix and /caption_end_ix are (N,) uint32 arrays of pointers to the 
  first and last indices (in range 1..M) of labels for each image
/caption_length stores the length of the sequence for each of the M sequences

The json file has a dict that contains:
- an 'ix_to_word' field storing the vocab in form {ix:'word'}, where ix is 1-indexed
- an 'images' field that is a list holding auxiliary information for each image, 
  such as in particular the 'split' it was assigned to.
"""

import os
import json
import argparse
from random import shuffle, seed
import string
# non-standard dependencies:
import h5py
import pickle
import numpy as np
from scipy.misc import imread, imresize

PAD_WORD = '<pad>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'

def build_vocab(imgs, params):
    count_thr = params['word_count_threshold']

    # count up the number of words
    counts = {}
    for img in imgs:
        if img['split'] != 'train':
            continue
        for s in img['sentences']:
            for w in s['tokens']:
                counts[w] = counts.get(w, 0) + 1
    cw = sorted(counts.items(), key=lambda x:x[1], reverse=True)
    print('top words and their counts:')
    print('\n'.join(map(str, cw[:20])))

    # print some stats
    total_words = sum(counts.values())
    print('total words:', total_words)
    bad_words = [w for w, n in counts.items() if n <= count_thr]

    vocab = [PAD_WORD, UNK_WORD, BOS_WORD, EOS_WORD] + [w for w, n in counts.items() if n > count_thr]
    bad_count = sum(counts[w] for w in bad_words)
    print('number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words) * 100.0 / len(counts)))
    print('number of words in vocab would be %d' % (len(vocab),))
    print('number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count * 100.0 / total_words))

    # lets look at the distribution of lengths as well
    sent_lengths = {}
    for img in imgs:
        for s in img['sentences']:
            nw = len(s['tokens'])
            sent_lengths[nw] = sent_lengths.get(nw, 0) + 1
    max_len = max(sent_lengths.keys())
    print('max length sentence in raw data: ', max_len)
    print('sentence length distribution (count, number of words):')
    sum_len = sum(sent_lengths.values())
    for i in range(max_len + 1):
        print('%2d: %10d   %f%%' % (i, sent_lengths.get(i, 0), sent_lengths.get(i, 0) * 100.0 / sum_len))

    # lets now produce the final annotations
    if bad_count > 0:
        # additional special UNK token we will use below to map infrequent words to
        print('inserting the special UNK token')

    for img in imgs:
        for s in img['sentences']:
            s['final_tokens'] = [BOS_WORD]
            caption = [w if counts.get(w, 0) > count_thr else UNK_WORD for w in s['tokens']]
            s['final_tokens'].extend(caption)
            s['final_tokens'].append(EOS_WORD)

    return vocab

def encode_captions(imgs, params, wtoi):
    """
    encode all captions into one large array, which will be 1-indexed.
    also produces caption_start_ix and caption_end_ix which store 1-indexed
    and inclusive (Lua-style) pointers to the first and last caption for
    each image in the dataset.
    """

    max_length = params['max_length']
    
    N = len(imgs)
    M = sum(len(img['sentences']) for img in imgs)  # total number of captions

    caption_arrays = []
    caption_start_ix = np.zeros(N, dtype='uint32')  # note: these will be one-indexed
    caption_end_ix = np.zeros(N, dtype='uint32')
    caption_length = np.zeros(M, dtype='uint32')
    caption_counter = 0
    counter = 1
    for i, img in enumerate(imgs):
        n = len(img['sentences'])
        assert n > 0, 'error: some image has no captions'

        Li = np.zeros((n, max_length), dtype='uint32')
        for j, s in enumerate(img['sentences']):
            caption_length[caption_counter] = min(max_length, len(s))  # record the length of this sequence
            caption_counter += 1
            for k, w in enumerate(s['final_tokens']):
                if k < max_length:
                    Li[j, k] = wtoi[w]

        # note: word indices are 1-indexed, and captions are padded with zeros
        caption_arrays.append(Li)
        caption_start_ix[i] = counter
        caption_end_ix[i] = counter + n - 1

        counter += n

    L = np.concatenate(caption_arrays, axis=0)  # put all the captions together
    assert L.shape[0] == M, 'lengths don\'t match? that\'s weird'
    assert np.all(caption_length > 0), 'error: some caption had no words?'

    print('encoded captions to array of size ' + str(L.shape))
    return L, caption_start_ix, caption_end_ix, caption_length


def main(params):
    dataset = json.load(open(params['data_json'], 'r'))
    imgs = dataset['images']

    # seed(123)  # make reproducible
    # shuffle(imgs)  # shuffle the order

    # tokenization and preprocessing

    # create the vocab
    vocab = build_vocab(imgs, params)
    itow = {i: w for i, w in enumerate(vocab)}  # a 1-indexed vocab translation table
    wtoi = {w: i for i, w in enumerate(vocab)}  # inverse table

    # assign the splits
    # assign_splits(imgs, params)

    # encode captions in large arrays, ready to ship to hdf5 file
    L, caption_start_ix, caption_end_ix, caption_length = encode_captions(imgs, params, wtoi)

    # create output h5 file
    N = len(imgs)
    h5_f = h5py.File(params['output_path'] + params['output_h5'] % params['img_size'], "w")
    h5_f.create_dataset("captions", dtype='uint32', data=L)
    h5_f.create_dataset("caption_start_ix", dtype='uint32', data=caption_start_ix)
    h5_f.create_dataset("caption_end_ix", dtype='uint32', data=caption_end_ix)
    h5_f.create_dataset("caption_length", dtype='uint32', data=caption_length)
    h5_imageset = h5_f.create_dataset("images", (N, 3, params['img_size'], params['img_size']), dtype='uint8')  # space for resized images

    train_list = []
    val_list = []
    test_list = []
    
    for i, img in enumerate(imgs):
        if img['split'] == 'train':
            img_path = params['train_images']
            train_list.append(i)
        else:
            img_path = params['valid_images']
            if img['split'] == 'val':
                val_list.append(i)
            elif img['split'] == 'test':
                test_list.append(i)
        # load the image
        I = imread(os.path.join(img_path, img['filename']))
        try:
            Ir = imresize(I, (params['img_size'], params['img_size']))
        except:
            print('failed resizing image %s - see http://git.io/vBIE0' % (img['filename'],))
            raise
        # handle grayscale input images
        if len(Ir.shape) == 2:
            Ir = Ir[:, :, np.newaxis]
            Ir = np.concatenate((Ir, Ir, Ir), axis=2)
        # and swap order of axes from (256,256,3) to (3,256,256)
        Ir = Ir.transpose(2, 0, 1)
        # write to h5
        h5_imageset[i] = Ir
        if i % 1000 == 0:
            print('processing %d/%d (%.2f%% done)' % (i, N, i * 100.0 / N))
    h5_f.close()

    print('wrote ', params['output_h5'] % params['img_size'])

    print('writing train/valid/test idx to files')
    with open(params['output_path'] + 'train_list.pkl', 'wb') as fp: pickle.dump(train_list, fp)
    with open(params['output_path'] + 'val_list.pkl', 'wb') as fp: pickle.dump(val_list, fp)
    with open(params['output_path'] + 'test_list.pkl', 'wb') as fp: pickle.dump(test_list, fp)

    # create output json file
    out = dataset
    out['idx2word'] = itow  # encode the (1-indexed) vocab
    out['word2idx'] = wtoi  # encode the (1-indexed) vocab
    out['images'] = imgs

    json.dump(out, open(params['output_path'] + params['output_json'], 'w'))
    print('wrote ', params['output_json'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--data_json', # required=True,
                        default='/home/memray/Data/coco/zhegan/dataset.json', help='input json file to process into hdf5')

    parser.add_argument('--train_images', default='/home/memray/Data/coco/train2014/', # required=True,
                        help='root location in which images are stored, to be prepended to file_path in input json')
    parser.add_argument('--valid_images', default='/home/memray/Data/coco/val2014/', #required=True,
                        help='root location in which images are stored, to be prepended to file_path in input json')

    parser.add_argument('--output_path', default='/home/memray/Data/coco/output/', help='path of output')
    parser.add_argument('--output_json', default='coco_data.json', help='output json file, no image')
    parser.add_argument('--output_h5', default='coco_img_%d.hdf5', help='output h5 file, containing everything')

    # options
    parser.add_argument('--img_size', default=256, type=int) # 256 is used in NeuralTalk model, 64 is in TriangleGAN
    parser.add_argument('--max_length', default=20, type=int,
                        help='max length of a caption, in number of words. captions longer than this get clipped.')
    parser.add_argument('--word_count_threshold', default=4, type=int,
                        help='only words that occur more than this number of times will be put in vocab, threshold=4 is used in NeuralTalk')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent=2))
    main(params)
