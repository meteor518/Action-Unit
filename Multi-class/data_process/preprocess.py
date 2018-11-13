import argparse
import os
import glob
import cv2
import pandas as pd
import mxnet as mx
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', '-i', required=True)
    parser.add_argument('--txt', '-t', required=True, help='.txt file or directory of .txt files')
    parser.add_argument('--record', '-rec', required=True)
    parser.add_argument('--resize', '-s', type=int)
    args = parser.parse_args()

    tqdm.write('Arguments:')
    for k, v in vars(args).items():
        tqdm.write(' - {}: {}'.format(k, v))

    # Get input arguments
    image_dir = os.path.abspath(os.path.expanduser(args.image_dir))
    txt = os.path.abspath(os.path.expanduser(args.txt))
    txt_paths = [txt] if os.path.isfile(txt) else glob.glob(os.path.join(txt, '*.txt'))
    record_path = os.path.abspath(os.path.expanduser(args.record))
    image_size = args.resize or None

    # Check validity of argument values
    if not os.path.exists(image_dir):
        raise IOError('image_dir "{}" not exist!'.format(image_dir))
    if not os.path.exists(txt):
        raise IOError('txt "{}" not exist!'.format(txt))
    if image_size and image_size <= 0:
        raise ValueError('resize value expected to be > 0, but get {}'.format(image_size))

    os.makedirs(os.path.dirname(record_path), exist_ok=True)

    # Scan all images
    all_images = []
    sub_dirs = [name for name in os.listdir(args.image_dir) if os.path.isdir(os.path.join(args.image_dir, name))]
    for sub_dir in tqdm(sub_dirs, desc='Scanning images'):
        image_fnames = [name for name in os.listdir(os.path.join(args.image_dir, sub_dir))
                        if not name.startswith('.') and name.endswith('.jpg')]
        all_images.extend(image_fnames)
    all_images = set(all_images)
    tqdm.write('Found total {} images in "{}".'.format(len(all_images), args.image_dir))

    # Load all txt files
    aus = [1, 2, 4, 5, 6, 9, 12, 17, 20, 25, 26, 43]
    names = ['image'] + ['AU{:0>2}'.format(i) for i in aus]
    cols = [0] + [i + 1 for i in aus]
    types = dict(zip(names, ['str'] + ['int'] * len(aus)))
    start_index = 0
    txt_dfs = []

    # Load all txt files
    for path in tqdm(txt_paths, desc='Loading txt files'):
        txt_df = pd.read_csv(path, sep='\t', names=names, usecols=cols, dtype=types)
        txt_df['image'] = txt_df['image'].str[-22:]
        txt_df.index = list(range(0, txt_df.shape[0]))
        txt_dfs.append(txt_df)
        start_index += txt_df.shape[0]
    # Concatenate and reset index
    total_df = pd.concat(txt_dfs)
    tqdm.write('Found {} labels in {}'.format(total_df.shape[0], args.txt))
    total_df = total_df[total_df['image'].isin(all_images)]
    total_df.reset_index(drop=True, inplace=True)

    # Write record files
    record = mx.recordio.MXIndexedRecordIO(os.path.splitext(record_path)[0] + '.idx', record_path, 'w')
    for i, row in tqdm(total_df.iterrows(), total=total_df.shape[0], desc='Converting'):
        sub_dir = row['image'][:12]
        path = os.path.join(image_dir, sub_dir, row['image'])
        image = cv2.imread(path)
        if image_size:
            image = cv2.resize(image, (image_size, image_size))
        label = row['AU01':'AU43'].tolist()
        header = mx.recordio.IRHeader(flag=0, label=label, id=i, id2=0)
        s = mx.recordio.pack_img(header, image, quality=100, img_fmt='.jpg')
        record.write_idx(i, s)
    record.close()
