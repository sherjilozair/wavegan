if __name__  == '__main__':
  import argparse
  import glob
  import os
  import random
  import sys

  import numpy as np
  import tensorflow as tf
  from tqdm import tqdm

  parser = argparse.ArgumentParser()

  parser.add_argument('in_dir', type=str)
  parser.add_argument('out_dir', type=str)
  parser.add_argument('--name', type=str)
  parser.add_argument('--ext', type=str)
  parser.add_argument('--fs', type=int)
  parser.add_argument('--nshards', type=int)
  parser.add_argument('--slice_len', type=int)
  parser.add_argument('--first_only', action='store_true', dest='first_only')

  parser.set_defaults(
      name='train',
      ext='wav',
      fs=16000,
      nshards=1,
      slice_len=16384,
      first_only=True)

  args = parser.parse_args()

  audio_fps = glob.glob(os.path.join(args.in_dir, '*.{}'.format(args.ext)))
  from collections import Counter
  counter = Counter()
  for fps in audio_fps:
    audio_name = os.path.splitext(os.path.split(fps)[1])[0]
    traits = audio_name.split('_')[:-1]
    counter.update(traits)
    assert len(traits) > 0
  di = sorted(counter.keys())

  random.shuffle(audio_fps)

  if args.nshards > 1:
    npershard = int(len(audio_fps) // (args.nshards - 1))
  else:
    npershard = len(audio_fps)

  audio_fp = tf.placeholder(tf.string, [])
  audio_bin = tf.read_file(audio_fp)
  samps = tf.contrib.ffmpeg.decode_audio(audio_bin, args.ext, args.fs, 1)[:, 0]
  if args.slice_len is not None:
    if args.first_only:
      pad_end = True
    else:
      pad_end = False

    slices = tf.contrib.signal.frame(samps, args.slice_len, args.slice_len, axis=0, pad_end=pad_end)

    if args.first_only:
      slices = slices[:1]
  else:
    slices = tf.expand_dims(samps, axis=0)

  sess = tf.Session()

  if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

  for i, start_idx in tqdm(enumerate(range(0, len(audio_fps), npershard))):
    shard_name = '{}-{}-of-{}.tfrecord'.format(args.name, str(i).zfill(len(str(args.nshards))), args.nshards)
    shard_fp = os.path.join(args.out_dir, shard_name)

    writer = tf.python_io.TFRecordWriter(shard_fp)

    for _audio_fp in audio_fps[start_idx:start_idx+npershard]:
      audio_name = os.path.splitext(os.path.split(_audio_fp)[1])[0]
      splits = audio_name.split('_')
      audio_labels = splits[:-1]
      audio_id = splits[-1]

      label = np.zeros(len(di),)
      for l in audio_labels:
          label[di.index(l)] = 1.0
      label /= label.sum()

      try:
        _slices = sess.run(slices, {audio_fp: _audio_fp})
      except:
        continue

      if _slices.shape[0] == 0 or _slices.shape[1] == 0:
        continue

      for j, _slice in enumerate(_slices):
        example = tf.train.Example(features=tf.train.Features(feature={
          'label': tf.train.Feature(float_list=tf.train.FloatList(value=label)),
          'slice': tf.train.Feature(int64_list=tf.train.Int64List(value=[j])),
          'samples': tf.train.Feature(float_list=tf.train.FloatList(value=_slice))
        }))

        writer.write(example.SerializeToString())

    writer.close()

  sess.close()
