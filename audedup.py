#!/usr/bin/python
# coding: utf-8

from __future__ import print_function, division

import os, sys, string, random, subprocess

try:
  import cPickle as pickle
except ImportError:
  import pickle

import numpy as np
from numpy import log2, sqrt
from numpy.fft import rfft, irfft, fftshift

try:
  import matplotlib.pyplot as plt   # matplotlib only for plotting stuff
except ImportError: pass            # don't have it? no worries.



__version__ = '0.1'



class Audedup:

  SRATE = 11025     # all audio will be (down)sampled to this rate
  WINLEN = 2**16    # WINDOW length when computing FFT (approx. 6 seconds)
  WINDOW = np.blackman(WINLEN)  # will hold (Blackman) WINDOW coefficients
  N_FREQS = 6       # local_maxima() by default returns so much most significant frequencies
  features = []     # eventually features that similarity will be computed from
  files = []        # eventually files to analyze
  args = None       # parsed sys.argv arguments
  rawfile = None    # used for storing PCM output from ffmpeg/mplayer
  
  
  # perhaps replace with audiolab in the future?
  ffmpeg_command = (
    "ffmpeg -loglevel quiet -i \"{{1}}\" -ac 1 -ar {{0}} -f s8 {0} \"{{2}}\"" )  # timearg; SRATE, src, dst
  mplayer_command = (
    "mplayer -really-quiet -vc null -vo null -SRATE {{0}} -af resample={{0}}:0:0,pan=1:0.5:0.5,channels=1,format=s8 -ao pcm:fast:nowaveheader:file=\"{{2}}\" {0} \"{{1}}\"" )  # timearg; SRATE, dst, src
  
  ffmpeg_timearg = '-t {0}'         # only use first {0} seconds
  mplayer_timearg = '-endpos {0}'   # if so specified
  
  
  
  def walk_directory(self, directory):
    already_added = {}
    for dirname, dirnames, filenames in os.walk(directory):
      for filename in filenames:
        if filename.endswith(self.args.filetypes):
          # TODO: make absolute
          filepath = os.path.join(dirname, filename)
          if not filepath in already_added:
            self.files.append(filepath)
            already_added[filepath] = True
            if self.args.verbose:
              print('Adding file {0}:'.format(len(self.files)), filepath, file=sys.stderr)
      
      if not self.args.recursive:
        break
  
  
  
  def random_string(self, size=20, chars=string.ascii_letters + string.digits):
    return ''.join([random.choice(chars) for x in range(size)])
  
  
  
  def local_maxima(self, vec, n_max=N_FREQS, neigh=-1, threshold=-1, start=3):
    neigh = 100 // (self.SRATE / self.WINLEN) if neigh == -1 else neigh
    threshold = vec.mean() + 2*vec.std() if threshold == -1 else threshold
    maximums = []
    i, l = start, len(vec)
    
    while i < l:
      maxval = vec[i:i+neigh].max()
      
      if maxval < threshold:
        i += neigh
        continue
      
      maxidx = vec[i:i+neigh].argmax()
      if maxidx == 0:
        maximums.append((vec[i], i))
        i += neigh
        continue
        
      i += maxidx
      
    maximums = sorted(maximums, reverse=True)[:n_max] # first n_max sorted by magnitude
    return maximums
  
  
  
  def frame_spectrum(self, pcm, frames=-1):
    if tuple == type(frames):  # tuple (start frame, frame count)
      (offset, count) = frames
    elif int == type(frames):
      if frames == -1:         # frames == -1 ==> cover whole pcm
        offset, count = 1, len(pcm) // self.WINLEN + 1
      else:
        offset, count = frames, 1
    
    #   |<=>|<=>|...
    # |<=>|<=>|<=>|... make sure spaces between frames get covered as well
    frames = [i*0.5 for i in range(2*offset, 2*(offset + count) - 1)]
    
    spectrum = np.zeros(self.WINLEN/2 + 1)
    
    for frame in frames:
      pcmpart = pcm[(frame - 1)*self.WINLEN : frame*self.WINLEN]
      if len(pcmpart) == self.WINLEN:
        spectrum += abs(rfft(pcmpart * self.WINDOW, n=self.WINLEN))
      else:
        spectrum += abs(rfft(pcmpart * np.blackman(len(pcmpart)), n=self.WINLEN))
        break   # pcm end; force last frame
      
    return spectrum / len(frames)
  
  
  
  def frequency_to_chroma(self, freq):
    # as per reversing the equation here: http://www.phy.mtu.edu/~suits/NoteFreqCalcs.html
    return round(((log2(freq / 440) * 12) % 12) + 12) % 12
  
  
  
  def preprocess_file(self, filename):
    """ run mplayer/ffmpeg to PCM-mono-downsample a file """
    rawfilename = ''
    for command in [self.mplayer_command, 
                    self.ffmpeg_command]:
      while True:
        rawfilename = self.random_string()
        if not os.path.exists(rawfilename):
          break
      
      if 0 != subprocess.call(
          command.format(self.SRATE, filename, rawfilename), 
          stdout=open(os.devnull, 'w'),
          stderr=subprocess.STDOUT,
          shell=True):
        os.remove(rawfilename)
        rawfilename = None
        continue
      
      break   # file is successfully converted
    return rawfilename


  
  def autocorr(self, x):
    """ multi-dimensional autocorrelation with FFT """
    X = rfft(x, n=(x.shape[1]*2-1), axis=1)
    xr = irfft(X * X.conjugate(), axis=1).real
    xr = fftshift(xr, axes=1)
    xr = xr.sum(axis=1)
    return xr

  
  
  def extract_features(self, filename, file_index):
    print('\r(Preprocessing file {0}/{1}) '.format(
      file_index, len(self.files)), file=sys.stderr, end='')
    
    rawfile = self.preprocess_file(filename)
    
    if rawfile is None:
      print('\nwarning: unable to convert file "{0}" to PCM.', 
        '(Either you\'re missing ffmpeg/mplayer/codec or the file is corrupted.)'.format(filename),
        'Skipping.', file=sys.stderr)
      return
    
    print('\r(Analyzing file     {0}/{1}) '.format(
      file_index, len(self.files)), file=sys.stderr, end='')
    
    pcm = np.fromfile(rawfile, dtype=np.int8)
    os.remove(rawfile)
    
    feature = {}
    feature['filename'] = filename
    feature['track_length'] = pcm.size / self.SRATE
    
    # among similar, relative magnitude discerns between (perceived) more vs less loud files
    rel_mag = pcm[pcm > 0].sum() + abs(pcm[pcm < 0]).sum()
    rel_mag /= 2 * pcm.size * abs(pcm).max()
    feature['relative_magnitude'] = rel_mag
    
    # TODO: make features representable and hashable so that similar files 
    # can be retrieved by a perceptual or range hash or tree or whatever
    
    n_windows = pcm.size // self.WINLEN
    chroma_time = np.zeros((n_windows, 12))
    chroma_amps = np.zeros((n_windows, 12))
    
    for i in range(n_windows):
      spectrum = self.local_maxima(self.frame_spectrum(pcm[i*self.WINLEN:], 1), 200)
      
      chroma, chroma_amp = np.zeros(12), np.zeros(12)
      for (amp, freq) in spectrum:
        ch = self.frequency_to_chroma(freq * self.SRATE / self.WINLEN)
        chroma[ch] += 1
        chroma_amp[ch] += amp
      
      chroma_time[i] = chroma
      chroma_amps[i] = chroma_amp / rel_mag   # normalize magnitude
    
    feature['chroma'] = chroma_time
    
    
    w = np.blackman(9)
    #~ w = np.hamming(14)
    #~ w = np.hamming(3)
    acorr = self.autocorr(chroma_amps)
    acorr = np.convolve(w/w.sum(), acorr, mode='valid')
    
    #~ plt.plot(acorr)
    #~ plt.show()
    #~ exit(1)
    
    tempos = self.local_maxima(acorr, 60, 10, 0, 1)
    (amps, offsets) = zip(*tempos)
    
    tempo = np.array(sorted(offsets))
    feature['tempo'] = tempo
    
    self.features.append(feature)
  
  
  
  def print_similar_clusters(self):
    def dist_to_closest(a, vec): return abs(vec - a).min()
    def idx_of_closest(a, vec): return abs(vec - a).argmin()
    def euclid_dist(a, b): return sqrt(((a-b) ** 2).sum())
    
    def cosine_similarity(a, b):
      numerator = np.dot(a, b)
      denominator = np.dot(sqrt(np.dot(a, a)), sqrt(np.dot(b,b)))
      if denominator == 0:
        return 1 if numerator == 0 else 0
      return numerator / denominator
    
    def cosine_similarity2(a, b):
      numerator = 0
      new_b = np.zeros(a.size)
      for i,v in enumerate(a):
        ib = idx_of_closest(v, b)
        numerator += v * b[ib]
        new_b[i] = b[ib]
      denominator = np.dot(sqrt(np.dot(a, a)), sqrt(np.dot(new_b,new_b)))
      if denominator == 0:
        return 1 if numerator == 0 else 0
      return numerator / denominator
    
    def DTW(a, b, w=0):
      if w == 0:
        w = max(a.shape[0], b.shape[0])/2
      w = int(max(w, abs(a.shape[0] - b.shape[0])))
      
      dtw = np.ones((a.shape[0], b.shape[0])) * np.inf
      dtw[0][0] = 0
      
      for i in range(0, a.shape[0]):
        for j in range(max(0, i-w), min(b.shape[0], i+w+1)):
          if i + j == 0:
            continue

          d = euclid_dist(a[i], b[j])
          ds = []
          if i > 0: ds.append(dtw[i-1][j])
          if j > 0: ds.append(dtw[i][j-1])
          if i > 0 and j > 0: ds.append(dtw[i-1][j-1])
          
          dtw[i][j] = d + min(ds)
      
      return dtw[a.shape[0] - 1][b.shape[0] - 1]
    
    def supposedly_are_same(f1, f2):
      tempo_dist = 0
      if f1['tempo'].size >= 2 and f2['tempo'].size >= 2:
        if f1['tempo'].size == f2['tempo'].size:
          tempo_dist = int(1000 * abs(cosine_similarity(f1['tempo'], f2['tempo']) - 1))
        else:
          tempo_dist = int(1000 * abs(cosine_similarity2(f1['tempo'], f2['tempo']) - 1))
      else:
        for t in f1['tempo']:
          tempo_dist += dist_to_closest(t, f2['tempo'])
      
      # skip if tempo doesn't match enough
      if tempo_dist > 2:
        return False
      
      dtw_dist = DTW(f1['chroma'], f2['chroma']) / (f1['track_length'] + f2['track_length'])
      
      # skip if DTW distance too large
      return True if dtw_dist <= 0.29 else False
    
    def mark_as_same(f1, f2):
      if 'cluster' in f1:
        if 'cluster' in f2:
          # if f1 and f2 have assigned clusters that aren't the same
          if f1['cluster'] != f2['cluster']:
            # mark the clusters as same
            mark_as_same.same_clusters[tuple(sorted([f1['cluster'], f2['cluster']]))] = True
        else: # if no cluster in f2
          # assign f2's cluster to be the same as f1's
          f2['cluster'] = f1['cluster']
      else: # if no cluster in f1
        if 'cluster' in f2:
          f1['cluster'] = f2['cluster']
        else: # if neither has assigned cluster
          f1['cluster'] = f2['cluster'] = mark_as_same.cluster_count
          mark_as_same.cluster_count += 1

    mark_as_same.same_clusters = {}
    mark_as_same.cluster_count = 0
    
    
    for i,f1 in enumerate(self.features):
      
      print('\r(Comparing file     {0}/{1}) '.format(
        i, len(self.features)), file=sys.stderr, end='')
      
      for f2 in self.features:
        
        # skip same files
        if f2['filename'] == f1['filename']:
          continue
        
        if not supposedly_are_same(f1, f2):
          continue
        
        # else mark f1 and f2 as same
        mark_as_same(f1, f2)
    
    
    if mark_as_same.cluster_count > 0:
      print('\r{0:30}\r'.format(''), end='', file=sys.stderr)
    
    same_clusters = mark_as_same.same_clusters.keys()
    already_printed = set()
    
    for c in range(mark_as_same.cluster_count):
      if c in already_printed:
        continue
      
      all_same, old_len = set([c]), 0
      
      while old_len != len(all_same):
        old_len = len(all_same)
        for i,j in list(same_clusters):
          if i in all_same or j in all_same:
            all_same.add(i)
            all_same.add(j)
            same_clusters.remove((i,j))
      
      already_printed.update(all_same)
      
      for f in self.features:
        if 'cluster' in f and f['cluster'] in all_same:
          print(self.args.output.format(
            f['filename'], 
            round(f['relative_magnitude'], 3), 
            round(f['track_length'], 1)))
      print('')


  
  def main(self):
    try:
      import argparse
    except ImportError:
      exit('audedup: error: Python2.7+ required.')
      
    parser = argparse.ArgumentParser(
      description='audedup {0} - audio deduplication'.format(__version__),
      epilog='Program outputs (to stdout) clusters (separated by a blank line)\
              of similar audio files (separated by a single "\\n"). See \
              website for more info: http://code.google.com/p/audedup/', 
      prog='audedup',
      usage='%(prog)s [OPTIONS] DIR [DIR ...]')
    parser.add_argument('directories', metavar='DIR', type=str, nargs='+',
      help='directory with audio files')
    parser.add_argument('-f', '--fast', metavar='N', type=int, default=False,
      help='use only first N seconds of each track (faster but considerably \
            less accurate -- use values above 120)')
    parser.add_argument('-r', '--recursive', action='store_true', default=False,
      help='recurse in sub-directories')
    parser.add_argument('-t', '--filetypes', metavar='LIST', default='mp3,ogg,flac,wma,mp4',
      help='filetypes to consider (default: mp3,ogg,flac,wma,mp4)')
    parser.add_argument('--output', metavar='FORMAT', default='{0}',
      help='a string representing output format of each cluster; you can use: \
            {0}=filename, {1}=relative loudness, {2}=track length in seconds; \
            example: "{0}\\t{2}"; (default: "{0}")')
    parser.add_argument('--save', metavar='FILE', default=False,
      help='save preprocessed and analyzed audio features')
    parser.add_argument('--load', metavar='FILE', default=False,
      help='load saved audio features -- note, main processing still takes \
      place for any DIR arguments, so if you don\'t want those, set DIR to one\
      that contains no audio files, or set --filetypes to some "random"')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
      help='verbose intermediate output instead of just the result')
                       
    self.args = parser.parse_args()
    self.args.filetypes = tuple(('.' + self.args.filetypes).replace(',', ',.').split(','))
    
    if not self.args.fast:
      self.ffmpeg_timearg = ''
      self.mplayer_timearg = ''
    else:
      self.mplayer_timearg = self.mplayer_timearg.format(self.args.fast)
      self.ffmpeg_timearg = self.ffmpeg_timearg.format(self.args.fast)
    
    self.mplayer_command = self.mplayer_command.format(self.mplayer_timearg)
    self.ffmpeg_command = self.ffmpeg_command.format(self.ffmpeg_timearg)
    
    if self.args.verbose:
      print('audedup {0} running (http://code.google.com/p/audedup/) ...'.format(__version__), file=sys.stderr)
      print('Runtime arguments:', file=sys.stderr)
      for (arg,val) in vars(self.args).items():
        print('  {0}: {1}'.format(arg, val), file=sys.stderr)
    
    for dirname in self.args.directories:
      self.walk_directory(dirname)
    
    if self.args.verbose:
      print('\nWill analyze', len(self.files), 'files:', file=sys.stderr)
    
    if self.args.load:
      self.features = pickle.load(open(self.args.load))
    
    for i,f in enumerate(self.files):
      self.extract_features(f, i + 1)
    
    if self.args.save:
      pickle.dump(self.features, open(self.args.save, 'w'), protocol=2)
    
    self.print_similar_clusters()
    
    #~ plt.show()




if __name__ == '__main__':
  Audedup().main()

