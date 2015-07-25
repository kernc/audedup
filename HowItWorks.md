# Outline #
```
        +----------------------------+
        |         audio file         |
        | (mp3, ogg, wav, flac, ...) |
        +----------------------------+
                      ↓                 _
        +----------------------------+   |
        |  signed 8-bit PCM signal   |   |
        +----------------------------+   |
                      ↓                  |
        +----------------------------+   |
        | stereo to mono conversion  |   |  preprocessing
        +----------------------------+   | (FFmpeg / MPlayer)
                      ↓                  |
        +----------------------------+   |
        |  downsampling to 11025 Hz  |   |
        +----------------------------+  _|
                      ↓                 _      _
        +----------------------------+   |      |
        |   FFT on ~6 seconds long,  |   |      |
        |     blackman-windowed      |   |      |
        |         extracts           |   |      |
        +----------------------------+   |      |
                      ↓                  |      |
        +----------------------------+   |      |
        |  local maxima selection =  |   |      |
        |     N most significant     |   | loop |
        |        frequencies         |   |      |
        +----------------------------+   |      |  feature
                      ↓                  |      | extraction
        +----------------------------+   |      |
        |     energy compaction =    |   |      |
        |    map frequencies onto    |   |      |
        |   12-note chromatic scale  |   |      |
        +----------------------------+  _|      |
                      ↓                         |
        +----------------------------+          |
        |      tempo/repetition      |          |
        |    identification with     |          |
        | FFT-based auto-correlation |          |
        +----------------------------+         _|
                      ↓                 _
        +----------------------------+   |
        |    cosine similarity of    |   |
        |    tempos of two tracks    |   |
        +----------------------------+   |
                      |                  |
                      | if within        |
                      | acceptable range |
                      ↓                  |
        +----------------------------+   |
        |    DTW on chroma-space     |   | clustering
        | spectrograms of two tracks |   |
        +----------------------------+   |
                      |                  |
                      | if within        |
                      | acceptable range |
                      ↓                  |
        +----------------------------+   |
         \    mark tracks as in     /    |
          \    the same cluster    /     |
           +----------------------+     _|
```
That's the gist of it.

The dynamic time warping (DTW) stage, though result-wise statistically significant, appears to be the [bottleneck](ExampleRun.md) of the clustering procedure. In any future versions, I'd like to see it replaced by a more robust, yet faster method, i.e. some kind of perceptual/locality-sensitive hashing (like with [LSHash](https://github.com/kayzh/LSHash)) or other spatial access method allowing ranged queries (e.g. X-tree).