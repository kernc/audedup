
```
iam@localhost:~/audedup/ $ ./audedup.py --help
usage: audedup [OPTIONS] DIR [DIR ...]

audedup 0.1 - audio deduplication

positional arguments:
  DIR                   directory with audio files

optional arguments:
  -h, --help            show this help message and exit
  -f N, --fast N        use only first N seconds of each track (faster but
                        considerably less accurate -- use values above 120)
  -r, --recursive       recurse in sub-directories
  -t LIST, --filetypes LIST
                        filetypes to consider (default: mp3,ogg,flac,wma,mp4)
  --output FORMAT       a string representing output format of each cluster;
                        you can use: {0}=filename, {1}=relative loudness,
                        {2}=track length in seconds; example: "{0}\t{2}";
                        (default: "{0}")
  --save FILE           save preprocessed and analzsed audio features
  --load FILE           load saved audio features -- note, main processing
                        still takes place for any DIR arguments, so if you
                        do not want those, set DIR to one that contains no
                        audio files, or set --filetypes to some "random"
  -v, --verbose         verbose intermediate output instead of just the result

Program outputs (to stdout) clusters (separated by a blank line) of similar
files (separated by a single "\n"). See website for more info:
http://code.google.com/p/audedup/
```
The full usage string is:
```
audedup [-h] [-f N] [-r] [-t LIST] [--output FORMAT] [--save FILE]
        [--load FILE] [-v]
        DIR [DIR ...]
```
I think the options are pretty self-explanatory!

See an ExampleRun with only one argument. Or see HowItWorks.