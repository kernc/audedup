# audedup run with default arguments #
Note, verbose output, if in effect via CommandLineArguments, is printed to standard error (stderr) while only the "calculation result" is printed to standard output (stdout).

```
iam@localhost:~/audedup/ $ ls -sh ../test/mp3/
total 121M
9.1M 01 Adele - Rolling In The Deep.mp3
8.8M Adele - Rolling In the Deep.mp3
2.4M Brand X Music - Celestial Life.mp3
2.4M Brand X Music - Celestial Life (No Choir).mp3
6.2M Brand X Music - Magical Mystery.mp3
6.2M Brand X Music - Magical Mystery (No Vox).mp3
5.0M Edward Maya feat. Alicia - Stereo Love (Radio Edit).mp3
3.9M Edward Maya feat. Vika Jigulina - Stereo love.mp3
 11M King Crimson - In the Court of the Crimson King - I Talk To The Wind.mp3
5.6M King Crimson - I Talk To The Wind.mp3
 13M Nero - Innocence(1).mp3
4.9M Nero - Innocence (Original Mix).mp3
2.3M Oktet 9 - Kiss the Girl.mp3
 12M Roger Waters - 06 - Comfortably Numb.mp3
9.3M The Departed Soundtrack - 02 Comfortably Numb.mp3
7.4M Van Morrison y Roger Waters_Comfortably numb.mp3
5.6M VINNIE MOORE - Lifeforce -.mp3
9.3M Vinnie Moore - Mind Eye - Lifeforce.mp3

iam@localhost:~/audedup/ $ time ./audedup.py ../test/mp3/
../test/mp3/Vinnie Moore - Mind Eye - Lifeforce.mp3
../test/mp3/VINNIE MOORE - Lifeforce -.mp3

../test/mp3/The Departed Soundtrack - 02 Comfortably Numb.mp3
../test/mp3/Roger Waters - 06 - Comfortably Numb.mp3
../test/mp3/Van Morrison y Roger Waters_Comfortably numb.mp3

../test/mp3/King Crimson - In the Court of the Crimson King - I Talk To The Wind.mp3
../test/mp3/King Crimson - I Talk To The Wind.mp3

../test/mp3/Brand X Music - Celestial Life.mp3
../test/mp3/Brand X Music - Celestial Life (No Choir).mp3

../test/mp3/Nero - Innocence(1).mp3
../test/mp3/Nero - Innocence (Original Mix).mp3

../test/mp3/01 Adele - Rolling In The Deep.mp3
../test/mp3/Adele - Rolling In the Deep.mp3

../test/mp3/Brand X Music - Magical Mystery (No Vox).mp3
../test/mp3/Brand X Music - Magical Mystery.mp3


real	0m29.157s
user	0m26.714s
sys	0m2.176s
```
You can see above that all duplicates are found correctly, with the exception of Edward Maya's Stereo Love, which in the "Radio Edit" case starts with a 2 minute intro not present in the other one. The other track left out has no duplicate.

I haven't (yet) tested it on a larger set because as it is now, the program takes quite a while to finish (in above case of 18 tracks, it hogs the CPU for ~30 seconds). I'd like to see that change in the future. :-/

The output is in a form easy to parse by other scripts/software. When it comes to choosing duplicates with audedup, I wouldn't put my faith in automation, though.