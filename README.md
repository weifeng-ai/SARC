# SARC

This repository is the implementation of ([Split-And-ReCombine network (SARC)](https://ieeexplore.ieee.org/document/8995317)). SARC is an embedding-based model which utilizes a split-and-recombine strategy for knowledge-based recommendation problem.

> W. Zhang, Y. Cao and C. Xu, "SARC: Split-and-Recombine Networks for Knowledge-Based Recommendation," 2019 IEEE 31st International Conference on Tools with Artificial Intelligence (ICTAI), Portland, OR, USA, 2019, pp. 652-659, doi: 10.1109/ICTAI.2019.00096.

### Running the code
```
$ cd src
$ python preprocess.py --dataset music (or book, or movie)
$ python main.py --dataset music
```