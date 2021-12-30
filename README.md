## PyTorch Implementation of SimulLR

There is an interesting work[1] about simultaneous lipreading, and I implement the main modules on GRID dataset.

### Environment
Python3 + PyTorch1.6
### Usage
Training
```
python main.py --train
``` 
Evaluating
```
python main.py --evaluate
``` 
### reference
[1] Zhijie Lin, Zhou Zhao, Haoyuan Li, Jinglin Liu, Meng Zhang, Xingshan Zeng, and Xiaofei He. Simullr: Simultaneous lip reading transducer with attention-guided adaptive memory. In MM ’21: ACM Multimedia Conference, Virtual Event, China, October 20 - 24, 2021, pages 1359–1367. ACM, 2021.