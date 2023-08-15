# L2T
Source code for [Learn to Tour: Operator Design For Feasible Solution Mapping ]()

# Data
The default path to data instances is `data/tsppdlib/instances/random-uniform`, which is from [TSPPD Test Instance Library](https://github.com/grubhub/tsppdlib). Or you could use `--instance_dir` to customize the directory path.

```
.
├── L2T
│   ├── src
│   ├── data
│   │   ├── tsppdlib
│   │   │   ├── instances
│   │   │   │   ├── random-uniform
│   │   │   │   |   ├──random-002-00232.tsp
│   │   │   │   |   └──...
│   └── main.py
```

You could also provide the path to LKH3 results, which will be used for comparison during the training. It can be downloaded from [its website](http://webhotel4.ruc.dk/~keld/research/LKH-3/BENCHMARKS/PDTSP.tgz). The results are located at `TOURS/U`

To run the code, you could run `main.py`:
```
python main.py --num_O 5 -et 10000 -el 10000 --episode 10 --n_steps 2000 --n_gradient_steps 20 -lr 1e-3 --batch_size 1000 --lkh3_dir ../U --ortools_dir None
```
