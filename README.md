# Path Planning Techniques for Robotic Grasping

This repository contains the source code for the bachelor's thesis *"Path Planning Techniques for Robotic Grasping"*, developed at the Czech Technical University (CTU) in Prague.

## Author

**Leoš Drahotský**  
Email: [drahotskyl@gmail.com](mailto:drahotskyl@gmail.com)

## Included Third-Party Libraries

This repository includes modified versions of the following third-party libraries. Each is subject to its original license:

- [MPNN Library](https://lavalle.pl/software/mpnn/mpnn.html)  
  → See [`pp4rg/mpnn/README.md`](pp4rg/mpnn/README.md) for more information.

- [Jogramop Library](https://mrudorfer.github.io/jogramop-framework/)  
  → See [`pp4rg/benchmark/jogramop/README.md`](pp4rg/benchmark/jogramop/README.md) for more information.

- [RM4D Library](https://mrudorfer.github.io/rm4d/)  
  → See [`pp4rg/fmap/README.md`](pp4rg/fmap/README.md) for more information.

## Repository Structure

- `pp4rg/`: Contains the source code for various path planning algorithms. This directory also functions as a standalone Python library.
- `experiments/`: Scripts to reproduce the benchmarks presented in the thesis.

To run the benchmark script (`experiments/jogramop-benchmark.py`), you must first:
1. Construct the feasibility maps using `experiments/create_fmaps.py`
2. Train the autoencoder model using `experiments/train_model.py`

## Installation

> **Tested on:** Ubuntu 24.04.2 LTS  
> **Python version:** 3.10

Follow these steps to install and run the experiments:

### 1. Clone the latest version of the repository

```bash
git clone https://github.com/leo-iv/pp4rg.git
cd pp4rg
```

### 2. Create a virtual environment and install the pp4rg package
```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
```


### 3. Build MPNN python bindings
```bash
cd pp4rg/mpnn/
make
cd ../../
```

### 4. Create the feasibility map and train the Autoencoder model (may take several hours)

```bash
python3 experiments/create_fmaps.py
python3 experiments/train_model.py
```

### 5. Run the benchmarks

```bash
python3 experiments/jogramop_benchmark.py
```

## License

This project (excluding the third-party libraries listed above) is licensed under the MIT License.
See the [LICENSE](LICENSE) file for details.
