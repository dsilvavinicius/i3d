# High Order Derivative Learning for Graphics
[Tiago Novello [1]](blank),
[Vinícius da Silva [2]](https://dsilvavinicius.github.io/),
[Guilherme Schardong [2]](https://schardong.github.io/),
[Luiz Schirmer [2]](https://www.lschirmer.com),
[Hélio Lopes [2]](http://www-di.inf.puc-rio.br/~lopes/),
[Luiz Velho [1]](https://lvelho.impa.br/)
<br>
[1] Institute for Pure and Applied Mathematics (IMPA),
[2] Pontifical Catholic University of Rio de Janeiro (PUC-Rio)

This is the official implementation of "High Order Derivative Learning for Graphics".

## Get started: the Double Torus Toy Example

### Windows

#### Prerequisites

1. [Anaconda](https://www.anaconda.com/products/individual#Downloads).
2. [Git](https://git-scm.com/download/win).
3. [Integrate Git Bash with conda](https://discuss.codecademy.com/t/setting-up-conda-in-git-bash/534473).
4. [SHADERed](https://shadered.org/).

#### Setup (all folders are relative to the clone root)

1. Open Git Bash
2. Clone the repository: `git clone --recurse-submodules git@github.com:dsilvavinicius/high_order_derivative_learning_for_graphics.git`.
3. Enter project folder: `cd high_order_derivative_learning_for_graphics`.
4. Change branch to refactor: `git checkout refactor`
5. Setup project dependencies:
```
conda env create -f environment.yml
conda activate hodl
```
5. Download the [Double Torus Mesh](https://drive.google.com/file/d/11PkscMHBUkkENhHfI1lpH5Dh6X9f2028/view?usp=sharing) into the `data` folder in the repository.
6. Run the double torus toy example script: `./tools/double_torus/double_torus_toy_example.sh`. This script preprocess the mesh using `preprocess_double_torus.sh`, train the model using `train_double_torus.sh` and create the binary neural network files to be loaded in shaders using `double_torus_pth2bin.sh`, all under the `tools/double_torus/` folder.
7. The resulting binary neural network files will be `logs/double_torus/checkpoints/model_current_biases.bin`, and `model_current_weights.bin`.
8. Open SHADERed.
9. `File -> Open` and select the SHADERed project file `Shader-Neural-Implicits/NeuralImplicits.sprj`.
10. Load The binary network weights: `Objects -> weights -> LOAD RAW DATA` and select the file `logs/double_torus/checkpoints/model_current_weights.bin`.
11. Load The binary network biases: `Objects -> biases -> LOAD RAW DATA` and select the file `logs/double_torus/checkpoints/model_current_biases.bin`.

### Linux

We tested the build steps stated above on Ubuntu 20.04. The prerequisites and setup remain the same, since all packages are available for both systems. We also provide a ```Makefile``` to cover the running of all scripts on step 6, defined above.

SHADERed should show the following image:

![Double Torus](figs/double_torus.png "Double Torus")

## Citation
If you find our work useful in your research, please cite:
```
Bibtex here!
```

## Contact
If you have any questions, please feel free to email the authors.
