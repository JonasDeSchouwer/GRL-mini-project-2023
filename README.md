# GRL-mini-project-2023

$$
\newcommand{\mca}{\mathcal{A}}
\newcommand{\mcb}{\mathcal{B}}
\newcommand{\lrelu}{\text{LeakyReLU}}
\newcommand{\feat}{\boldsymbol{h}}
\newcommand{\nbh}{\mathcal{N}}
\newcommand{\softmax}{\text{softmax}}
$$

Welcome to the (anonymized) git repository for my GRL mini project. Using this codebase, I compared the ability of two architectures to capture correlation between the input features and the target labels in a node classification task.
The first studied architecture is GATv2 ($\mca$) \cite{GATv2}, consisting of GATv2 layers interleaved with nonlinearities. The second studied architecture ($\mcb$) is an extension thereof, consisting of a slightly modified type of layers interleaved with nonlinearities:



$$
\mca_{\text{layer}} :
\begin{cases}
    e_{ij}^{(t)} = {\boldsymbol{a}^{(t)}}^T \lrelu\left(
    \begin{pmatrix}
        W_l^{(t)} & W_r^{(t)}
    \end{pmatrix}
    \begin{pmatrix}
        \feat_i^{(t-1)} \\
        \feat_j^{(t-1)}
    \end{pmatrix}
    \right)

    & \text{for } j\in\nbh_i
    \\
    \alpha_{ij}^{(t)} = \softmax_j(e_{ij}^{(t)})
    \\
    \feat_i^{(t)} = \sum_{j\in\nbh_i} \alpha_{ij}^{(t)} W_r^{(t)} \feat^{(t-1)}_j
\end{cases}
$$


$$
\mcb_{\text{layer}} :
\begin{cases}
    e_{ij}^{(t)} = {\boldsymbol{a}^{(t)}}^T \lrelu\left(
    \begin{pmatrix}
        W_l^{(t)} & W_r^{(t)}
    \end{pmatrix}
    \begin{pmatrix}
        \feat_i^{(t-1)} \\
        \feat_j^{(t-1)}
    \end{pmatrix}
    \right)

    & \text{for } j\in\nbh_i
    \\
    \alpha_{ij}^{(t)} = \softmax_j(e_{ij}^{(t)})
    \\
    \feat_i^{(t)} = W_l^{(t)}\feat_i^{(t-1)} + \sum_{j\in\nbh_i} \alpha_{ij}^{(t)} W_r^{(t)} \feat^{(t-1)}_j
\end{cases}
$$


## Repository structure

### Files (code)

| File             | Description                                                                                                     |
|------------------|-----------------------------------------------------------------------------------------------------------------|
| `dataset.py`     | Code for generating, loading, and saving the dataset.                                                            |
| `gatv2.py`       | Implementation of the $\mca$ and $\mcb$ single-layer and two-layer architectures.                                |
| `experiment.py`  | Code for training each of the four models on all 10 sections of the dataset.                                      |
| `baseline.py`    | Evaluation of the *argmax* model on the datasets.                                                          |
| `figures.ipynb`  | Jupyter notebook used to plot figures in the project report. Accuracies were copied from TensorBoard.           |

### Folders (results)

- `mini-study`
  = a smaller-scale pilot study I conducted before doing any heavy experiments
  - `datasets`: Contains datasets for the mini-study.
  - `runs`: Contains binary log files for each run, viewable with TensorBoard.

- `study1`, `study2`, `study3`
  = the three independent runs of the main study, from which the median accuracies were taken
  - `datasets`: Identical datasets for studies 1, 2, and 3.
  - `runs`: Contains binary log files for each run, viewable with TensorBoard.