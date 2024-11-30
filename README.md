# UPSCALE GNN

Official Implementation of UPSCALE GNN.

## Abstract

*Graph Neural Networks (GNNs) have become fundamental tools in tasks like node classification and link prediction. However, as these models grow in size and complexity, they often encounter significant computational and memory bottlenecks. In this work, we propose a novel application of the UPSCALE algorithm—originally designed for unconstrained channel pruning in convolutional neural networks—to optimize GNN node vectors. By eliminating redundant feature dimensions, UPSCALE enables faster inference and reduced memory usage without compromising accuracy. Our experiments on benchmark datasets demonstrate that this approach achieves substantial efficiency improvements while maintaining competitive performance across key tasks. These results underscore the potential of unconstrained pruning strategies to improve the scalability and efficiency of GNNs, making them more viable for real-world applications.*

## Related Works

1. [UPSCALE: Unconstrained Channel Pruning](http://arxiv.org/abs/2307.08771) by *Wan et al*.
2. [Channel Pruning for Accelerating Very Deep Neural Networks](http://arxiv.org/abs/1707.06168) by *He et al*.
3. [Accelerating large scale real-time GNN inference using channel pruning](https://dl.acm.org/doi/10.14778/3461535.3461547) by *Zhou et al*.
4. [Accelerating GNN Inference by Soft Channel Pruning](https://ieeexplore.ieee.org/document/10010603) by *Zhang et al*.
5. [GAT TransPruning: progressive channel pruning strategy combining graph attention network and transformer](https://peerj.com/articles/cs-2012) by *Lin et al*.
6. [Structured Pruning for Deep Convolutional Neural Networks: A survey](http://arxiv.org/abs/2303.00566) by *He and Xiao*.

## Resources

1. [notes](notes.md)
2. [repos](resources/repos/)
3. [papers](resources/papers/)
4. [algorithms](algorithms.md)
5. [overleaf](https://www.overleaf.com/project/66e051a3b4a9aa50440a7d6a)

### Pruning Tutorial

1. [Pytorch Pruning](resources/tutorial/pruning_tutorial.ipynb)
2. [Pytorch Structured Pruning](resources/tutorial/pruning_structured_tutorial.ipynb)

### How to Run the code

```Bash
upscale-gcnn/
├── upscale/
│   ├── __init__.py
│   ├── masking.py
│   ├── pruning.py
│   └── models/
│       ├── __init__.py
│       ├── gcn.py
│       └── gatV2.py
│       └── sage.py
│       └── graphTransformer.py
│       └── JK-NET.py
|
├── examples/
│   ├── example_gcnn.py
├── tests/
│   ├── test_masking.py
│   ├── test_pruning.py
│   └── test_models.py
├── requirements.txt
├── README.md
└── setup.py

```

```Python
import torch
from upscale.models.gcn import SimpleGCN
from upscale import MaskingManager, PruningManager

N, F = 100, 64  # Number of nodes and feature dimensions
x = torch.rand((N, F), device='cuda')  # Input tensor for graph data

model = SimpleGCN(input_dim=F, hidden_dim=32, output_dim=16).cuda()

masking_manager = MaskingManager(model)

masking_manager.importance().mask(amount=0.25)

pruning_manager = PruningManager(model)
pruning_manager.compute([x]).prune()

```

## Results

- [Baseline](baseline.md)
- [Upscale](results.md)
- [UPSCALE-VIG](vig_results.md)

## References

- He, Y., & Xiao, L. (2024). Structured Pruning for Deep Convolutional Neural Networks: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(5), 2900–2919. [https://doi.org/10.1109/TPAMI.2023.333461]
- He, Y., Zhang, X., & Sun, J. (2017). Channel Pruning for Accelerating Very Deep Neural Networks (arXiv:1707.06168). arXiv. [http://arxiv.org/abs/1707.06168]
- Lin, Y.-C., Wang, C.-H., & Lin, Y.-C. (2024). GAT TransPruning: Progressive channel pruning strategy combining graph attention network and transformer. PeerJ Computer Science, 10, e2012. [https://doi.org/10.7717/peerj-cs.2012]
- Wan, A., Hao, H., Patnaik, K., Xu, Y., Hadad, O., Güera, D., Ren, Z., & Shan, Q. (2023). UPSCALE: Unconstrained Channel Pruning (arXiv:2307.08771). arXiv. [http://arxiv.org/abs/2307.08771]
- Zhang, W., Sun, J., & Sun, G. (2022). Accelerating GNN Inference by Soft Channel Pruning. 2022 IEEE 13th International Symposium on Parallel Architectures, Algorithms and Programming (PAAP), 1–6. [https://doi.org/10.1109/PAAP56126.2022.10010603]
- Zhou, H., Srivastava, A., Zeng, H., Kannan, R., & Prasanna, V. (2021). Accelerating large scale real-time GNN inference using channel pruning. Proc. VLDB Endow., 14(9), 1597–1605. [https://doi.org/10.14778/3461535.3461547]
