# 3D Map

## Quick Start

Conda environment

```bash
conda create --name 3D-Map python=3.11
```

### Dependencies

Pytorch 2.4.1

```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
```

Pytorch3d, open3d

```bash
pip install git+https://github.com/facebookresearch/pytorch3d.git@stable --no-build-isolation
pip install open3d
```

Pytorch Geometric

```bash
pip install torch_geometric==2.7.0
pip install torch_scatter -f https://data.pyg.org/whl/torch-2.4.0+cu118.html
```

Finally, install LIFT3D following <https://github.com/PKU-HMI-Lab/LIFT3D> exclude simulation environment

### Simulation Environments

MetaWorld

```bash
pip install git+https://github.com/Farama-Foundation/Metaworld.git@master#egg=metaworld
```

**TODO: RLBench**

**TODO: ManipSkill**

### Data 

MetaWorld, remember to change parameters in ```scripts/gen_data_metaworld.sh``` such as ```--save-dir```

```bash
bash scripts/gen_data_metaworld.sh
```

**TODO: RLBench**

**TODO: ManipSkill**

### Train and Eval

```scripts/lift3d_metaworld.sh``` for lift3d encoder, ```scripts/GNN.sh``` for PointNet encoder.

Note that **ALL PARAMETERS** can be changed!
