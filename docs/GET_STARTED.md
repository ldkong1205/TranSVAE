<img src="../figs/logo.png" align="right" width="28%">

# Getting Started

After setting all necessary packages and libraries, you can now run experiments by the following scripts:

### HMDB2UCF
```shell
sh exp/train_script_I3D_hmdb2ucf.sh
```

### UCF2HMDB
```shell
sh exp/train_script_I3D_ucf2hmdb.sh
```

### Jester
```shell
sh exp/train_script_I3D_jester.sh
```

We have set a fixed random seed and the optimal hyperparameters for all experiments, so that the reported scores are directly reproducible.

If you prefer to use Jupyter Notebook, you can find an example in ``exp/TransVAE_I3D_hmdb-ucf.ipynb``.
