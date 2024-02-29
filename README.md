# Whittle Index with Multiple Actions and State Constraint for Inventory Management

This directory is the released source code for the ICLR-24 paper "Whittle Index with Multiple Actions and State Constraint for Inventory Management".

## Install Environment

```bash
cd env
python setup.py install
```

## Run 

```bash
bash run.sh 
```

## Misc

* Do not use the latest version of gym (TOO MANY incompatible changes in Gym!): ``pip install gym==0.24.1`` 
* Owing to data privacy, the data files are removed. You can refer to ``env/ReplenishmentEnv/data`` for the data format and use your own data or synthetic data.