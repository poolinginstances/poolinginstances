# Collection of instances of the Pooling Problem

This repository contains a collect of randomly generated pooling instances, the scripts to create them, and results, e.g. for values for different relaxations.

The instances have been used in the paper 

> _Strong Convex Nonlinear Relaxations of the Pooling Problem_  
> James Luedtke, Claudia D'Ambrosio, Jeff Linderoth, Jonas Schweiger  
> ZIB-Report 18-12

and in the PhD thesis

> _Exploiting structure in non-convex quadratic optimization and
  gas network planning under uncertainty_  
> Jonas Schweiger  
> [Technische Universtät Berlin](http://dx.doi.org/10.14279/depositonce-6015)

## Instances

The instances are described in more details in the above mentioned paper and can be found in the `data` folder. For each instance, a GAMS compatible `.dat` and a JSON file with metadata is available.

## Instance generation

The instances have been generated randomly based on the classical pooling instance vom Haverly. The scripts to generate the instances are in the `scripts` folder. `makerandom.py` generated one instances. Call `makerandom.py --help` for the available options. The bash script `genrandom_addededges.sh` calls `makerandom.py` to generate all the instances.

## Results

The `results` folder contains result files for the instances.
