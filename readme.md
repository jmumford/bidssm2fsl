bidssm2fsl
===
This tool takes a <a href=
'https://fitlins.readthedocs.io/en/latest/model.html'>BIDS Statistical Model Spec </a> (json file) and produces the files required to run the analysis in FSL.

Model spec tutorials coming soon!  The model spec referred to in the 2022 OHBM poster is included in the repository as an example and corresponds to the <a href='https://openneuro.org/datasets/ds000003/versions/00001'>ds00003 rhyme judgement data</a>.  The model first estimates a word>psueodoword contrast at the first level.  At the group level age, mean centered, is modeled and the overall mean as well as age slope are tested.  See this <a href='https://mumfordbrainstats.tumblr.com/post/685068677462228992/bidssm2fsl-connecting-bids-statistical-model'> blog post </a> for a walk through the model spec example.


Installation
======
This tool depends upon FitLins, so follow the instructions on the <a href='https://fitlins.readthedocs.io/en/latest/installation.html#manually-prepared-environment-python-3-6'> FitLins webpage</a> to install Fitlins first.  

Next clone this repo into the directory of your choice using:
```
git clone https://github.com/jmumford/bidssm2fsl.git
```
 Note part of the FitLins installation requires creating a fitlins environment.  Be sure to activate the FitLins environment when using this tool.

Usage
===
Within the fitlins environment run using the following required input arguments
* bids_dir
    *  the root folder of a BIDS valid dataset (sub-XXXXX folders should be found at the top level in this folder).
* fmriprep_dir
    * the root folder of fmriprep data
* output_dir
    * the output path for the outcomes of preprocessing and visual reports
* database_path
    * Path to directory containing SQLite database indices for this BIDS dataset. If a value is passed and the file already exists, indexing is skipped.
* model
    * location of BIDS model description

Additional preprocessing settings, that are not in the model spec, are also controlled at the commandline.  This includes
```
Preprocessing settings for FSL:
  -s SMOOTHING, --smoothing SMOOTHING
                        Amount of spatial smoothing applied to data prior to model fit (FWHM mm kernel).  Default = 5mm.
  -omit_deriv, --omit_regressor_derivatives
                        Use this flag to omit derivatives of regressors in model.  Otherwise derivatives will be included.
  -hrf {none,gamma,double_gamma}, --hrf_type {none,gamma,double_gamma}
                        HRF to use in convolution. Default = double_gamma.
```

A call using the defaults (5mm smoothing, including temporal derivatives with a double gamma HRF) would then be:

```
> ./run_btf.py \
/Users/jeanettemumford/Projects/Data/ds003 \
/Users/jeanettemumford/Projects/Data/ds003_fmriprep \
/Users/jeanettemumford//Projects/Data/fsl_files \
/Users/jeanettemumford//Projects/Data/ds003/dbcache \
/Users/jeanettemumford/Projects/Data/model-ds0003_age_smdl.json
```

