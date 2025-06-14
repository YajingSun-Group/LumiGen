# LumiGen

LumiGen is a de novo design framework for luminescent molecules with specific photophysical properties, featuring an integrated Molecular Generator, Spectral Discriminator, and Sampling Augmentor.

The Molecular Generator contains methods for transfer learning and molecular generation. The Spectral Discriminator includes methods for evaluating target optical properties. The Sampling Augmentor provides methods for iterative selection of elite subsets. The Results section presents the classification outcomes of MolElite and MolMediocrity after iterations.

![Image text](https://github.com/YajingSun-Group/LumiGen/blob/main/LumiGen.png)

### Requirements
```
absl-py==0.7.1
altair==3.1.0
attrs==19.1.0
backcall==0.1.0
bleach==1.5.0
certifi==2021.5.30
cycler==0.11.0
decorator==4.4.0
defusedxml==0.6.0
entrypoints==0.3
h5py==2.9.0
html5lib==0.9999999
inflect==2.1.0
ipykernel==5.1.1
ipython==7.7.0
ipython-genutils==0.2.0
ipywidgets==7.5.1
jedi==0.14.1
Jinja2==2.10.1
jsonschema==3.0.2
jupyter==1.0.0
jupyter-client==5.3.1
jupyter-console==6.0.0
jupyter-core==4.5.0
Keras==2.2.0
Keras-Applications==1.0.2
Keras-Preprocessing==1.0.1
kiwisolver==1.3.1
llvmlite==0.29.0
Markdown==3.1.1
MarkupSafe==1.1.1
matplotlib==3.2.1
mistune==0.8.4
mkl-fft==1.0.12
mkl-random==1.0.2
nbconvert==5.5.0
nbformat==4.4.0
notebook==6.0.0
numba==0.45.1
numpy==1.16.4
olefile==0.46
pandas==0.24.0
pandocfilters==1.4.2
parso==0.5.1
pexpect==4.7.0
pickleshare==0.7.5
Pillow==6.1.0
prometheus-client==0.7.1
prompt-toolkit==2.0.9
protobuf==3.9.1
ptyprocess==0.6.0
Pygments==2.4.2
pyparsing==3.0.4
pyrsistent==0.15.4
python-dateutil==2.8.0
pytz==2019.1
PyYAML==5.1.2
pyzmq==18.0.2
qtconsole==4.5.2
scikit-learn==0.20.3
scipy==1.3.0
seaborn==0.9.0
Send2Trash==1.5.0
six==1.12.0
tensorflow==1.5.0
tensorflow-tensorboard==1.5.1
terminado==0.8.2
testpath==0.4.2
toolz==0.10.0
tornado==6.1
traitlets==4.3.2
umap-learn==0.3.8
wcwidth==0.1.7
webencodings==0.5.1
Werkzeug==0.15.5
widgetsnbextension==3.5.1
```
Once the installation is done, you can activate the virtual conda environement for this project:

```
source activate LumiGen
```
Please note that you will need to activate this virtual conda environement every time you want to use this project. 

### How to run the Molcular Generator
```
cd MolcularGenerator/experiments/
bash run_morty.sh ../data/test_chrom_performance.txt
```
The results of the analysis can be found in experiments/results

### How to run the Spectral Discriminator
you can use Spectral Discriminator for prediction cocrystal density by the following command.
```
python graph_main7.py --dir='FWHM_sol.csv'
```





