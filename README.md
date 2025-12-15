# LumiGen

LumiGen is a de novo design framework for luminescent molecules with specific photophysical properties, featuring an integrated Molecular Generator, Spectral Discriminator, and Sampling Augmentor.

The Molecular Generator contains methods for transfer learning and molecular generation. The Spectral Discriminator includes methods for evaluating target optical properties. The Sampling Augmentor provides methods for iterative selection of elite subsets. The Results section presents the classification outcomes of MolElite and MolMediocrity after iterations.

![Image text](https://github.com/YajingSun-Group/LumiGen/blob/main/LumiGen.png)

```
bash install_linux.sh
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





