# 685 project - Evaluate Generative Models for MedMNIST

# Explanation on Code

## `model.py`
Note: All GAN models in the code follow the revised architecture mentioned in the report.

`weights_init` intializes parameters in a GAN model.

`CondGenerator`, `CondDiscriminator` are the build blocks of `CDCGAN`, which is in the vanilla framework.

`CondWGenerator`, `CondWDiscriminator` are the build blocks of `CWDCGAN`, which is in the Wasserstein framework. `gradient_penalty` is used to calculate the penalty loss during the training of `CWDCGAN`.

`CNN` "internalizes" a ResNet50 model and can return image embedding in the forward pass. `CNN.calculate_embedding` is used to compute the embeddings for the entire dataloader.

`generate_synthetic_images` generates fake images (in the transformed formate where all values are between -1 and 1) for a given array of labels. If we want to visualize them, we have do the inverse transformation: `img=img*0.5+0.5`.

## `fid.py`
`fid_base` is the core function to calculate FID score given a well-trained CNN, a Generator, and training data. It can calculate both the uncondtional score and a condtional score for a specified class label.  
`get_class_weight` is a utility function to calculate the proportion of each category given a dataloader.  
`extract_k_class` is a utility function to extract all the data of class `k` and store them in another dataloader to output.

`fid` & `fid_handler` are used to calculate FID without using `scipy.linalg.sqrtm`. This is not covered by the report as this approach tend to overestimate the results. Detailed information is provided in Appendix.

## `experiment_utils.py`
`visualize_history` visualizes the training history of model.  
`compare_real_fake_by_class` compares real and fake images through a stratification by class labels.

# Appendix
## Fast FID
*Note: this fast algorithm overestimate the FID calculated using a smaller fake image set*

Given the formula of FID

$$
\mathrm{FID} = 
\| \mu_r - \mu_f \|_2^2 + \mathrm{tr}(\Sigma_r + \Sigma_f - 2(\Sigma_r\Sigma_f)^{1/2})
$$

Calculating $(\Sigma_r\Sigma_f)^{1/2}$ with standard method provided by `scipy.linalg.sqrtm` requires a lot of computation power as the embedding dimension is usually large (i.e. 2048). Inspired by Mathiasen and Hvilshøj (https://arxiv.org/abs/2009.14075), a more efficient way is as follows:

	1. Compute mu_r, C_r (normalized, recentered batch embeddings s.t. Sigma_r = <C_r, C_r>) for reals once
	2. Sample B batches (batch size m << d) from target population and generate fakes
		For each batch:
		a. compute
			mu_fi (mean of batch embeddings), 
			C_fi
		b. tr[sqrtm(Sigma_r @ Sigma_fi)] = sum[sqrt(eigval(C_fi @ C_r.T @ C_r @ C_fi.T))]
		c. compute the batch specific FID
	3. Compute sample mean

## box shared folder for all saved model checkpoints
https://duke.app.box.com/folder/237418820952?s=2hs40qz2t23axe8zdtufvidj5c8bkf17&tc=collab-folder-invite-treatment-b
## Colab code vignette
The are a few examples in the notebook to show how different classes and methods work.
https://colab.research.google.com/drive/1tZj1e67tzmTYbI2lD53fXuu5KpgknP_v#scrollTo=0CJwKsFI1cjV
## write-up
https://www.overleaf.com/2636841369kygdtjgqmpfz#4f20dd

## Meeting memo
**1st meeting (11.3):**

0. Created a shared GitHub repo

1. Selected three datasets:  
	PneumoniaMNIST 	(5k+, G)  
	OrganAMNIST 	(50k+, G)  
	BloodMNIST		(10k+, C)

2. Planned a timeline for the project:  
	11.17 GAN,  
	11.24 Resnet,  
	12.1 Run through experiments and make sure the processes are all correct; fine-tune the hyper-parameters  
	12.8 final report due

**2nd meeting (11.22):**

1. Reviewed all progress  
2. Planned Experiments (v/w _ oh/e32 _ blood _ 10.pt)
	* vanilla vs. Wasserstein 10 vs 100 epochs
	* embedding vs. one-hot (output, efficiency) 
	* embedding size (output, efficiency) 4, 8, 32

**3rd-4th meeting (12.1 & 12.4):**

1. Reviewed all progress
2. Discussed about model results
3. Drafted report

**5th-6th meeting (12.6 & 12.7):**

1. Reviewed all progress
2. Discussed about visualizations of results
3. Continued on report


