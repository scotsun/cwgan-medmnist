# 685 project - Evaluate Generative Models for MedMNIST

## box shared folder for all saved model checkpoints
https://duke.app.box.com/folder/237418820952?s=2hs40qz2t23axe8zdtufvidj5c8bkf17&tc=collab-folder-invite-treatment-b
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

1. Review all progress  
2. Experiments plan (v/w _ oh/e32 _ blood _ 10.pt)
	* vanilla vs. Wasserstein 10 vs 100
	* embedding vs. one-hot (output, efficiency) 
	* embedding size (output, efficiency)
		4, 8, 32
	* latent p(z) clipping (images may look good, but what about fid?)

**3rd-4th meeting (12.1 & 12.4):**

1. Reviewed all progress
2. Discussed about model results
3. Drafted report

## A more efficient way to calculate a 'proxy' of FID
*Note: this fast algorithm overestimate the FID calculated using a large fake image set. Thus, the two scores are not comparable. However, it can be compared with itself over training.*

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
