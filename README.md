# 685 project - Evaluate Generative Models for MedMNIST
 
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
2. Experiments plan
	* embedding vs. one-hot
	* embedding size
	* vanilla vs. Wasserstein
	* latent p(z) clipping

## More efficient way to calculate FID
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
			C_fi,
			Sigma_fi = <C_fi, C_fi>
		b. tr[sqrtm(Sigma_r @ Sigma_fi)] = sum[sqrt(eigval(C_fi @ C_r.T @ C_r @ C_fi.T))]
		c. compute the batch specific FID
	3. Compute sample mean
