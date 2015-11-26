function net = mlpinit(net, prior)
%MLPINIT Initialise the weights in a 2-layer feedforward network.
%
%	Description
%
%	NET = MLPINIT(NET, PRIOR) takes a 2-layer feedforward network NET and
%	sets the weights and biases by sampling from a Gaussian distribution.
%	If PRIOR is a scalar, then all of the parameters (weights and biases)
%	are sampled from a single isotropic Gaussian with inverse variance
%	equal to PRIOR. If PRIOR is a data structure of the kind generated by
%	MLPPRIOR, then the parameters are sampled from multiple Gaussians
%	according to their groupings (defined by the INDEX field) with
%	corresponding variances (defined by the ALPHA field).
%
%	See also
%	MLP, MLPPRIOR, MLPPAK, MLPUNPAK
%

%	Copyright (c) Ian T Nabney (1996-2001)

if isstruct(prior)
  sig = 1./sqrt(prior.index*prior.alpha);
  w = sig'.*randn(1, net.nwts); 
elseif size(prior) == [1 1]
  w = randn(1, net.nwts).*sqrt(1/prior);
else
  error('prior must be a scalar or a structure');
end  

net = mlpunpak(net, w);

end