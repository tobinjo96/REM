
overlap <- function(Pi, Mu, S){
  
  library(MixSim)
  
  ncomponents = length(Pi)
  
  nfeatures = length(Mu)/ncomponents
  
  Mu <- matrix(Mu, nrow = ncomponents, ncol = nfeatures, byrow = TRUE)
  
  S <- array(S, dim = c(nfeatures, nfeatures, ncomponents))
  
  Q <- MixSim::overlap(Pi = Pi, Mu = Mu, S = S)
  
  return(Q$OmegaMap)
}
