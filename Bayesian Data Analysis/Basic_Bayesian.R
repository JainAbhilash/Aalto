#Generate a sequence 
x=seq(0,1,length=10000)
mean=0.2
var=0.01
#Calculating Alpha
alpha=mean*(((mean*(1-mean))/var)-1)
#Calculating Beta
beta=(alpha*(1-mean))/mean
#Plotting Density function of the Beta distribution
dist=dbeta(x,alpha,beta)
plot(dist)

dist1=rbeta(1000,alpha,beta)
hist(dist1)
mean(dist1)
var(dist1)
quantile(dist1,probs = (0.975))
quantile(dist1,probs=(0.025))
