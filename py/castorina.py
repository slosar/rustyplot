#
# Bias and shot noise from Emanuele Castorina
#
# Evolution of HI bias and shot noise as a function of redshift. Then nP includes nonlinear damping 
# From equation 4 and 5 of https://arxiv.org/abs/1609.05157. In equation 7, alpha=1 and M_{min} = 5*10^9 Msun/h to fit DLA bias b_{DLA}=2 at z=2.3
# The only ASSUMPTION is that M_{min} does not depend on redshift, note also that bias and shot noise are independent from \Omega_{HI}
# Not to be trusted below z~1.5
# | z | b_{HI}(z)  | n_{eff}(z)  |  nP_{0.2}(z) |
#
# AS: updated 2/25/18
# AS: update April 2019
import numpy as np
from scipy.interpolate import interp1d

### old version
# z_, bias_, nbar_, nP=np.array([[float(x) for x in line.split()] for line in 
# """0   	1.33659 	0.00066083  	0.582292 
# 0.15	1.361511	0.0008738778	0.8411342
# 0.3 	1.389651	0.001154436 	1.181139 
# 0.45	1.420774	0.001516439 	1.612019 
# 0.6 	1.454665	0.001974334 	2.142182 
# 0.75	1.491148	0.0025426   	2.779038 
# 0.9 	1.530082	0.003235229 	3.529192 
# 1.05	1.571361	0.004065181 	4.398496 
# 1.2 	1.614906	0.005043821 	5.391953 
# 1.35	1.660661	0.006180378 	6.513541 
# 1.5 	1.708584	0.007481427 	7.765987 
# 1.65	1.75865 	0.008950437 	9.150536 
# 1.8 	1.810838	0.0105874   	10.66672 
# 1.95	1.865138	0.01238857  	12.31219 
# 2.1 	1.921542	0.01434631  	14.08252 
# 2.25	1.980047	0.01644905  	15.97113 
# 2.4 	2.04065 	0.01868143  	17.96926 
# 2.55	2.103354	0.02102451  	20.06595 
# 2.7 	2.168158	0.02345614  	22.2481  
# 2.85	2.235067	0.02595143  	24.50065 
# 3.  	2.304082	0.02848329  	26.80676 
# 3.15	2.375207	0.03102304  	29.14803 
# 3.3 	2.448446	0.03354113  	31.5048  
# 3.45	2.523801	0.03600774  	33.8565  
# 3.6 	2.601276	0.03839353  	36.18197 
# 3.75	2.680874	0.04067025  	38.45986 
# 3.9 	2.762598	0.04281134  	40.669   
# 4.05	2.846452	0.04479248  	42.78875 
# 4.2 	2.932438	0.04659205  	44.79937 
# 4.35	3.020559	0.04819144  	46.68237 
# 4.5 	3.110817	0.04957541  	48.42076 
# 4.65	3.203216	0.05073216  	49.99938 
# 4.8 	3.297757	0.05165349  	51.40504 
# 4.95	3.394443	0.05233475  	52.62673 
# 5.1 	3.493276	0.05277476  	53.65572 
# 5.25	3.594257	0.0529756   	54.48567 
# 5.4 	3.697389	0.05294245  	55.11255 
# 5.55	3.802673	0.05268327  	55.53474 
# 5.7 	3.910111	0.05220847  	55.75285 
# 5.85	4.019706	0.05153062  	55.76968 
# 6.  	4.131457	0.05066401  	55.59005 
# 6.1  	4.131457	0.05066401  	55.59005 """.split("\n")]).T


z_, bias_, Psn_=np.array([[float(x) for x in line.split()] for line in 
"""0.2 	1.0826043	118.89453
0.3 	1.0826043	118.89453
0.35	1.1184118	120.58044
0.4 	1.1530133	122.0485 
0.45	1.1864525	123.30235
0.5 	1.2187726	124.34564
0.55	1.2500174	125.18202
0.6 	1.2802301	125.81513
0.65	1.3094544	126.24863
0.7 	1.3377337	126.48614
0.75	1.3651115	126.53134
0.8 	1.3916313	126.38785
0.85	1.4173365	126.05932
0.9 	1.4422707	125.54941
0.95	1.4664774	124.86175
1.  	1.49     	124.     
1.05	1.512882 	122.9678 
1.1 	1.5351669	121.7688 
1.15	1.5568983	120.40663
1.2 	1.5781195	118.88496
1.25	1.5988741	117.20743
1.3 	1.6192055	115.37767
1.35	1.6391573	113.39935
1.4 	1.6587729	111.2761 
1.45	1.6780959	109.01157
1.5 	1.6971696	106.60941
1.55	1.7160377	104.07326
1.6 	1.7347435	101.40678
1.65	1.7533306	98.613596
1.7 	1.7718425	95.697368
1.75	1.7903226	92.66174 
1.8 	1.8088144	89.51055 
1.85	1.8273531	86.259266
1.9 	1.8459621	82.941369
1.95	1.8646634	79.591894
2.  	1.883479 	76.245874
2.05	1.9024312	72.938343
2.1 	1.921542 	69.704335
2.15	1.9408335	66.578884
2.2 	1.9603278	63.597022
2.25	1.980047 	60.793784
2.3 	2.0000086	58.193943
2.35	2.0202115	55.781224
2.4 	2.04065  	53.529093
2.45	2.0613196	51.414446
2.5 	2.0822205	49.427908
2.55	2.103354 	47.563534
2.6 	2.1247214	45.814965
2.65	2.1463227	44.174184
2.7 	2.168158 	42.632761
2.75	2.1902271	41.182753
2.8 	2.2125301	39.818173
2.85	2.235067 	38.533522
2.9 	2.2578378	37.323454
2.95	2.2808428	36.18324 
3.  	2.304082 	35.108304
3.05	2.3275557	34.094243
3.1 	2.3512639	33.137354
3.15	2.375207 	32.234107
3.2 	2.399385 	31.381091
3.25	2.423798 	30.575374
3.3 	2.448446 	29.814142
3.35	2.4733291	29.094677
3.4 	2.4984473	28.414645
3.45	2.523801 	27.771807
3.5 	2.5493903	27.163996
3.55	2.5752152	26.589344
3.6 	2.601276 	26.046055
3.65	2.6275727	25.532394
3.7 	2.6541053	25.046856
3.75	2.680874 	24.587997
3.8 	2.7078789	24.15442 
3.85	2.7351202	23.744909
3.9 	2.762598 	23.358297
3.95	2.7903126	22.993454
4.  	2.8182639	22.649396
4.05	2.846452 	22.325176
4.1 	2.874877 	22.019878
4.15	2.9035389	21.732704
4.2 	2.932438 	21.462889
4.25	2.9615744	21.20969 
4.3 	2.990948 	20.972459
4.35	3.020559 	20.750573
4.4 	3.0504074	20.543428
4.45	3.0804933	20.350502
4.5 	3.110817 	20.171291
4.55	3.1413787	20.005309
4.6 	3.1721784	19.852135
4.65	3.203216 	19.711363
4.7 	3.2344916	19.582601
4.75	3.2660052	19.465513
4.8 	3.297757 	19.359776
4.85	3.3297472	19.26508 
4.9 	3.3619758	19.18116 
4.95	3.394443 	19.107763
5.  	3.4271487	19.044646
5.05	3.4600931	18.991606
5.1 	3.493276 	18.948452
5.15	3.5266975	18.915   
5.2 	3.5603578	18.891101
5.25	3.594257 	18.876615
5.3 	3.6283953	18.871408
5.35	3.6627726	18.875379
5.4 	3.697389 	18.888435
5.45	3.7322445	18.910487
5.5 	3.7673391	18.941478
5.55	3.802673 	18.981358
5.6 	3.8382462	19.030081
5.65	3.8740588	19.087627
5.7 	3.910111 	19.15398 
5.75	3.946403 	19.229132
5.8 	3.9829347	19.313107
5.85	4.019706 	19.405938
5.9 	4.0567169	19.507655
5.95	4.0939672	19.618291
6.0  	4.131457 	19.737877
6.1  	4.131457 	19.737877 """.split("\n")]).T
castorinaBias=interp1d(z_,bias_)
castorinaPn=interp1d(z_,Psn_)
