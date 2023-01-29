from TCGPR import TCGPR
dataSet = "FMOdataset.csv"
sampling_cap = 1
# for fdata screening (backward)
# ratio is recommend as a float <= 0.01
ratio = 0.1
up_search = 100

TCGPR.fit(filePath = dataSet,Sequence = 'backward',sampling_cap = sampling_cap, ratio = ratio)