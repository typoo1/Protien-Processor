import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd
import seaborn as sns
import os
def run(fileName, guanFileName):
	# Note to Tye on 3/30/2016
	# This program assumes that the following two files are inputted from Flask
	# This program produces three sets of graphs
	# assume n unfolding control lines in the CSV file
	# 1) unfold_0.png to unfold_n-1.png
	# 2) unfoldCorrelation.png provides graph of cross-correlation plot between all
	# unfolding controls
	# 3) allUnfolds.png shows all of the normalzied unfolding controls on the same
	# plot with the mean with error bars.
	

	# String that defines if a row is a control
	# MAY NEED TO BE AN INPUT. Look at ways to automate this.
	controlString = 'Negative control'

	# Loads the plate reader file. Note that the first ten lines of
	# the file are ignored. First row after the header (provides times)
	# shows which columns to include in the averages.
	data  =  pd.read_csv(os.getcwd() + '/app/uploads/' + fileName, skiprows = 12, encoding = "ISO-8859-1")

	# If the data is missing from the file, replace the  n.a. with nan
	data.replace(to_replace = 'n.a.', value = np.nan, inplace = True)

	# Split the fileName for later use
	inFile = fileName.split('.')[0]

	# Load in the Guanadine concentations from a CSV file
	guanData =  pd.read_csv(os.getcwd() + '/app/uploads/' + guanFileName, encoding = "ISO-8859-1")

	# Set the guanidine concentration values to a list called guanConc from
	# the DataFrame
	guanConc = np.array(guanData['Denaturant conc. (M) with evap. correction'])


	# Determines the average of each column
	s = data.ix[0,:]
	colMin = [col for col in data.columns if 'min' in col]
	dataMin = s[colMin].astype(float)
	for i in range(1,10):
		t = np.where(np.array(dataMin) == i)
		if i == 1:
			averages = t
		elif np.size(t) > 0:
			averages = np.vstack((averages,t))

	# The first sample is included. Some times it will need to be
	# removed from the fit but still shown in the graph.

	average = 0
	protocol = 0
	sample = 0
	firstZero = True
	for column in data.columns:
		# -1 in first row designates a column that is only used for the
		# first protocal
		if data.ix[0,column] == -1:
		#if column == '0 min ':
			data.loc['protocol',column] = protocol
			data.loc['sample',column] = sample
			data.loc['average', column] = average
			data.loc['guanadine', column] = guanConc[average]
			average += 1
			sample += 1
		elif 'min' in column:
			if column[0:5] == '0 min':
				sample = 0
				protocol += 1
			data.loc['protocol',column] = protocol
			data.loc['sample',column] = sample  
			for avg in averages:
				if sample in avg:
					data.loc['average', column] = average
					data.loc['guanadine', column] = guanConc[average]
					# Is the current sample the last in the averaged group
					if sample == avg[-1]:
						average += 1
					break
			sample += 1

	# Remove the first row since it only contains the information about
	# which rows to average together
	data = data.iloc[1:,:]

	# Add columns avg0 through avgn toDataFrame which contains the average
	# for each of the averages based on three samples
	maxAverage = int(data.loc['average',:].max()) + 1
	avg = np.zeros(maxAverage)
	averages = pd.DataFrame(index = data.index,
							columns = ['avg' + str(i) for i in range(maxAverage)])
	for index, row in enumerate(data.iterrows()):
		for i in range(maxAverage):
			avg[i] = row[1][data.ix['average',:] == i].mean()
		averages.iloc[index,:] = avg
	data = data.merge(averages, how = 'left', left_index = True, right_index = True)
	data.to_csv(os.getcwd() + '/app/uploads/part1DataFinal.csv')

	# Find the unfolding or control rows
	s = data.dropna(subset=['Time'])
	s = s[s['Time'].str.contains(controlString)]

	# Use the columns that have avgn in them
	unfoldRows = s[['Time'] + [col for col in s.columns if 'avg' in col]]

	# Create the jpeg files that will be loaded by Flask into the web page
	unfoldSignal = pd.DataFrame(columns = ['unfold' + str(i) for i in range(len(unfoldRows))])
	for i, row in enumerate(unfoldRows.iterrows()):
		avgUnfold = np.array(row[1][1:], dtype=float)
		unfoldSignal['unfold' + str(i)] = avgUnfold
		plt.scatter(guanConc, avgUnfold)
		plt.plot(guanConc, avgUnfold, 'r')
		plt.xlabel('Guanidine Concentration (M)')
		plt.ylabel('Average Fluorescence Signal')
		variation = (np.max(avgUnfold) - np.min(avgUnfold))/np.min(avgUnfold) * 100
		# Typically keep if the percent variation < 10%
		if variation < 10:
			plt.title(row[1].Time + ': % variation = ' +
					  str(np.round(variation, decimals=2)) + ', within 10% (yes)')
		else:
			plt.title(row[1].Time + ': % variation = ' +
					  str(np.round(variation, decimals=2)) + ', NOT within 10% (no)')
		plt.savefig(os.getcwd() + '/app/static/unfoldPlot_' + str(i) + '.png', bbox_inches='tight')
		plt.close()

	sns.set(style="white")

	# Compute the correlation matrix
	corr = unfoldSignal.corr()

	# Plot heatmap of the correlation between the different
	# unfolding controls

	s = [i for i in range(len(unfoldRows))]
	sns.heatmap(corr.abs(), annot=True, xticklabels=s,
				yticklabels=s, square=True, fmt = '.2f',
				vmin=0, vmax=1)
	plt.yticks(rotation=0)
	plt.xlabel('Unfolding Control')
	plt.ylabel('Unfolding Control')
	plt.title('Cross-Correlation Between Controls')
	plt.savefig(os.getcwd() + '/app/static/unfoldCorrelation.png', bbox_inches='tight')
	plt.close()

	for i, column in enumerate(unfoldSignal.columns):
		s = unfoldSignal[column]
		maxVal = np.max(s)
		minVal = np.min(s)
		slope = 1 / (maxVal - minVal)
		unfoldSignal['norm' + str(i)] = slope * (s - minVal)
		plt.plot(guanConc, unfoldSignal['norm' + str(i)], linewidth = 1,
				 label = str(i))
	plt.legend()
	# Plot the normed fluorescence for each unfloding control with
	# the average
	cols = [col for col in unfoldSignal.columns if 'norm' in col]
	normData = unfoldSignal[cols].transpose().as_matrix()
	sns.tsplot(data=normData, time=guanConc)
	plt.xlabel('Guanidine Concentration (M)')
	plt.ylabel('Normalized Fluorescence Signal')
	plt.savefig(os.getcwd() + '/app/static/allUnfolds.png', bbox_inches='tight')
	plt.close()
		
