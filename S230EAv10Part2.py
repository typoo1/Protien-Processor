import matplotlib.pyplot as plt
import numpy as np
import sys
import csv
import pandas as pd
from lmfit import minimize, Parameters, Parameter, report_fit
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.backends.backend_pdf
import os

# Defines the fit for the data.
def fcn2min(params, guan, data):
    RT = .008314*303
    D50 = params['D50'].value
    m = params['m'].value
    slope_F = params['slope_F'].value
    intercept_F = params['intercept_F'].value
    slope_U = params['slope_U'].value
    intercept_U = params['intercept_U'].value
    FL = slope_F*guan + intercept_F
    UL = slope_U*guan + intercept_U
    #print([D50,m,slope_F,intercept_F,slope_U,intercept_U])
    s = np.exp(m*(guan - D50)/RT)
    model = (FL + UL*s)/(1 + s)
    return model - data

def run(flaskArray):
    # THIS IS WHERE PART 2 WILL START

    # String that defines if a row is a control
    controlString = 'Negative control'

    # Not sure if these are implemented correctly
    # If the rmsError of the fit is greater than rmsErrorThreshold it is flagged in the rmsFlag
    # column of the fit file
    # The code may provide an error if the rmsErrorThreshold and fitRemoveResidualThreshold
    # do not have similar sizes. If you increase the rmsErrorThreshold it will make it more
    # likely that the data is marked as GOOD and the data will be refitted.
    rmsErrorThreshold = .3

    # If the rmsError for a plot < rmsErrorThreshold then consider refitting by
    # removing points whose abs(residual) < fitRemoveResidualThreshold
    # If you increase the fitRemoveResidualThreshold it will possibly increase the number of
    # points that are used in the refit.
    fitRemoveResidualThreshold = .02

    # The following will need to be inputted from Part 1. Use test as a
    # dummy for the current work.

    inFile = os.getcwd() + '/app/uploads/result'
    fitFile = inFile + '_fit.csv'
    pdfFile = inFile + '_plot.pdf'
    enterSlope_U = 0
    enterSlope_F = 0

    # includeFold is a list that contains indices of which unfolding controls were chosen
    # from Part 1
    # Use the following to test
    #flaskArray = [True, True, False, True]
    includeUnfold = [i for i,s in enumerate(flaskArray) if s]

    np.where(flaskArray)
    print(flaskArray)

    # Need to average the unfolding controls. This file should be created after Part 1 runs.
    data = pd.read_csv(os.getcwd() + '/app/uploads/part1DataFinal.csv', index_col=0)

    # Extract the guanConc from the data
    guanConc = np.array(data.ix['guanadine',[col for col in data.columns if 'avg' in col]],
                        dtype=float)

    # Number of averages taken
    maxAverage = int(data.loc['average',:].max()) + 1

    # Determine the rows that have the unfolding controls
    samples = data.ix[:-4,:]
    allUnfoldControl = samples[samples['Time'].str.contains(controlString)]
    allUnfoldControl  = allUnfoldControl.reset_index()

    # Select the good unfolding control rows
    s = allUnfoldControl.ix[includeUnfold, :]

    # Use the columns that have avgn in them
    unfoldRows = s[['Time'] + [col for col in s.columns if 'avg' in col]]
    avgUnfoldGood = np.array(unfoldRows.mean())

    # Determine the minimum/maximum fluorescence and range of fluorescence
    # for each sample
    cols = data.columns[data.columns.str.contains('min ')]
    data['maximum fluor'] = data[cols].max(axis=1)
    data['minimum fluor'] = data[cols].min(axis=1)
    data['range fluor'] = data['maximum fluor'] - data['minimum fluor']

    # this for loop iterates through all lines from 9 to 63. under this for loop we
    #normalize the sample values, plot them vs. guanConc, apply the formula we defined
    #above, create legends, create lables, fit the plot using lmfit, save it to pdf
    #and create a csv files with all the data.

    # Only consider the rows with sample data and averaged columns
    s = data.dropna(subset=['Time'])
    s = s[~s['Time'].str.contains(controlString)]
    sampleRows = s[['Time'] + [col for col in s.columns if 'avg' in col]]

    # Holds the normalized data that is fitted
    index = sampleRows.index
    columns = ['norm' + str(i) for i in range(maxAverage)]
    normalized = pd.DataFrame(index = index, columns = columns, dtype = float)

    # Holds the fitted parameters
    columns = ['count', 'D50', 'm', 'slope_F', 'intercept_F', 'slope_U', 'intercept_U']
    fitParams = pd.DataFrame(index = index, columns = columns, dtype = float)
    columns = ['RF_count', 'RF_D50', 'RF_m', 'RF_slope_F', 'RF_intercept_F', 'RF_slope_U',
               'RF_intercept_U']
    refitParams = pd.DataFrame(index = index, columns = columns, dtype = float)

    # Holds the fitted curve
    columns = ['fit' + str(i) for i in range(maxAverage)]
    fitCurve = pd.DataFrame(index = index, columns = columns, dtype = float)
    columns = ['RF_fit' + str(i) for i in range(maxAverage)]
    refitCurve = pd.DataFrame(index = index, columns = columns, dtype = float)

    # Holds all points for refitted curve - different from refitCurve in that
    # it will store all points - some not used in refit.
    columns = ['RFAll_fit' + str(i) for i in range(maxAverage)]
    refitAllCurve = pd.DataFrame(index = index, columns = columns, dtype = float)
    columns = ['RFAll_residual' + str(i) for i in range(maxAverage)]
    refitAllResidual = pd.DataFrame(index = index, columns = columns, dtype = float)

    # Holds the difference between the fitted and normalized curve
    columns = ['residual' + str(i) for i in range(maxAverage)]
    fitResidual = pd.DataFrame(index = index, columns = columns, dtype = float)
    columns = ['RF_residual' + str(i) for i in range(maxAverage)]
    refitResidual = pd.DataFrame(index = index, columns = columns, dtype = float)

    # Holds the rms error for the data and the refitted data
    columns = ['rmsError', 'refitrmsError', 'rmsErrorFlag', 'refitrmsErrorFlag']
    rmsError = pd.DataFrame(index = index, columns = columns, dtype = float)

    # Fits each of the samples and stores the data in fitParams and refitParams

    for row in sampleRows.iterrows():
        index = row[0]
        scaledSample = np.array(row[1][1:], dtype=float) / avgUnfoldGood
        maxVal = np.max(scaledSample)
        minVal = np.min(scaledSample)
        slope = 1 / (maxVal - minVal)
        normalizedSample = slope * (scaledSample - minVal)
        normalized.loc[index,:] = normalizedSample
        
        # Sets up the parameters fitted by minimize beloq
        params = Parameters()
        params.add('D50', value = 2, min = 0, max = 5)
        params.add('m', value = 1)
        if enterSlope_F == None:
            params.add('slope_F', value = 0)
        else:
            params.add('slope_F', value = enterSlope_F, vary = False)
        params.add('intercept_F', value = 0.8)
        if enterSlope_U == None:
            params.add('slope_U', value = 0)
        else:
            params.add('slope_U', value = enterSlope_U, vary = False)
        params.add('intercept_U', value = 0.1)
        # performs the fit
        
        result = minimize(fcn2min, params, args=(guanConc, normalizedSample))

        fitParams.loc[index,'count'] = len(scaledSample)
        fitParams.loc[index,'D50'] = result.params['D50'].value
        fitParams.loc[index,'m'] = result.params['m'].value
        fitParams.loc[index,'slope_F'] = result.params['slope_F'].value
        fitParams.loc[index,'intercept_F'] = result.params['intercept_F'].value
        fitParams.loc[index,'slope_U'] = result.params['slope_U'].value
        fitParams.loc[index,'intercept_U'] = result.params['intercept_U'].value

        # fitted curve is the data plus the residual determined by minimize
        fitCurve.loc[index,:] = normalizedSample + result.residual
        fitResidual.loc[index,:] = result.residual

        # Store the rms error
        error = np.sum(result.residual**2)
        rmsError.loc[index, 'rmsError'] = error
        if error < rmsErrorThreshold:
            rmsError.loc[index, 'rmsErrorFlag'] = 'Good'
        else:
            rmsError.loc[index, 'rmsErrorFlag'] = 'Bad'
        # Refit the data by removing points that have rmsResidual above the
        # rmsErrorresidualThreshold
        # Only refit the good fits.
        # We may need to renormalize the data after throwing out some of the points
        # The rmsErrorThreshold and the fitRemoveResidualThreshold need to be chosen
        # together since making the rmsErrorThreshold big (allows the refitting) and
        # the fitRemoveResidualThreshold smaller may only leavce a few points left
        # for fitting. The fitting needs enough points to allow fitting.
        # 2/21/2016
        if error < rmsErrorThreshold:
            keepPoints = abs(result.residual) < fitRemoveResidualThreshold
            refitGuanConc = guanConc[keepPoints]
            refitNormalizedSample = normalizedSample[keepPoints]
            
            result = minimize(fcn2min, params, args=(refitGuanConc, refitNormalizedSample))
            refitParams.loc[index,'RF_count'] = keepPoints.sum()
            refitParams.loc[index,'RF_D50'] = result.params['D50'].value
            refitParams.loc[index,'RF_m'] = result.params['m'].value
            refitParams.loc[index,'RF_slope_F'] = result.params['slope_F'].value
            refitParams.loc[index,'RF_intercept_F'] = result.params['intercept_F'].value
            refitParams.loc[index,'RF_slope_U'] = result.params['slope_U'].value
            refitParams.loc[index,'RF_intercept_U'] = result.params['intercept_U'].value

            # Change this to store all of the points, not just the refitted points
            RT = .008314*303
            D50 = result.params['D50'].value
            m = result.params['m'].value
            slope_F = result.params['slope_F'].value
            intercept_F = result.params['intercept_F'].value
            slope_U = result.params['slope_U'].value
            intercept_U = result.params['intercept_U'].value
            FL = slope_F*guanConc + intercept_F
            UL = slope_U*guanConc + intercept_U
            s = np.exp(m*(guanConc - D50)/RT)
            refitAllCurve.loc[index, :] = (FL + UL*s)/(1 + s)
            refitAllResidual.loc[index,:] = np.array(refitAllCurve.loc[index, :]) - normalizedSample

            # Only includes the points that are used in refitting
            refitCurve.loc[index, keepPoints] = refitNormalizedSample + result.residual
            refitResidual.loc[index, keepPoints] = result.residual

            # Save the rms error
            error = np.sum(refitAllResidual.loc[index,:]**2)
            rmsError.loc[index, 'refitrmsError'] = error
            if error < rmsErrorThreshold:
                rmsError.loc[index, 'refitrmsErrorFlag'] = 'Good'
            else:
                rmsError.loc[index, 'refitrmsErrorFlag'] = 'Bad'
            
    #include for print out of fits
        #report_fit(result.params)

    # Print all of the data to a CSV file including input data, normalized data,
    # fit parameters and the fitted curve.
    data = data.merge(normalized, how = 'left', left_index = True, right_index = True)
    data = data.merge(fitParams, how = 'left', left_index = True, right_index = True)
    data = data.merge(fitCurve, how = 'left', left_index = True, right_index = True)
    data = data.merge(fitResidual, how = 'left', left_index = True, right_index = True)
    data = data.merge(rmsError, how = 'left', left_index = True, right_index = True)
    data = data.merge(refitParams, how = 'left', left_index = True, right_index = True)
    data = data.merge(refitCurve, how = 'left', left_index = True, right_index = True)
    data = data.merge(refitResidual, how = 'left', left_index = True, right_index = True)
    data = data.merge(refitAllCurve, how = 'left', left_index = True, right_index = True)
    data = data.merge(refitAllResidual, how = 'left', left_index = True, right_index = True)

    # Plot all of the protein sample data to one PDF file
    count = 1
    figCount = 1
    fig = plt.figure()
    pdf = matplotlib.backends.backend_pdf.PdfPages(pdfFile)

    # Find the rows that contains sample data
    s = data.dropna(subset=['Time'])
    s = s[~s['Time'].str.contains(controlString)]
    print(s)
    dataPlot = s.convert_objects(convert_numeric=True) #pd.to_numeric(s, errors='ignore')

    for index, row in enumerate(dataPlot.iterrows()):
        info = row[1]
        plt.rc('xtick', labelsize=5)
        plt.rc('ytick', labelsize=5)
        ax = fig.add_subplot(3,3,count)
        #normalizedCurve = info.filter(regex = 'norm')
        normalizedCurve = info[[indx for indx in info.index if 'norm' in indx]]
        ax.scatter(guanConc, normalizedCurve, c = 'r', marker = 'o')
        #fitCurve = info.filter(regex = '^fit')
        fitCurve = info[[indx for indx in info.index if indx[:3] == 'fit']]
        # ax.plot(guanConc, fitCurve)
        # Plot the fitted curve after removing high residual points
        #refitCurve = info.filter(regex = '^RF_fit')
        refitCurve = info[[indx for indx in info.index if indx[:6] == 'RF_fit']]
        refitAllCurve = info[[indx for indx in info.index if indx[:9] == 'RFAll_fit']]
        # Don't plot a curve if refitting is not used
        if refitCurve.isnull().sum() < len(refitCurve):
            # Note that the points plotted are the refitted points
            ax.plot(guanConc, refitAllCurve, c = 'b')
            ax.scatter(guanConc, refitCurve, c = 'g', marker='o')
            # Changed to include a curve constructed from the formula and not stored points
        count += 1
        plt.ylim([-0.1,1.1])
        plt.xlim([0,5])
        ax.set_xlabel('Guanidine Concentration (M)').set_fontsize('6')
        ax.set_ylabel('Normalized Fluorescence Signal').set_fontsize('6')
        
        ax.set_title(info['Time'] + ' + ' + info['rmsErrorFlag'],
                     fontsize=12).set_color('m')

    # Plot the FL = folded line and UL = unfolded line
        FL = info['slope_F']*guanConc + info['intercept_F']
        UL = info['slope_U']*guanConc + info['intercept_U']
        FLlegend, = ax.plot(guanConc, FL, '--', label='FL')
        ULlegend, = ax.plot(guanConc, UL, '--', label='UL')
        plt.text(2, .93, 'Min Fluorescence = ' + str(int(info['minimum fluor'])),
                 fontsize=6)
        plt.text(2, .83, 'Max Fluorescence = ' + str(int(info['maximum fluor'])),
                 fontsize=6)
        plt.text(2, .73, 'Range Fluorescence = ' + str(int(info['range fluor'])),
                 fontsize=6)
        plt.text(2.85, .6, 'D50 = ' + str(round(info['D50'],2)) + ', ' +
                 str(round(info['RF_D50'],2)),fontsize=6)
        plt.text(2.85, .5, 'MV = ' + str(round(info['m'],2)) + ', ' +
                 str(round(info['RF_m'],2)),fontsize=6)
        plt.text(2.85, .4, 'slope_F = ' + str(round(info['slope_F'],2)) + ', ' +
                 str(round(info['RF_slope_F'],2)),fontsize=6)
        plt.text(2.85, .3, 'slope_U = ' + str(round(info['slope_U'],2))  + ', ' +
                 str(round(info['RF_slope_U'],2)),fontsize=6)
        refitUsed = len(refitCurve) - refitCurve.isnull().sum()
        plt.text(2.85, .2, 'refit ' + str(refitUsed) + ' of ' +
                 str(len(refitCurve)), fontsize=6)
        
    # Plot the average of FL and UL
        avgFLULLegend, = ax.plot(guanConc, (FL + UL)/2, 'm--', label='(FLrmsErrorThreshold + UL)/2')
        
    #  Intersection of the line y = D50 and (FL + UL) / 2 should be on the fit curve
        #D50legend, = ax.plot([info['D50'], info['D50']], [0, 1], 'k--', label='D50value')
        #plt.legend(handles=[FLlegend, ULlegend,avgFLULLegend, D50legend,])
        #ax.legend(loc='upper right', labelspacing=1, fontsize=3)
        plt.tight_layout(pad=0)
        if index == len(dataPlot)-1:
            pdf.savefig( fig )
            fig = plt.figure()
        elif count % 9 == 1:
            pdf.savefig( fig )
            count = 1
            figCount += 1
            fig = plt.figure()


    ### Plots the residual at each point after the refit
    ##count = 1
    ##for index, row in enumerate(dataPlot.iterrows()):
    ##    info = row[1]
    ##    plt.rc('xtick', labelsize=5)
    ##    plt.rc('ytick', labelsize=5)
    ##    ax = fig.add_subplot(3,3,count)
    ##    normalizedCurve = info[[indx for indx in info.index if 'RFAll_residual' in indx]]
    ##    ax.scatter(guanConc, normalizedCurve, c = 'r', marker = 'o')
    ##    count += 1
    ##    plt.xlim([0,5])
    ##    ax.set_xlabel('Guanidine Concentration (M)').set_fontsize('6')
    ##    ax.set_ylabel('Residual Signal').set_fontsize('6')
    ##    
    ##    ax.set_title('Residual for ' + info['Time'], fontsize=12).set_color('m')
    ##    
    ##    plt.tight_layout(pad=0)
    ##    if index == len(dataPlot)-1:
    ##        pdf.savefig( fig )
    ##    elif count % 9 == 1:
    ##        pdf.savefig( fig )
    ##        fig = plt.figure()
    ##        count = 1
        
    # Plot the residuals after refitting

    # Plot the residual plots for each sample
    # fitted and refitted - actual

    pdf.close()

    # Print the data to the file
    data.ix[0,'rmsErrorThreshold'] = rmsErrorThreshold
    data.ix[0,'fitRemoveResidualThreshold'] = fitRemoveResidualThreshold
    data.ix[0,'enterSlope_U'] = enterSlope_U
    data.ix[0,'enterSlope_F'] = enterSlope_F
    data.ix[0,'includeUnfold'] = '-'.join(str(x) for x in includeUnfold)
    data.to_csv(fitFile)
