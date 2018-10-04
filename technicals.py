## This code computes a good number of technical indicators.
## Unless otherwise stated, the source for formulas is FMlabs.com.
## Author: Miguel Opeña
## Version: 1.1.1

import logging
import math
import numpy as np
import os
import pandas as pd

import download
import plotter

LOGDIR = "/Users/openamiguel/Desktop/LOG"
# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Set file path for logger
handler = logging.FileHandler('{}/equitysim.log'.format(LOGDIR))
handler.setLevel(logging.DEBUG)
# Format the logger
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
# Add the new format
logger.addHandler(handler)
# Format the console logger
consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(logging.INFO)
consoleHandler.setFormatter(formatter)
# Add the new format to the logger file
logger.addHandler(consoleHandler)

logger.info("----------INITIALIZING NEW RUN OF %s----------", os.path.basename(__file__))

def test_technical():
	""" Hardcoded test of technical indicator """
	symbol = "AAPL"
	folderpath = "/Users/openamiguel/Documents/EQUITIES/stockDaily"
	start_date = "2015-03-01"
	end_date = "2018-06-01"
	tick_data = download.load_single_drive(symbol, folderpath=folderpath)
	tick_data = tick_data[start_date:end_date]
	sine, leadsine = mesa_sine_wave(tick_data, num_periods=30)
	price_with_trends = pd.concat([tick_data.close, sine], axis=1)
	price_with_trends.columns = ['price', 'MESA30']
	plotter.price_plot(price_with_trends, symbol, subplot=[False,True,True], returns=[False,False,False], folderpath=folderpath, showPlot=True)

def accum_swing(tick_data, limit):
	""" Plots the cumulative sum (running total) of swing index, aka accumulation swing index.
		Inputs: exact same as swing_index function
		Outputs: running total of swing index
	"""
	si = swing_index(tick_data, limit=limit)
	return si.cumsum()

def ad_line(tick_data):
	""" Plots the accumulation-distribution line ("AD" or "AD line") as a measure of volume
		Inputs: dataframe with closing price, low price, high price, and volume over given timespan
		Outputs: dataframe with AD over time
	"""
	# Assumes that input is dataframe
	# Gets the close-location value index
	clv = ((tick_data.close - tick_data.low) - (tick_data.high - tick_data.close)) / (tick_data.high - tick_data.low)
	clv = clv * tick_data.volume
	ad_series = clv.cumsum()
	ad = pd.DataFrame(ad_series, columns=['ad_line'])
	return ad

def adx(tick_data, num_periods):
	""" Computes the average directional (movement) index (ADX), based on DX.
		Inputs: dataframe of data fed into the DX function
		Outputs: dataframe of ADX over time
	"""
	dx = directional_movt_index(tick_data, num_periods)
	adx = pd.DataFrame(index=dx.index, columns=['ADX'])
	# Sets the seed value to be zero
	adx.ADX[adx.index[0]] = 0
	# Walks through the data and computes ADX
	for i in range(1, len(adx.index)):
		this_date = adx.index[i]
		last_date = adx.index[i-1]
		adx.ADX[this_date] = (adx.ADX[last_date] * (num_periods - 1) + dx.DX[this_date]) / num_periods
	return adx

def adxr(tick_data, num_periods):
	""" Computes the ADX rating of a stock over time, based on ADX.
		Inputs: dataframe of data fed into the ADX function
		Outputs: dataframe of ADXR over time
	"""
	adx_df = adx(tick_data, num_periods)
	adxr = pd.DataFrame(index=adx_df.index, columns=['ADXR'])
	# Walks through the data and computes ADXR
	for i in range(0, len(tick_data.index) - num_periods):
		start_date = adx_df.index[i]
		end_date = adx_df.index[i+1]
		adxr.ADXR[end_date] = 0.5 * (adx_df.ADX[start_date] + adx_df.ADX[end_date])
	return adxr

def aroon(tick_data, num_periods=25):
	""" Computes the Aroon indicator of an asset over time. 
		This code assumes that number of periods refers to the number of periods for which data is provided, not the number of actual time periods.
		Inputs: dataframe with opening price, closing price, high price, low price over given timespan;
			also includes number of periods to perform calculation
		Outputs: dataframes with AroonUp and AroonDown over time
	"""
	# Assume that input is dataframe
	aroon_up = pd.DataFrame(index=tick_data.index, columns=['aroon_up'])
	aroon_down = pd.DataFrame(index=tick_data.index, columns=['aroon_down'])
	# Iterates through all datewindows
	for i in range(0, len(tick_data.index) - num_periods):
		# Gets the proper tick date window
		start_date = tick_data.index[i]
		end_date = tick_data.index[i + num_periods]
		tick_data_window = tick_data[start_date:end_date]
		# Gets the recent maximum and minimum relative to the date window
		max_index = tick_data_window.close.idxmax()
		min_index = tick_data_window.close.idxmin()
		# Gets number of periods since previous extremum
		max_dist = len(tick_data[max_index:end_date]) - 1
		min_dist = len(tick_data[min_index:end_date]) - 1
		# Populates the output dataframes
		aroon_up.aroon_up[end_date] = 100 * (num_periods - max_dist) / num_periods
		aroon_down.aroon_down[end_date] = 100 * (num_periods - min_dist) / num_periods
	return aroon_up, aroon_down

def aroon_oscillator(tick_data, num_periods=25):
	""" Computes the Aroon oscillator of an asset over time, which is simply AroonUp minus AroonDown
		Inputs: dataframe with opening price, closing price, high price, low price over given timespan;
			also includes number of periods to perform calculation
		Outputs: dataframe with Aroon oscillator over time
	"""
	# Gets AroonUp and AroonDown from the aroon function
	aroon_up, aroon_down = aroon(tick_data, num_periods=num_periods)
	aroon_up.columns = ['aroon']
	aroon_down.columns = ['aroon']
	# Initializes and populates output
	aroon_osc = pd.DataFrame(index=tick_data.index, columns=['aroon_oscillator'])
	aroon_osc = aroon_up.subtract(aroon_down,axis=1)
	# Returns Aroon oscillator
	return aroon_osc

def average_price(tick_data):
	""" Computes the average price of an asset over time. 
		Inputs: dataframe with opening price, closing price, high price, low price over given timespan
		Outputs: average price over given timespan
	"""
	# Assume that input is dataframe
	avg_price = pd.DataFrame(index=tick_data.index, columns=['average_price'])
	# Adds up the prices into avg_price
	avg_price['average_price'] = tick_data.open + tick_data.close + tick_data.high + tick_data.low
	# Divides by four
	avg_price = avg_price.divide(4)
	return avg_price

def average_true_range(tick_data, num_periods=14):
	""" Uses the true range to compute the average true range (ATR) of an asset over time.
		Inputs: data on high, low, and close of asset over given timespan
		Outputs: ATR indicator
	"""
	# Sets up dataframe for true range
	tr = pd.DataFrame(index=tick_data.index, columns=["true_range"])
	for i in range(1, len(tick_data.index)):
		# Gets the date window (not dependent on num_periods)
		now_date = tick_data.index[i]
		last_date = tick_data.index[i-1]
		# Adds this true range to the dataframe
		tr.true_range[now_date] = max(tick_data.high[now_date], tick_data.close[last_date]) - max(tick_data.low[now_date], tick_data.close[last_date])
	# Sets up dataframe for average true range
	atr = pd.DataFrame(index=tick_data.index, columns=["ATR"])
	# The seed value is NOT zero
	atr.ATR[atr.index[0]] = tr.true_range.mean()
	for i in range(1, len(tr.index)):
		# Gets the date window (not dependent on num_periods)
		now_date = tr.index[i]
		last_date = tr.index[i-1]
		# Adds this true range to the dataframe
		atr.ATR[now_date] = (atr.ATR[last_date] * (num_periods - 1) + tr.true_range[now_date]) / num_periods
	# Returns ATR
	return atr

def bollinger(tick_data, num_periods=20, num_deviations=2):
	""" Computes the Bollinger bands and width of an asset over time. 
		Inputs: dataframe with closing price, high price, low price over given timespan
		Outputs: Bollinger bands and width over given timespan
	"""
	# Calculates typical price and standard deviation thereof
	typ_price = typical_price(tick_data)
	stdev = typ_price['typical_price'].std()
	# Calculates the three Bollinger bands
	midband = simple_moving_average(typ_price, num_periods=num_periods)
	lowband = midband - num_deviations * stdev
	hiband = midband + num_deviations * stdev
	# Calculates the width of said bands
	width = 2 * num_deviations * stdev
	# Returns all the needed information
	return lowband, midband, hiband, width

def chande_momentum_oscillator(price, num_periods):
	""" Computes the Chande momentum oscillator of a price input over time.
		Inputs: price of asset, number of periods in CMO
		Outputs: CMO of price
	"""
	up_df = pd.DataFrame(index=price.index, columns=['upp'])
	dn_df = pd.DataFrame(index=price.index, columns=['down']) 
	cmo = pd.DataFrame(index=price.index, columns=['CMO'])
	# Walks through the dates and gets up/down indices at each interval
	for i in range(0, len(price.index) - 1):
		# Gets the proper tick date window
		start_date = price.index[i]
		end_date = price.index[i+1]
		# Gets some more variables
		up = 0
		dn = 0
		if price[end_date] > price[start_date]: 
			up = price[end_date] - price[start_date]
		else: 
			dn = price[start_date] - price[end_date]
		# Saves up and down accordingly
		up_df.upp[end_date] = up
		dn_df.down[end_date] = dn
	# Walks through up and down to get cmo
	for i in range(0, len(price.index) - num_periods):
		start_date = price.index[i]
		end_date = price.index[i + num_periods]
		# Walks backward by num_periods to get sum
		ups = up_df.upp[start_date:end_date].sum()
		downs = dn_df.down[start_date:end_date].sum()
		# Saves CMO
		cmo.CMO[end_date] = 100 * (ups - downs) / (ups + downs)
	return cmo

def chaikin(tick_data, num_periods):
	""" Computes the Chaikin money flow (volume indicator) of a stock over time.
		Inputs: dataframe with closing price, low price, high price, volume; number of periods
		Outputs: Chaikin money flow over given timespan
	"""
	# Builds the closing location value index and multiplies by volume
	clv = ((tick_data.close - tick_data.low) - (tick_data.high - tick_data.close)) / (tick_data.high - tick_data.low)
	clv_vol = clv * tick_data.volume
	chk = pd.DataFrame(index=tick_data.index, columns=['chaikin'])
	# Iterates through the prices using given window
	for i in range(0, len(tick_data.index) - num_periods):
		start_date = tick_data.index[i]
		end_date = tick_data.index[i + num_periods]
		chk.chaikin[end_date] = clv_vol[start_date:end_date].sum() / tick_data.volume[start_date:end_date].sum()
	return chk

def chaikin_ad_osc(tick_data):
	""" Computes the Chaikin A/D oscillator over given timespan.
		Very related to the Chaikin money flow, in that CLV is used.
		Inputs: dataframe with closing price, low price, high price, volume
		Outputs: Chaikin A/D oscillator over given timespan
	"""
	ad = ad_line(tick_data)
	component1 = exponential_moving_average(ad.ad_line, num_periods=3)
	component2 = exponential_moving_average(ad.ad_line, num_periods=10)
	return component1 - component2

def chaikin_volatility(tick_data, num_periods):
	""" Computes the Chaikin volatility over given timespan.
		Related in principle to the other Chaikin indicators.
		Inputs: dataframe with low and high price; number of periods
		Outputs: Chaikin volatility over given timespan
	"""
	emahl = exponential_moving_average(tick_data.high - tick_data.low, num_periods=num_periods)
	cv = pd.DataFrame(index=tick_data.index, columns=['ChkVol'])
	for i in range(0, len(tick_data.index) - num_periods):
		start_date = tick_data.index[i]
		end_date = tick_data.index[i + num_periods]
		cv.ChkVol[end_date] = 0.01 * (emahl[end_date] - emahl[start_date]) / emahl[start_date]
	return cv

def dema(input_values, num_periods=30):
	""" Computes the so-called double exponential moving average (DEMA) of a time series over certain timespan.
		Inputs: input values, number of periods in DEMA
		Outputs: DEMA over given timespan
	"""
	# If input is Series, output is Dataframe
	if isinstance(input_values, pd.Series):
		ema = exponential_moving_average(input_values, num_periods=num_periods)
		ema2 = exponential_moving_average(ema, num_periods=num_periods)
		dema = pd.DataFrame(index=input_values.index, columns=['DEMA'])
		# This is the formula for DEMA
		for i in range(0, len(input_values.index)):
			dema.DEMA[i] = 2 * ema[i] - ema2[i]
		return dema
	# If input is list, output is list
	elif isinstance(input_values, list):
		ema = np.array(exponential_moving_average(input_values, num_periods=num_periods))
		ema2 = np.array(exponential_moving_average(exponential_moving_average(input_values, num_periods=num_periods), num_periods=num_periods))
		# This is the formula for DEMA
		dema = 2 * ema - ema2
		return dema.tolist()
	else:
		raise ValueError("Unsupported data type given as input to dema in technicals_calculator.py")
		return None

def detrended_price_osc(price, num_periods):
	""" Computes the detrended price oscillator (DPO). 
		Related to detrended price in principle only; actual method is dissimilar. 
		Inputs: price Series over time; number of periods in DPO
		Outputs: DPO over given timespan
	"""
	dpo = pd.DataFrame(index=price.index, columns=['DPO'])
	for i in range(1, len(dpo.index) - num_periods):
		start_date = dpo.index[i]
		last_price = price[dpo.index[i + num_periods - 1]]
		end_date = dpo.index[i + num_periods]
		# Subtracts the previous price in the window by the moving avg of price
		sma = simple_moving_average(price[start_date:end_date], 
						num_periods=num_periods)
		dpo.DPO[end_date] = last_price - sma[sma.index[-1]] / num_periods
	return dpo

def directional_index(tick_data, num_periods):
	""" Computes the directional indices (+DI and -DI).
		Inputs: close, high, and low data on asset; number of periods
		Outputs: +DI and -DI on asset over given timespan
	"""
	# Initializes starting variables
	di_positive = pd.DataFrame(index=tick_data.index, columns=['DI_PLUS'])
	di_negative = pd.DataFrame(index=tick_data.index, columns=['DI_MINUS'])
	# Initializes running sums
	plus_dm_sum = 0
	minus_dm_sum = 0
	tr_sum = 0
	# Walks through the price dataframe
	for i in range(1, len(tick_data.index)):
		now_date = tick_data.index[i]
		last_date = tick_data.index[i-1]
		# Gets starting variables
		delta_high = tick_data.high[last_date] - tick_data.high[now_date]
		delta_low = tick_data.low[now_date] - tick_data.low[last_date]
		plus_dm = delta_high if delta_high > delta_low else 0
		minus_dm = delta_low if delta_high < delta_low else 0
		tr = max(tick_data.high[now_date], tick_data.close[last_date]) - max(tick_data.low[now_date], tick_data.close[last_date])
		# Updates running sums
		plus_dm_sum = plus_dm_sum - (plus_dm_sum / num_periods) + plus_dm
		minus_dm_sum = minus_dm_sum - (minus_dm_sum / num_periods) + minus_dm
		tr_sum = tr_sum - (tr_sum / num_periods) + tr
		# Updates output dataframe
		di_positive.DI_PLUS[now_date] = 100 * plus_dm_sum / tr_sum
		di_negative.DI_MINUS[now_date] = 100 * minus_dm_sum / tr_sum
	# Return output
	return di_positive, di_negative

def directional_movt_index(tick_data, num_periods):
	""" Computes the directional movement index (DX), which is derived directly from +DI and -DI.
		Inputs: close, high, and low data on asset; number of periods
		Outputs: DX on asset over given timespan
	"""
	di_positive, di_negative = directional_index(tick_data, num_periods)
	di_positive.columns = ['DX']
	di_negative.columns = ['DX']
	return (di_positive - di_negative) / (di_positive + di_negative)

def dynamic_momentum_index(price):
	""" Computes the dynamic momentum index, the DSI, of a price over time.
		Inputs: price Series over time (typically close)
		Outputs: DSI dataframe over time
	"""
	stdev_5 = price.rolling(5).std()
	numerator = 14 * simple_moving_average(stdev_5, num_periods=10)
	return numerator  / stdev_5

def ease_of_movt(tick_data, constant=1000000000):
	""" Computes the ease of movement indicator (EMV). The constant is set to 1e+9 for plotting purposes. 
		Inputs: dataframe with high price, low price, and volume over given timespan; constant in the box ratio calculation
		Outputs: EMV over given timespan
	"""
	# Initializes empty dataframe to hold EMV values
	emv = pd.DataFrame(index=tick_data.index, columns=['EMV'])
	for i in range(1, len(tick_data.index)):
		# Calculates the midpoint move and box ratio at current time
		midpoint_move = (tick_data.high[i] - tick_data.low[i] - (tick_data.high[i-1] - tick_data.low[i-1])) / 2
		box_ratio = (tick_data.volume[i] / constant) / (tick_data.high[i] - tick_data.low[i])
		# Calculates EMV from the previous variables
		emv.EMV[i] = midpoint_move / box_ratio
	return emv

def exponential_moving_average(input_values, num_periods=30):
	""" Computes the exponential moving average (EMA) of a time series over certain timespan.
		Inputs: input values, number of periods in EMA
		Outputs: EMA over given timespan
	"""
	K = 2 / (num_periods + 1)
	inputs_refined = input_values.fillna(0)
	# If input is Series, output is Dataframe
	if isinstance(inputs_refined, pd.Series):
		ema = pd.Series(inputs_refined[0], index=inputs_refined.index)
		# Iterates through and populates dataframe output
		for i in range(1, len(inputs_refined.index)):
			ema[i] = ema[i-1] + K * (inputs_refined[i] - ema[i-1])
		return ema
	# If input is list, output is list
	elif isinstance(inputs_refined, list):
		ema = [inputs_refined[0]]
		# Iterates through and populates list output
		for i in range(1, len(inputs_refined)):
			ema.append(ema[i-1] + K * (inputs_refined[i] - ema[i-1]))
		return ema
	else:
		raise ValueError("Unsupported data type given as input to exponential_moving_average in technicals_calculator.py")
		return None

def general_stochastic(price, num_periods):
	""" Computes the General Stochastic calculation of an asset over time. 
		Inputs: series with price over given timespan
		Outputs: General Stochastic over given timespan
	"""
	# Assume that input is dataframe
	general_stoch = pd.DataFrame(index=price.index, columns=['general_stochastic'])
	# Iterates through all datewindows
	for i in range(0, len(price.index) - num_periods):
		# Gets the proper tick date window
		start_date = price.index[i]
		end_date = price.index[i + num_periods]
		price_window = price[start_date:end_date]
		# Gets the recent maximum and minimum relative to the date window
		max_price = price_window.max()
		min_price = price_window.min()
		# Populates the output dataframes
		general_stoch.general_stochastic[end_date] = (price[end_date] - min_price) / (max_price - min_price)
	return general_stoch

def klinger_osc(tick_data):
	""" Compues the Klinger oscillator for asset data.
		Inputs: high price, low price, closing price, and volume
		Outputs: Klinger oscillator over given timespan
	"""
	# Computes the trend
	trend = (tick_data.high - tick_data.low - tick_data.close)
	trend = trend - trend.shift(-1)
	trend = trend.fillna(0)
	trend = trend.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
	# Computes the daily measurement
	dm = tick_data.high - tick_data.low
	# Computes the cumulative measurement
	trend_comp = trend == trend.shift(-1)
	cm = pd.DataFrame(index=tick_data.index)
	cm['CM'] = 0
	# Builds the cm dataframe
	for i in range(0, len(cm.index) - 1):
		start_date = cm.index[i]
		end_date = cm.index[i + 1]
		cm.CM[end_date] = cm.CM[start_date] + dm[end_date] if trend_comp[end_date] else dm[start_date] + dm[end_date]
	# Builds the volume force dataframe
	vf = tick_data.volume * abs(2 * dm / cm.CM - 2) * trend * 100
	vf = vf.replace([np.inf, -np.inf], np.nan)
	vf = vf.dropna()
	return exponential_moving_average(vf, num_periods=34) - exponential_moving_average(vf, num_periods=50)

def macd(price):
	""" Computes the MACD of a time series over certain timespan, which is essentially price oscillator for 26 and 12 periods, with EMA. 
		Inputs: price input
		Outputs: MACD over given timespan
	"""
	return price_oscillator(price, exponential_moving_average, num_periods_slow=26, num_periods_fast=12)

def market_fac_index(tick_data):
	""" Computes a very straightforward indicator.
		Best compared with volume trends, in order to derive conclusions.
		Inputs: dataframe with high, low, and volume
		Outputs: market facilitation index over given timespan
	"""
	return (tick_data.high - tick_data.low) / tick_data.volume

def mass_index(tick_data, num_periods):
	""" Computes the mass index, based on rolling sum of 9-period EMAs.
		Inputs: high and low price; number of periods
		Outputs: mass index over given timespan
	"""
	high_low = tick_data.high - tick_data.low
	mass = exponential_moving_average(high_low, num_periods=9) / exponential_moving_average(exponential_moving_average(high_low, num_periods=9), num_periods=9)
	return mass.rolling(num_periods).sum()

def median_price(tick_data):
	""" Computes the median price of an asset over time. 
		Inputs: dataframe with high and low price over given timespan
		Outputs: median price over given timespan
	"""
	# Assume that input is dataframe
	med_price = pd.DataFrame(index=tick_data.index, columns=['median_price'])
	# Adds up the prices into med_price
	med_price['median_price'] = tick_data.high + tick_data.low
	# Divides by two
	med_price = med_price.divide(2)
	return med_price

def mesa_sine_wave(tick_data, num_periods, threshold=0.001):
	""" Computes the MESA sine wave indicator.
		Part of the iffy sine-wave indicator family.
		Inputs: closing price; number of periods; threshold value
		Outputs: MESA sine wave over given timespan
	"""
	# Builds the real and imaginary part coefficients
	real_coeff = sum([math.sin(360 * j) / num_periods for j in range(0, num_periods + 1)])
	imag_coeff = sum([math.cos(360 * j) / num_periods for j in range(0, num_periods + 1)])
	# Builds the real and imaginary parts
	real_part = real_coeff * tick_data.close
	imag_part = imag_coeff * tick_data.close
	## Builds the DC phase
	# Compares imaginary part to the threshold value
	imag_comp = imag_part > threshold
	imag_comp_not = imag_part <= threshold
	dc_phase = pd.Series(index=tick_data.index)
	dc_phase[imag_comp] = real_part.divide(imag_part)
	dc_phase[imag_comp] = dc_phase[imag_comp].apply(lambda x: math.atan(x) + 90)
	dc_phase[imag_comp_not] = 90
	# Modifies the DC phase based on given requirements
	dc_phase[imag_part < 0] = dc_phase[imag_part < 0] + 180
	dc_phase[dc_phase > 270] = dc_phase[dc_phase > 270] - 360
	# Returns the actual indicator
	sine = dc_phase.apply(lambda x: math.sin(x))
	leadsine = dc_phase.apply(lambda x: math.sin(x) + 45)
	return sine, leadsine

def momentum(price):
	""" Computes the price momentum, to measure the acceleration of prices.
		Inputs: price
		Outputs: price momentum over given timespan
	"""
	return price - price.shift(-1)

def money_flow_index(tick_data, num_periods=None):
	"""
		Computes three closely-related metrics pertaining to price and volume
		Inputs: dataframe with high, low, closing, and volume
		Outputs: money flow, money flow index, and money (flow) ratio
	"""
	# Calculates the typical price
	tp = typical_price(tick_data)
	tp.columns = ['TP']
	# Money flow is simply typical price times tick data
	mf = tp.TP * tick_data.volume
	# Money flow index is a simple transformation of money flow
	mfi = 100 - (100 / (1 + mf))
	# Money flow ratio relates to upsums and downsums along positive/negative money flow
	tp_compare = tp - tp.shift(-1)
	# Retrieves dataframe of upsums and downsums from money flow
	pos_neg_mf = pd.DataFrame(index=tick_data.index)
	pos_neg_mf['pmf'] = mf[tp_compare.TP > 0]
	pos_neg_mf['nmf'] = mf[tp_compare.TP < 0]
	pos_neg_mf = pos_neg_mf.fillna(0)
	pos_neg_mf['pmf_cum'] = pos_neg_mf['pmf'].cumsum()
	pos_neg_mf['nmf_cum'] = pos_neg_mf['nmf'].cumsum()
	# Checks if num_periods given
	if not num_periods:
		return mf, mfi
	# Gets money flow ratio itself
	mr = pd.DataFrame(index=tick_data.index)
	mr['MonRatio'] = 0
	for i in range(0, len(tick_data.index) - num_periods):
		# Gets the proper tick date window
		start_date = tick_data.index[i]
		end_date = tick_data.index[i + num_periods]
		mr.MonRatio[end_date] = pos_neg_mf['pmf_cum'][start_date:end_date].sum() / pos_neg_mf['nmf_cum'][start_date:end_date].sum()
	return mf, mfi, mr

def negative_volume_index(tick_data):
	""" Computes a coefficient on close price, with increments only if volume is increasing
		Closely related to the PVI indicator
		Inputs: volume and closing price
		Outputs: NVI indicator over given timespan
	"""
	nvi = pd.DataFrame(index=tick_data.index, columns=['NVI'])
	nvi.NVI[nvi.index[0]] = 0
	for i in range(1, len(tick_data.index) - 1):
		start_date = nvi.index[i - 1]
		end_date = nvi.index[i]
		# Indicator increments if volume has increased
		increment = (tick_data.close[end_date] - tick_data.close[start_date]) / tick_data.close[start_date] if tick_data.volume[end_date] < tick_data.volume[start_date] else 0
		nvi.NVI[end_date] = nvi.NVI[start_date] + increment
	return nvi

def normalized_price(price, baseline):
	""" Computes the normalized price (aka performance indicator) against a baseline.
		Inputs: price series and baseline series
		Outputs: normalized price over given timespan
	"""
	norm_price = 100 * (price - baseline) / baseline
	norm_price.columns = ['normalized_price']
	return norm_price

def on_balance_volume(tick_data):
	""" Computes the on-balance volume (OBV) of an asset over time
		Inputs: volume series
		Outputs: OBV indicator
	"""
	obv = pd.DataFrame(0, index=tick_data.index, columns=['OBV'])
	for i in range(1, len(tick_data.index)):
		# Gets current window of time
		now_date = tick_data.index[i]
		last_date = tick_data.index[i-1]
		# Three conditions to consider when updating OBV
		if tick_data.close[now_date] > tick_data.close[last_date]:
			obv.OBV[now_date] = obv.OBV[last_date] + tick_data.volume[now_date]
		elif tick_data.close[now_date] > tick_data.close[last_date]:
			obv.OBV[now_date] = obv.OBV[last_date] - tick_data.volume[now_date]
		else:
			obv.OBV[now_date] = obv.OBV[last_date]
	return obv

def parabolic_sar(tick_data, accel_start=0.02, accel_thresh=0.2, accel_step=0.02):
	""" Computes the parabolic SAR of an asset over time. 
		Source: https://www.tradinformed.com/2014/03/24/calculate-psar-indicator-using-excel/
		Inputs: dataframe with high, low, and closing price over given timespan
		Outputs: parabolic SAR over given timespan
	"""
	# Initializes timestamp and start date variables
	timestamp = tick_data.index
	start_date = timestamp[0]
	# Initializes the seed values for parabolic SAR
	accel = accel_start
	is_falling = True
	extreme_point = tick_data.low[start_date]
	initial_psar = 0
	# Fills PSAR dataframe with seed value
	psar = pd.DataFrame(index=timestamp, columns=['PSAR'])
	psar.PSAR[start_date] = tick_data.high[start_date]
	for i in range(1, len(timestamp)):
		# Gets the date now, the last date, and the date before that
		now_date = timestamp[i]
		last_date = timestamp[i-1]
		# Safeguard for the first iteration, when i == 1
		farther_date = timestamp[i-2] if i > 1 else last_date
		difference = psar.PSAR[last_date] - extreme_point * accel
		# Saves last extreme point for incrementing accel
		last_extreme_point = extreme_point
		# Falling parabolic SAR calculation
		if is_falling:
			initial_psar = max(psar.PSAR[last_date] - difference, tick_data.high[last_date], tick_data.high[farther_date])
			psar.PSAR[now_date] = initial_psar if tick_data.high[now_date] < initial_psar else extreme_point
			extreme_point = min(extreme_point, tick_data.low[now_date])
		# Rising parabolic SAR calculation
		else:
			initial_psar = max(psar.PSAR[last_date] - difference, tick_data.low[last_date], tick_data.low[farther_date])
			psar.PSAR[now_date] = initial_psar if tick_data.low[now_date] > initial_psar else extreme_point
			extreme_point = max(extreme_point, tick_data.high[now_date])
		# Compares previous and current state of is_falling to increment accel properly
		last_is_falling = is_falling
		is_falling = tick_data.close[now_date] > psar.PSAR[now_date]
		# Conditional updates and checks on accel
		if last_is_falling == is_falling:
			if last_extreme_point != extreme_point and accel < accel_thresh:
				accel = accel + accel_step
			else:
				accel = accel_thresh
		else: 
			accel = accel_start
	return psar

def percent_volume_oscillator(volume, num_periods_slow, num_periods_fast):
	""" Computes the percent volume oscillator of an asset over time
		Inputs: choice of function, price input, number of periods for slow MA, number of periods for fast MA
		Outputs: price oscillator over given timespan
	"""
	# Gets the fast EMA of volume
	fast_ema = exponential_moving_average(volume, num_periods=num_periods_fast)
	# Gets the slow EMA of volume
	slow_ema = exponential_moving_average(volume, num_periods=num_periods_slow)
	pct_vol_osc = 100 * (fast_ema - slow_ema) / fast_ema
	return pct_vol_osc

def polarized_fractal_efficiency(tick_data, num_periods, price_col='close'):
	""" Computes the polarized fractal efficiency of stock price.
		Background lies in fractal math.
		Inputs: price Series, number of periods, column to use as price
		Outputs: polarized fractal efficiency over given timespan
	"""
	# Computes PFE before transformation
	price = tick_data[price_col]
	num = (price - price.shift(-num_periods - 1)) ** 2 + num_periods ** 2
	num = num.apply(lambda x: math.sqrt(x) if not pd.isnull(x) else x)
	pfe = 100 * num / price
	# Transforms PFE based on closing price; takes EMA
	close_comp = tick_data.close < tick_data.close.shift(-1)
	pfe[close_comp] = pfe[close_comp].apply(lambda x: -x if not pd.isnull(x) else x)
	return exponential_moving_average(pfe, num_periods=num_periods)

def positive_volume_index(tick_data):
	""" Computes a coefficient on close price, with increments only if volume is increasing
		Closely related to the NVI indicator
		Inputs: volume and closing price
		Outputs: PVI indicator over given timespan
	"""
	pvi = pd.DataFrame(index=tick_data.index, columns=['PVI'])
	pvi.PVI[pvi.index[0]] = 0
	for i in range(1, len(tick_data.index) - 1):
		start_date = pvi.index[i - 1]
		end_date = pvi.index[i]
		# Indicator increments if volume has increased
		increment = (tick_data.close[end_date] - tick_data.close[start_date]) / tick_data.close[start_date] if tick_data.volume[end_date] > tick_data.volume[start_date] else 0
		pvi.PVI[end_date] = pvi.PVI[start_date] + increment
	return pvi

def price_channel(price, num_periods):
	""" Computes the price channels (recent maximum and minimum) of an asset over time.
		Inputs: Series of price over given timespan
		Outputs: high channel and low channel over given timespan
	"""
	# Assume that input is dataframe
	hichannel = pd.DataFrame(index=price.index, columns=['high_channel'])
	lochannel = pd.DataFrame(index=price.index, columns=['low_channel'])
	# Iterates through all datewindows
	for i in range(0, len(price.index) - num_periods):
		# Gets the proper tick date window
		start_date = price.index[i]
		end_date = price.index[i + num_periods]
		price_window = price[start_date:end_date]
		# Gets the recent maximum and minimum relative to the date window
		max_price = price_window.max()
		min_price = price_window.min()
		# Populates the output dataframes
		hichannel.high_channel[end_date] = max_price
		lochannel.low_channel[end_date] = min_price
	return hichannel, lochannel

def price_oscillator(price, moving_avg, num_periods_slow, num_periods_fast):
	""" Computes the price oscillator of a time series over certain timespan, which depends on a choice of moving average function.
		Inputs: choice of function, price input, number of periods for slow MA, number of periods for fast MA
		Outputs: price oscillator over given timespan
	"""
	price_osc = moving_avg(price, num_periods_slow) - moving_avg(price, num_periods_fast)
	price_osc_percent = 100 * price_osc / moving_avg(price, num_periods_fast)
	return price_osc, price_osc_percent

def price_rate_of_change(price, factor=100):
	""" Computes the rate of change of a price, weighted with given factor
		More specific version of general rate of change (not given in this code)
		Inputs: price Series, factor to weigh data
		Outputs: price rate of change over given timespan
	"""
	return factor * price / price.shift(-1)

def price_volume_rank(tick_data, price_col='close'):
	""" Computes a simple indicator based on ranking price and volume changes.
		Inputs: data on closing price, volume, and possibly one other price; 
			choice of column for said price.
		Outputs: price volume ranking
	"""
	price = tick_data[price_col]
	# Gets and combines two boolean comparisons
	comp1 = price > tick_data.close.shift(-1)
	comp2 = tick_data.volume > tick_data.volume.shift(-1)
	# Gets the actual ranking
	pv_rank = pd.Series(index=tick_data.index)
	for ind in pv_rank.index:
		if comp1[ind] and comp2[ind]:
			this_rank = 1
		elif comp1[ind] and not comp2[ind]:
			this_rank = 2
		elif not comp1[ind] and not comp2[ind]:
			this_rank = 3
		else:
			this_rank = 4
		pv_rank[ind] = this_rank
	return pv_rank

def price_volume_trend(tick_data):
	""" Computes the price-volume trend (PVT), which directly depends on price and volume data.
		Related closely to on-balance volume (OBV)
		Inputs: volume and closing price
		Outputs: PVT indicator over given timespan
	"""
	pvt = pd.DataFrame(index=tick_data.index, columns=['PVT'])
	pvt.PVT[pvt.index[0]] = 0
	for i in range(1, len(tick_data.index) - 1):
		start_date = pvt.index[i - 1]
		end_date = pvt.index[i]
		# Indicator accounts for volume and closing price
		pvt.PVT[end_date] = pvt.PVT[start_date] + tick_data.volume[end_date] * (tick_data.close[end_date] - tick_data.close[start_date]) / tick_data.close[start_date]
	return pvt

def qstick(tick_data, moving_avg, num_periods):
	""" Computes the Q-stick indicator of asset data over certain timespan, which depends on a choice of moving average function.
		Inputs: choice of function, dataframe with close and open price over time
		Outputs: price oscillator over given timespan
	"""
	return moving_avg(tick_data.close - tick_data.open, num_periods)

def random_walk_index(tick_data, num_periods=7):
	rwi = pd.DataFrame(index=tick_data.index, columns=['RWI'])
	atr = average_true_range(tick_data, num_periods=num_periods)
	for i in range(0, len(rwi.index) - num_periods - 1):
		start_date = rwi.index[i]
		# Gets the maximum RWI across lookback from 2 to maximum
		rwi_maxes = []
		for lookback in range(2, num_periods + 1):
			end_date = rwi.index[i + lookback + 1]
			# Save the max of each RWI high and RWI low pair
			rwi_low = (tick_data.high[start_date] - tick_data.low[end_date]) / (atr.ATR[end_date] * math.sqrt(lookback))
			rwi_high = (tick_data.high[end_date] - tick_data.low[start_date]) / (atr.ATR[end_date] * math.sqrt(lookback))
			rwi_maxes.append(max(rwi_high, rwi_low))
		rwi.RWI[end_date] = max(rwi_maxes)
	return rwi

def range_indicator(tick_data, num_periods):
	""" Computes the range indicator (RI).
		Its use requires comparing interday RI with intraday RI.
		Inputs: dataframe with high, low, and close data
		Outputs: RI over given timespan
	"""
	# Saves the true range as a Series
	tr_df = true_range(tick_data)
	tr = tr_df.true_range
	# Computes the w-term as intermediate
	w = pd.DataFrame(index=tick_data.index, columns=['W_term'])
	close_comp = tick_data.close > tick_data.close.shift(-1)
	w.W_term[close_comp] = tr[close_comp] / (tick_data.close[close_comp] - tick_data.close.shift(-1)[close_comp])
	close_comp_not = tick_data.close <= tick_data.close.shift(-1)
	w.W_term[close_comp_not] = tr[close_comp_not]
	# Computes the stochastic range
	stoch_range = pd.DataFrame(index=tick_data.index, columns=['SR'])
	stoch_range.SR = (w.W_term - w.W_term.rolling(num_periods).min()) / (w.W_term.rolling(num_periods).max() - w.W_term.rolling(num_periods).min())
	# RI is an EMA of stochastic range
	return exponential_moving_average(stoch_range.SR, num_periods=num_periods)

def rel_momentum_index(price, num_periods):
	""" Computes the relative momentum index of a (closing) price dataset given the number of periods.
		Inputs: price Series (close), number of periods
		Outputs: RMI of closing price
	"""
	# Assume that input is dataframe/Series
	rmi = pd.DataFrame(index=price.index, columns=['RMI'])
	# Gets the variables used in computing at all time points
	upavg = 0
	dnavg = 0
	for i in range(0, len(price.index) - num_periods):
		# Gets the proper tick date window
		start_date = price.index[i]
		end_date = price.index[i + num_periods]
		# Gets some more variables
		up = 0
		dn = 0
		if price[end_date] > price[start_date]: 
			up = price[end_date] - price[start_date]
		else: 
			dn = price[start_date] - price[end_date]
		# Updates upavg and dnavg
		upavg = (upavg * (num_periods - 1) + up) / num_periods
		dnavg = (dnavg * (num_periods - 1) + dn) / num_periods
		# Computes the RMI
		rmi.RMI[end_date] = 100 * upavg / (upavg + dnavg)
	return rmi

def rel_strength_index(price):
	return rel_momentum_index(price, num_periods=14)

def rel_vol_index(price, num_periods):
	""" Computes the relative volatility index of a (closing) price dataset given the number of periods.
		Inputs: price Series (close), number of periods
		Outputs: RMI of closing price
	"""
	# Assume that input is dataframe/Series
	rvi = pd.DataFrame(index=price.index, columns=['RVI'])
	# Gets the variables used in computing at all time points
	upavg = 0
	dnavg = 0
	for i in range(0, len(price.index) - num_periods):
		# Gets the proper tick date window
		start_date = price.index[i]
		end_date = price.index[i + num_periods]
		# Gets some more variables
		up = 0
		dn = 0
		if price[end_date] > price[start_date]: 
			up = price[start_date:end_date].std()
		else: 
			dn = price[start_date:end_date].std()
		# Updates upavg and dnavg
		upavg = (upavg * (num_periods - 1) + up) / num_periods
		dnavg = (dnavg * (num_periods - 1) + dn) / num_periods
		# Computes the RMI
		rvi.RVI[end_date] = 100 * upavg / (upavg + dnavg)
	return rvi

def simple_moving_average(input_values, num_periods=30):
	""" Computes the simple moving average (SMA) of a time series over certain timespan.
		Inputs: input values, number of periods in SMA
		Outputs: SMA over given timespan
	"""
	# Computes the rolling mean (default: 30-day and 90-day)
	sma = input_values.rolling(num_periods).mean()
	sma.columns = ['SMA' + str(num_periods)]
	return sma

def stochastic_momentum_index(tick_data, num_periods=14):
	smi = pd.DataFrame(index=tick_data.index, columns=['SMI'])
	for i in range(0, len(smi.index) - num_periods - 1):
		start_date = smi.index[i]
		end_date = smi.index[i + num_periods]
		cm = tick_data.close[end_date] - (max(tick_data.high[start_date:end_date]) + min(tick_data.low[start_date:end_date])) / 2
		hl = max(tick_data.high[start_date:end_date]) - min(tick_data.low[start_date:end_date])
		smi.SMI[end_date] = 200 * cm / hl
	return smi

def stochastic_oscillator(tick_data, moving_avg, num_periods):
	""" Computes the Stochastic oscillator of an asset over time. 
		Inputs: series with price over given timespan, number of periods to look back, type of moving average to apply
		Outputs: Stochastic oscillator over given timespan
	"""
	percent_k = 100 * general_stochastic(tick_data, num_periods=num_periods)
	percent_k.columns = ['PctK']
	percent_k_smoothed = moving_avg(percent_k.PctK, num_periods)
	fast_d = moving_avg(percent_k.PctK, num_periods)
	slow_d = moving_avg(percent_k_smoothed, num_periods)
	return fast_d, slow_d

def stochastic_rsi(price):
	""" Computes the general Stochastic of the RSI.
		Inputs: price data
		Outputs: stochastic RSI over given timespan
	"""
	rsi = rel_strength_index(price)
	srsi = general_stochastic(rsi, num_periods=14)
	return srsi

def swing_index(tick_data, limit):
	""" Computes the (unnecessarily?) complicated swing index.
		Inputs: data on high, low, open, and close price
		Outputs: swing index over given timespan
	"""
	# Gets the numerator of swing index
	num = tick_data.close - tick_data.close.shift(-1) + 0.5 * (tick_data.close - tick_data.open) + 0.25 * (tick_data.close.shift(-1) - tick_data.open.shift(-1))
	# Gets the K term, a maximum of two differences
	K = pd.concat([tick_data.high - tick_data.close.shift(-1), 
			 tick_data.low - tick_data.close.shift(-1)], axis=1)
	K_max = K.max(axis=1)
	# Gets the R term, which varies at each timestamp based on differences
	R_comp = pd.concat([tick_data.high - tick_data.close.shift(-1), 
				 tick_data.low - tick_data.close.shift(-1), 
				 tick_data.high - tick_data.close], axis=1)
	# Builds dataframe that compares values of aforementioned differences
	R_col_df = pd.DataFrame(index=R_comp.index, columns=['R_term'])
	for ind in R_comp.index:
		R_col_df.R_term[ind] = R_comp.loc[ind, :].idxmax(axis=0)
	# Converts said dataframe to series
	R_col = R_col_df.R_term
	# Builds the R term itself
	R = pd.DataFrame(index=R_col.index, columns=['R_val'])
	R.R_val[R_col == 0] = 0.25 * (tick_data.close.shift(-1)[R_col == 0] - tick_data.open.shift(-1)[R_col == 0]) + 0.5 * (tick_data.low[R_col == 0] - tick_data.close.shift(-1)[R_col == 0]) + (tick_data.high[R_col == 0] - tick_data.close.shift(-1)[R_col == 0])
	R.R_val[R_col == 1] = 0.25 * (tick_data.close.shift(-1)[R_col == 1] - tick_data.open.shift(-1)[R_col == 1]) + 0.5 * (tick_data.high[R_col == 1] - tick_data.close.shift(-1)[R_col == 1]) + (tick_data.low[R_col == 1] - tick_data.close.shift(-1)[R_col == 1])
	R.R_val[R_col == 2] = 0.25 * (tick_data.close.shift(-1)[R_col == 2] - tick_data.open.shift(-1)[R_col == 2]) + (tick_data.high[R_col == 2] - tick_data.low[R_col == 2])
	# Checks for divide-by-zero errors
	R_zero = R[R.R_val == 0]
	if len(R_zero) > 0:
		logger.warning("Divide by zero error indicated in the following fields: {}".format(R_zero))
		R[R.R_val == 0] = 1
	# Puts the terms together
	term1 = num / R.R_val
	term2 = 50 * K_max / limit
	return term1 * term2

def tee_three(input_values, num_periods, vfactor=0.7):
	""" Computes the third generalized DEMA  of an input value. Formally called T3, but
		that would have been an unstylish function name.
		Inputs: input values, factor weight of double ema in the "GD" function
		Outputs: T3 averaging method of input over time
	"""
	# GD is the generalized DEMA function
	def gd(in_val, vfactor=vfactor):
		return exponential_moving_average(in_val, num_periods=num_periods) * (1 + vfactor) - exponential_moving_average(exponential_moving_average(in_val, num_periods=num_periods), num_periods=num_periods) * vfactor
	return gd(gd(gd(input_values)))

def tee_four(input_values, num_periods, vfactor=0.7):
	""" Computes the fourth generalized DEMA of an input value. Formally called T4, but
		that would have been an unstylish function name.
		Inputs: input values, factor weight of double ema in the "GD" function
		Outputs: T4 averaging method of input over time
	"""
	# GD is the generalized DEMA function
	def gd(in_val, vfactor=vfactor):
		return exponential_moving_average(in_val, num_periods=num_periods) * (1 + vfactor) - exponential_moving_average(exponential_moving_average(in_val, num_periods=num_periods), num_periods=num_periods) * vfactor
	return gd(gd(gd(gd(input_values))))

def trend_score(price, num_periods):
	""" Computes the trend score, a rolling sum of binary price movements.
		Inputs: price series
		Outputs: trend score over time
	"""
	# Computes trend increments (same as Klinger's trend variable)
	trend = price - price.shift(-1)
	trend = trend.fillna(0)
	trend = trend.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
	# Computes rolling sum across window
	trend_roll = trend.rolling(num_periods).sum()
	return trend_roll

def triangular_moving_average(input_values, num_periods=30):
	""" Computes the triangular moving average (TMA) of a time series over certain timespan, which weighs the middle values more.
		Inputs: input values, number of periods in TMA
		Outputs: TMA over given timespan
	"""
	periods = num_periods if num_periods % 2 == 0 else num_periods + 1
	per1 = int(periods / 2 + 1)
	tma = simple_moving_average(input_values, num_periods=per1)
	per2 = per1 - 1
	tma = simple_moving_average(tma, num_periods=per2)
	return tma

def triple_ema(input_values, num_periods=30):
	""" Computes the triple exponential moving average (TEMA) of a time series over certain timespan, which weighs the middle values more.
		Inputs: input values, number of periods in TEMA
		Outputs: TEMA over given timespan
	"""
	term1 = 3 * exponential_moving_average(input_values, num_periods=num_periods)
	term2 = 3 * exponential_moving_average(exponential_moving_average(input_values, num_periods=num_periods), num_periods=num_periods)
	term3 = exponential_moving_average(exponential_moving_average(exponential_moving_average(input_values, num_periods=num_periods), num_periods=num_periods), num_periods=num_periods)
	return term1 - term2 + term3

def trix(price, num_periods=30):
	""" Computes the TRIX, dependent on a triple EMA of price.
		Inputs: price Series; number of periods in TRIX
		Outputs: TRIX indicator over given timespan
	"""
	triple_ema = exponential_moving_average(exponential_moving_average(exponential_moving_average(price, num_periods=num_periods), num_periods=num_periods), num_periods=num_periods)
	return 100 * (triple_ema - triple_ema.shift(-1)) / triple_ema

def true_range(tick_data):
	""" Computes the true range of an asset over time.
		Inputs: dataframe wtih closing price, high price, and low price
		Outputs: true range over given timespan
	"""
	# Initializes output as empty dataframe
	trange = pd.DataFrame(index=tick_data.index, columns=['true_range'])
	for i in range(1, len(tick_data.index)):
		# Gets the current and previous date
		this_date = tick_data.index[i]
		last_date = tick_data.index[i-1]
		# Gets the three possibilities for true range
		option1 = abs(tick_data.high[this_date] - tick_data.low[this_date])
		option2 = abs(tick_data.high[this_date] - tick_data.close[last_date])
		option3 = abs(tick_data.low[this_date] - tick_data.close[last_date])
		# Filters based on which is largest
		trange.true_range[this_date] = max(option1,option2,option3)
	return trange

def true_strength_index(price, num_periods=14):
	""" Computes the true strength index (double EMA) of price.
		Inputs: price Series; number of periods
		Outputs: true strength index
	"""
	price_shift = price - price.shift(-1)
	num = exponential_moving_average(exponential_moving_average(price_shift, num_periods=num_periods), num_periods=num_periods)
	denom = exponential_moving_average(exponential_moving_average(abs(price_shift), num_periods=num_periods), num_periods=num_periods)
	return num / denom

def typical_price(tick_data):
	""" Computes the typical price of an asset over time. 
		Inputs: dataframe with closing price, high price, low price over given timespan
		Outputs: average price over given timespan
	"""
	# Assume that input is dataframe
	typ_price = pd.DataFrame(index=tick_data.index, columns=['typical_price'])
	# Adds up the prices into typ_price
	typ_price['typical_price'] = tick_data.close + tick_data.high + tick_data.low
	# Divides by three
	typ_price = typ_price.divide(3)
	return typ_price

def ultimate_oscillator(tick_data, periods=(7,14,28)):
	""" Computes the ultimate oscillator, a triple weighted sum of price info
		Inputs: data on close, low, and high of stock; periods for the moving average
		Outputs: ultimate oscillator over given timespan
	"""
	truelow = pd.concat([tick_data.close.shift(-1), tick_data.low], axis=1).min(axis=1)
	input1 = tick_data.close - truelow
	truerange = true_range(tick_data)['true_range']
	terms = [0, 0, 0]
	for i, period in zip(range(0, 3), periods):
		a1 = simple_moving_average(input1, num_periods=period) * period
		b1 = simple_moving_average(truerange, num_periods=period) * period
		terms[i] = a1.divide(b1)
		break
	return (terms[0] * 4 + terms[1] * 2 + terms[2]) / 7

def variable_moving_average(price, num_periods=30):
	""" Computes the variable moving average, weights based on volatility, in this case CMO
		Inputs: price series and number of periods (default: 30)
		Outputs: VMA indicator
	"""
	# Initializes smoothing constant
	smoothing_constant = 2 / (num_periods + 1)
	# Volatility is the 9-period CMO
	cmo = chande_momentum_oscillator(price, num_periods=9)
	vma = pd.DataFrame(index=price.index, columns=['VMA'])
	for i in range(1, len(price.index)):
		# Gets the current and previous date
		now_date = price.index[i]
		last_date = price.index[i-1]
		# Fills the output dataframe
		vma.VMA[now_date] = smoothing_constant * cmo.CMO[now_date] * price[now_date] + (1 - smoothing_constant * cmo.CMO[now_date]) * price[last_date]
	# Returns output dataframe
	return vma

def vertical_horizontal_filter(tick_data, num_periods):
	""" Computes the vertical horizontal filter (VHF).
		Inputs: tick data with close, high, and low; number of periods
		Outputs: VHF over time
	"""
	# Fills dataframes for global maximum/minimum at each timestep
	highest = pd.Series(index=tick_data.index)
	lowest = pd.Series(index=tick_data.index)
	start_date = tick_data.index[0]
	for date in tick_data.index:
		highest[date] = tick_data.high[start_date:date].max()
		lowest[date] = tick_data.low[start_date:date].min()
	num = highest - lowest
	close_diff = tick_data.close / tick_data.close.shift(-1) - 1
	denom = close_diff.rolling(num_periods).sum()
	return num / denom

def vol_adj_moving_average(tick_data, num_periods, price_col='close'):
	"""
		Computes a moving average adjusted for volume.
		Inputs: dataframe with at least one price and volume; number of periods; 
		choice of price column (default: closing price)
		Outputs: moving average adjusted for volume over given timespan
	"""
	price_vol = tick_data[price_col] * tick_data.volume
	return price_vol.rolling(num_periods).sum() / tick_data.volume.rolling(num_periods).sum()

def weighted_close(tick_data):
	""" Computes the weighted closing price of an asset over time. 
		Inputs: dataframe with closing price, high price, low price over given timespan
		Outputs: weighted closing price over given timespan
	"""
	# Assume that input is dataframe
	weighted_close_price = pd.DataFrame(index=tick_data.index, columns=['weighted_close_price'])
	# Adds up the prices into weighted_close_price
	weighted_close_price['weighted_close_price'] = tick_data.close + tick_data.close + tick_data.high + tick_data.low
	# Divides by four
	weighted_close_price = weighted_close_price.divide(4)
	return weighted_close_price

def weighted_moving_average(price, num_periods):
	""" Computes the weighted moving average, which weighs recent data more.
		Inputs: price Series; number of periods
		Outputs: weighted moving average over given timespan
	"""
	wma = pd.Series(index=price.index)
	for i in range(0, len(price.index) - num_periods - 2):
		start_date = wma.index[i]
		end_date = wma.index[i + num_periods + 1]
		price_int = price[start_date:end_date]
		price_int_sum = 0
		count = 1
		for row in price_int:
			price_int_sum += row * (num_periods - count)
			count = count + 1
		wma[i] = price_int_sum
	wma = wma / (num_periods * (num_periods - 1) / 2)
	return wma

def williams_ad(tick_data):
	""" Computes the Williams accumulation-distribution indicator (cumulative).
		Closely related to the accumulation-distribution line.
		Inputs: dataframe of high, low, and closing price
		Outputs: Williams AD over given timespan
	"""
	ad = pd.Series(index=tick_data.index)
	ad[ad.index[0]] = 0
	for i in range(1, len(ad.index)):
		this_date = ad.index[i]
		last_date = ad.index[i - 1]
		if tick_data.close[this_date] > tick_data.close[last_date]:
			ad[this_date] = ad[last_date] + tick_data.close[this_date] - min(tick_data.low[this_date], tick_data.close[last_date])
		elif tick_data.close[this_date] < tick_data.close[last_date]:
			ad[this_date] = ad[last_date] + max(tick_data.high[this_date], tick_data.close[last_date]) - tick_data.close[this_date]
		else:
			ad[this_date] = ad[last_date]
	return ad

def williams_percent(tick_data, num_periods):
	""" Computes the Williams %R indicator, denoted `williams_percent` or `pct_r` in the code.
		Inputs: dataframe of high, low, and closing price
		Outputs: Williams %R indicator over given timespan
	"""
	pct_r = pd.Series(index=tick_data.index)
	# Iterates through rolling data window
	for i in range(0, len(pct_r.index) - num_periods):
		start_date = pct_r.index[i]
		end_date = pct_r.index[i + num_periods]
		# Computation relies on highest high and lowest low
		pct_r[end_date] = 100 * (max(tick_data.high[start_date:end_date]) - tick_data.close[end_date]) / (max(tick_data.high[start_date:end_date]) - min(tick_data.low[start_date:end_date]))
	return pct_r

def zero_lag_ema(price, num_periods):
	""" Computes the so-called zero lag exponential moving average, which substracts older data to minimize cumulative effect.
		Inputs: price Series, number of periods to run calculation on
		Outputs: zero-lag EMA
	"""
	lag = int((num_periods - 1) / 2)
	ema = pd.Series(index=price.index)
	# Iterates through all datewindows
	for i in range(0, len(price.index) - lag):
		# Computes the de-lagged data
		lag_date = price.index[i]
		now_date = price.index[i + lag]
		price_window = price[lag_date:now_date]
		ema[now_date] = 2 * price_window[now_date] - price_window[lag_date]
	zlema = exponential_moving_average(ema, num_periods=num_periods)
	zlema.name = 'ZLEMA'
	return zlema

if __name__ == "__main__":
	test_technical()