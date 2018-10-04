#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 18:44:30 2018

Builds a JSON file to be fed into Algolia-powered search engine. 

@author: openamiguel
"""
from datetime import date, timedelta
from download import CLoader
import pandas as pd
import technicals as ti
import time
import ticker_universe as tu

# Set up the equitysim loader object
folderpath = "/Users/openamiguel/Documents/EQUITIES/stockDaily"
loader = CLoader(folderpath=folderpath)

# Parses today's date into date object
date_today_str = "2018-08-13"
dt_split = date_today_str.split("-")
date_today = date(int(dt_split[0]), int(dt_split[1]), int(dt_split[2]))

# Saves the period lookbacks (for signals) and the week lookbacks (for pct change)
period_lookbacks = [10, 30, 50, 100, 200, 300]
week_lookbacks = [1, 2, 3, 5, 13, 26, 52, 104, 156, 208, 260]

# Gets the list of symbols to analyze
symbols = set(tu.obtain_parse_wiki(selection="SNP500", ) + 
		    tu.obtain_parse_wiki(selection="DOW30") + 
		    tu.obtain_parse_nasdaq())

# Gets the list of symbols to retrieve detailed information for
special_symbols = ["FB", "AAPL", "MSFT", "GE", "TSLA", "INTC", "QCOM", "GOOG", "GS"]

def get_financials(symbol):
	""" Gets a limited selection of data from the Financials files """
	folderpath = "/Users/openamiguel/Documents/EQUITIES/stockDaily/Financials"
	filepath = "{}/{}_Financials.json".format(folderpath, symbol)
	infile = None
	try:
		infile = open(filepath, 'r+')
	except FileNotFoundError:
		return None
	# All the metadata on a company is on the first line of the file
	first_line = infile.readline()
	# Gets the company name from metadata
	company_name = ""
	try: 
		company_name = first_line[first_line.index("company_name") + 15:]
		company_name = company_name[:company_name.index("\"")]
		# Remove annoying single quotes!
		company_name = company_name.replace("\'", "")
	except ValueError:
		print("Company name for symbol {} not found".format(symbol))
	# Gets the location from metadata (feature on hold)
	location = "Unknown"
	# Gets the industry name from metadata
	industry_name = ""
	try: 
		industry_name = first_line[first_line.index("industry_name") + 16:]
		industry_name = industry_name[:industry_name.index("\"")]
		# Remove annoying single quotes!
		industry_name = industry_name.replace("\'", "")
	except ValueError:
		print("Industry name for symbol {} not found".format(symbol))
	return company_name, location, industry_name

def get_beta(close, baseline_close):
	""" Calculates beta from the past year """
	# Evaluates beta from the past year
	past_date = date_today - timedelta(days=365)
	past_date_str = past_date.strftime("%Y-%m-%d")
	close = close[past_date_str:]
	baseline_close = baseline_close[past_date_str:]
	combined = pd.concat([close, baseline_close], axis=1, sort=False)
	cov = combined.cov()
	var = baseline_close.var()
	all_beta = cov.iloc[1, 0] / var
	return all_beta

def get_technicals(tick_data, baseline_close):
	""" Calculates most recent values of technical indicators """
	price = tick_data.close
	entry_technicals = {}
	# Adds the accumulation-distribution line
	#ad_line = ti.ad_mline(tick_data)
	#entry_technicals['accum_distro_line'] = ad_line[date_today_str]
	# Adds the average price
	avg_price = ti.average_price(tick_data)
	entry_technicals['average_price'] = avg_price.average_price[date_today_str]
	# Adds the Bollinger bands
	lowband, midband, hiband, width = ti.bollinger(tick_data)
	entry_technicals['low_band'] = lowband.SMA20[date_today_str]
	entry_technicals['mid_band'] = midband.SMA20[date_today_str]
	entry_technicals['high_band'] = hiband.SMA20[date_today_str]
	# Adds the MACD
	mcd, macdPct = ti.macd(price)
	entry_technicals['MACD'] = mcd[date_today_str]
	# Adds the parabolic SAR
	psar = ti.parabolic_sar(tick_data)
	entry_technicals['parabolic_SAR'] = psar.PSAR[date_today_str]
	# Adds the relative strength index
	rsi = ti.rel_strength_index(price)
	entry_technicals['RSI'] = rsi.RMI[date_today_str]
	# Adds the swing index (limit = 1000)
	si_1000 = ti.swing_index(tick_data, limit=1000)
	entry_technicals['swing_1000'] = si_1000[date_today_str]
	# Adds the true range
	tr = ti.true_range(tick_data)
	entry_technicals['TR'] = tr.true_range[date_today_str]
	# Adds all the period-based signals
	for per in period_lookbacks:
		# Adds the relative momentum index
		rmi = ti.rel_momentum_index(price, num_periods=per)
		entry_technicals['rel_momentum_index_' + str(per)] = rmi.RMI[date_today_str]
		# Adds the exponential moving average
		ema = ti.exponential_moving_average(price, num_periods=per)
		entry_technicals['exponential_moving_avg_' + str(per)] = ema[date_today_str]
		# Adds the aroon oscillator
		aroon = ti.aroon_oscillator(tick_data, num_periods=per)
		entry_technicals['aroon_oscillator_' + str(per)] = aroon.aroon[date_today_str]
		# Adds the average true range
		atr = ti.average_true_range(tick_data, num_periods=per)
		entry_technicals['avg_true_range_' + str(per)] = atr.ATR[date_today_str]
		# Adds the simple moving average
		sma = ti.simple_moving_average(price, num_periods=per)
		entry_technicals['simple_moving_avg_' + str(per)] = sma[date_today_str]
	return entry_technicals

def get_json(symbol, baseline_close, detail=False): 
	# Add symbol to JSON entry
	json_entry = {}
	json_entry["symbol"] = symbol
	# Add company name, location, and industry name
	if get_financials(symbol) is None:
		print("Financials not found for symbol {}, skipping...")
		return ""
	company_name, location, industry_name = get_financials(symbol)
	json_entry["company"] = company_name
	json_entry["location"] = location
	json_entry["industry"] = industry_name
	# Stand-in for market cap
	json_entry["market_cap"] = -1
	# Prices, prices, prices
	tick_data = loader.load_single_drive(symbol)
	op = tick_data.open
	hi = tick_data.high
	lo = tick_data.low
	close = tick_data.close
	vol = tick_data.volume
	op_tod = op[date_today_str]
	json_entry["price_open"] = op_tod
	hi_tod = hi[date_today_str]
	json_entry["price_high"] = hi_tod
	lo_tod = lo[date_today_str]
	json_entry["price_low"] = lo_tod
	cl_tod = close[date_today_str]
	json_entry["price_close"] = cl_tod
	vl_tod = vol[date_today_str]
	json_entry["price_volume"] = vl_tod
	# Calculates beta of stock over past year
	beta = get_beta(close, baseline_close)
	json_entry["beta"] = beta
	# Percent changes over time
	for week_num in week_lookbacks:
		num_days = week_num * 7
		past_date = date_today - timedelta(days=num_days)
		past_date_str = past_date.strftime("%Y-%m-%d")
		pct_change = -1
		try: 
			pct_change = 100 * (close[date_today_str] - close[past_date_str]) / close[past_date_str]
		except KeyError:
			print("Warning: no date found!!!")
		json_entry["pct_change_" + str(week_num) + "W"] = pct_change
	# Adds all the signals to the JSON terms from fundamentals/basic info
	entry_technicals = get_technicals(tick_data, baseline_close) if detail else {}
	json_entry_all = {**json_entry, **entry_technicals}
	# Replace single-quotes with double-quotes
	json_str = str(json_entry_all).replace("\'", "\"")
	# Replace NaN with -1
	json_str = json_str.replace(": nan", ": -1")
	return json_str

def main():
	# Saves the baseline data
	baseline_symbol = "^GSPC"
	baseline_data = loader.load_single_drive(baseline_symbol)
	baseline_close = baseline_data.close
	
	# Opens the output file and writes a bracket to it
	outpath = "/Users/openamiguel/Desktop/stocks.json"
	outfile = open(outpath, 'w+')
	outfile.write("[\n")
	
	# Builds JSON term for each symbol
	time_start0 = time.time()
	for symbol in symbols:
		print("Starting symbol {}...".format(symbol))
		# Start the timer
		time_start = time.time()
		check = symbol in special_symbols
		json_str = ""
		json_str = get_json(symbol, baseline_close, detail=check)
		"""
		try:
			json_str = get_json(symbol, baseline_close, detail=check)
		except Exception as e:
			print(e)
			print("Skipping symbol {}...".format(symbol))
			continue"""
		if json_str != "":
			outfile.write(json_str + ",\n")
		time_end = time.time()
		time_elapsed = time_end - time_start
		print("Symbol {0} done in time {1:.3f} sec!".format(symbol, time_elapsed))
	outfile.write("]")
	outfile.close()
	time_end0 = time.time()
	time_elapsed = time_end0 - time_start0
	print("Entire process done in time {0:.3f} sec!".format(time_elapsed))

if __name__ == "__main__":
	main()
