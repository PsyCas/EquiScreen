## This code contains the re-consolidated download functions, and can perform any one of the following tasks:
## Download one stock (one-stock-one-file) from API, load one stock (one-stock-one-variable) from local drive, download many stocks (one-stock-one-file) from API, or load many stocks (many-stocks-one-variable) from local drive
## Author: Miguel Opeña
## Version: 2.3.0

import datetime
import logging
import numpy as np
import os
import pandas as pd
import time
import sys
from urllib.request import urlopen

from command_parser import CCmdParser
import io_support

LOGDIR = "/Users/openamiguel/Desktop/LOG"
# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Set file path for logger
handler = logging.FileHandler('{}/equitysim_download.log'.format(LOGDIR))
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

# Sample format for stocks: "{}function=TIME_SERIES_{}&symbol={}&apikey={}&datatype={}&outputsize={}"
# Sample format for forex: "{}function=FX_{}&from_symbol={}&to_symbol={}&apikey={}&datatype={}&outputsize={}"

class CDownloader:
	""" A class to download one, or many, symbols from AlphaVantage """
	def __init__(self, folderpath, api_key, function="DAILY", interval="", 
		    output_size="full", datatype="csv", 
		    url_format="{}function=TIME_SERIES_{}&symbol={}&apikey={}&datatype={}&outputsize={}"):
		self.folderpath = folderpath
		self.api_key = api_key
		self.function = function
		self.interval = interval
		# Checks if user has failed to account for the interval
		if function == "INTRADAY" and interval == "":
			logger.warning("Class CDownloader constructor must take interval as a parameter if INTRADAY chosen as function")
			logger.warning("Default value of interval set to 1 minute")
			self.interval = "1min"
		self.output_size = output_size
		self.datatype = datatype
		self.url_format = url_format
		# Number of seconds to delay between each downloader query
		self.delay = 15
		# Start of the URL for AlphaVantage queries
		self.main_url = "https://www.alphavantage.co/query?"

	def load_single(self, symbol, writefile=False):
		""" Downloads data on a single symbol from AlphaVantage according to user parameters, as a dataframe and (if prompted) as a file. 
			See the AlphaVantage documentation for more details. 
			Inputs: symbol (can be a tuple or list of two symbols), order to
				write file (default: No)
			Outputs: dataframe with all available data on symbol
		"""
		# Checks if the read path involves a stock or forex
		read_path = ""
		# Symbol string will come up in the file name
		symbol_str = ""
		# Forex case
		if type(symbol) is tuple or type(symbol) is list and len(symbol) >= 2:
			logger.info("Downloading the provided symbols: {} and {}".format(symbol[0], symbol[1]))
			read_path = self.url_format.format(self.main_url, self.function, symbol[0], symbol[1], self.api_key, self.datatype, self.output_size)
			symbol_str = symbol[0] + "_" + symbol[1]
		# Equity case
		elif type(symbol) is str or len(symbol) == 1:
			symbol_str = symbol[0] if type(symbol) is not str else symbol
			logger.info("Downloading the symbol {} from AlphaVantage".format(symbol_str))
			read_path = self.url_format.format(self.main_url, self.function, symbol_str, self.api_key, self.datatype, self.output_size)
		# Outputs the read path file
		logger.debug("Attempting to scrape URL: %s", read_path)
		# Checks if the function is intraday (regardless of the type of data)
		if self.function == "INTRADAY":
			read_path = read_path + "&interval=" + self.interval
		tick_data = None
		# Accounts for the fact that AlphaVantage lacks certain high-volume ETFs and mutual funds
		try:
			tick_data = pd.read_csv(read_path, index_col='timestamp')
		except ValueError:
			logger.error(symbol_str + " not found by AlphaVantage. Download unsuccessful.")
			return tick_data
		logger.info(symbol_str + " successfully downloaded!")
		# Flips the data around (AlphaVantage presents it in reverse chronological order, but I prefer regular chronological)
		tick_data = tick_data.reindex(index=tick_data.index[::-1])
		# Saves ticker data to file, if requested
		if writefile:
			logger.info("Saving data on " + symbol_str + "...")
			write_path = self.folderpath + "/" + symbol_str + "_" + self.function
			if self.interval != "": 
				write_path = write_path + "&" + self.interval
			tick_data.to_csv(write_path + "." + self.datatype)
			logger.info("Data on " + symbol_str + " successfully saved!")
		# Returns the data on symbol
		return tick_data

	def load_separate(self, tickerverse):
		""" Downloads OHCLV (open-high-close-low-volume) data on given tickers.
			Inputs: ticker universe
			Outputs: True if everything works
		"""
		current_symbols = io_support.get_current_symbols(self.folderpath)
		for symbol in tickerverse:
			if symbol in current_symbols: continue
			# Read each symbol and write to file (hence writeFile=True)
			self.load_single(symbol, writefile=True)
			# Delay prevents HTTP 503 errors
			time.sleep(self.delay)
		return True

class CLoader:
	""" A class to load one, or several, symbols from local hard drive """
	def __init__(self, folderpath, function="DAILY", interval="", 
		    output_size="full", datatype="csv"):
		self.folderpath = folderpath
		self.function = function
		self.interval = interval
		# Checks if user has failed to account for the interval
		if function == "INTRADAY" and interval == "":
			logger.warning("Class CLoader constructor must take interval as a parameter if INTRADAY chosen as function")
			logger.warning("Default value of interval set to 1 minute")
			self.interval = "1min"
		self.output_size = output_size
		self.datatype = datatype

	def load_single_drive(self, symbol):
		""" Downloads data on a single file (equity or forex) from local drive. 
			Inputs: symbol String or tuple object
			Outputs: dataframe with all available data on symbol
		"""
		# Checks if the symbol input is forex or equity
		readpath = ""
		symbol_str = ""
		# Forex case
		if type(symbol) is tuple or type(symbol) is list and len(symbol) >= 2:
			symbol_str = symbol[0] + "_" + symbol[1]
		# Equity case
		elif type(symbol) is str or len(symbol) == 1:
			symbol_str = symbol[0] if type(symbol) is not str else symbol
		readpath = self.folderpath + "/" + symbol_str + "_" + self.function
		if self.interval != "":
			readpath = readpath + "&" + self.interval
		readpath = readpath + "." + self.datatype
		logger.info("Retrieving " + symbol_str + " from local drive...")
		tick_data = None
		try:
			tick_data = pd.read_csv(readpath, index_col='timestamp')
		except FileNotFoundError:
			logger.error("Retrieval unsuccessful. File not found at " + readpath)
			return tick_data
		# De-duplicates the index
		tick_data = tick_data[~tick_data.index.duplicated(keep='first')]
		logger.info("Data on " + symbol_str + " successfully retrieved!")
		return tick_data

	def load_combined_drive(self, tickerverse, column_choice="close"):
		""" Downloads OHCLV (open-high-close-low-volume) data on given tickers in compact or full form.
			Inputs: ticker universe, choice of column to write (default: close)
			Outputs: combined output as dataframe
		"""
		combined_output = pd.DataFrame()
		for symbol in tickerverse:
			# Read each symbol and concatenate with previous symbols
			tick_data = self.load_single_drive(symbol)
			combined_output = pd.concat([combined_output, tick_data[column_choice]], axis=1)
		# Makes each column the symbol of asset (to avoid confusion)
		combined_output.columns = tickerverse
		return combined_output

class CClinicalDownloader:
	""" A class to download clinical trial data from clinicaltrials.gov
		Word of warning: this dataset is massive.
	"""
	def __init__(self):
		self.url_format = "https://clinicaltrials.gov/ct2/show/{}?displayxml=true"

	def parse_record(self):
		# Sample ID: NCT00001372
		url = self.url_format.format("NCT03254875")
		# study.nct_id
		# study.title
		# study.status (open? recruiting?)
		# study.gender
		# study.min_age
		# study.enrollment
		# funded_by_nih (look for "NIH" in a study.funded_bys.funded_by field)
		# study.start_date
		# study.primary_completion_date
		# study.phases.phase (first entry)
		# study.locations.location (all entries)
			# Split the location entry on comma
			# Delete "United States" if present in location entry
			# Replace US state name with abbrev (build mapper)
			# Replace "/" with ": "
		# study.study_results
		# study.conditions (first three)
		# study.interventions
			# Save the intervention type (as lowercase) and name as follows:
			# "{}!{}".format(type.lower(), name)
			# ex. "drug!Cabozantinib"
		# study.sponsors.lead_sponsor
		# pipe-delimited list of study.sponsors.collaborator
		# study.url

class CMacroDownloader:
	""" A class to download macro data from handpicked sources. """
	def __init__(self):
		# Initialize variables
		now = datetime.datetime.now()
		self.current_year = now.year

	def yield_curve_year(self, year, maturity='10Y'):
		""" Scrapes the US Treasury's website for yield curve data stored in XML files. 
			Note: the provided URL gets data from all years
			Input: year to get data from, maturity of desired bond (default: 10-year T-Note)
			Output: pandas Series of desired bond's yield curve since January 2, 1990
		"""
		# Builds a simple code to parse an XML line
		def parse_line(line):
			logger.debug("Currently reading line: %s", line)
			# Checks if the line entry is null
			# Interestingly, this happened in 2010 data, but not 1991-2009
			if "null=\"true\"" in line:
				return np.NaN
			start_index = line.index('>') + 1
			line = line[start_index:]
			end_index = line.index('<')
			line = line[:end_index]
			return line
		# Checks if the input year is valid
		if int(year) < 1991 and int(year) > self.current_year:
			logger.error("Invalid input year given to CMacroDownloader.yield_curve(...)")
			logger.error("Please give CMacroDownloader.yield_curve(...) an input year between 1991 and %d", self.current_year)
			return None
		logger.info("Processing US Treasury yield curve data, year %d", year)
		# Opens the current link to Treasury yield curve data
		yield_curve_url = "http://data.treasury.gov/feed.svc/DailyTreasuryYieldCurveRateData?$filter=year(NEW_DATE)%20eq%20{}".format(year)
		logger.debug("Opening US Treasury webpage...")
		webpage = urlopen(yield_curve_url)
		logger.debug("Webpage successfully opened!")
		html_str = webpage.read()
		webpage.close()
		# Builds a String version of the maturity
		maturity_code = maturity.replace('M', 'MONTH').replace('Y', 'YEAR')
		# Builds dataframe of timestamped yield curve data
		yield_curve_df = pd.DataFrame()
		# Builds new row for each data entry
		new_row = {'timestamp':'', maturity_code:''}
		for line in str(html_str).split('\\n'):
			# Reads the date where provided
			if "NEW_DATE" in line:
				date = parse_line(line).split('T')[0]
				new_row['timestamp'] = date
			# Reads the value where provided
			# Also writes to the folder
			elif maturity_code in line and "DISPLAY" not in line:
				value = parse_line(line)
				new_row[maturity_code] = value
				logger.debug("Adding new row to the dataframe: %s", str(new_row))
				yield_curve_df = yield_curve_df.append(new_row, ignore_index=True, sort=False)
				new_row = {'timestamp':'', maturity_code:''}
		# Returns the dataframe
		yield_curve_df = yield_curve_df.set_index('timestamp')
		yield_curve_df = yield_curve_df.sort_index()
		return yield_curve_df

	def yield_curve_multi(self, start_year=1991, end_year=2018, maturity='10Y'):
		""" Employs the yield_curve_year to get multiple years of yield curve data, 
			as a consolidated dataframe.
			Inputs: start year (default: 1991), end year (default: 2018); 
				maturity of desired bond (default: 10-year T-Note)
			Outputs: dataframe with all the desired data
		"""
		multi_year_df = pd.DataFrame()
		for year in range(start_year, end_year + 1):
			year_df = self.yield_curve_year(year, maturity=maturity)
			multi_year_df = pd.concat([multi_year_df, year_df], sort=False)
		return multi_year_df

	def get_world_bank(self):
		# Gets data from the World Bank
		# Download file from the World Bank link
		# Parse it and clean if needed
		# Save to folderpath
		return

def main():
	""" User interacts with interface through command prompt, which obtains several "input" data. 
		Here are some examples of how to run this program: 

		python download.py -tickerUniverse SNP500 -folderPath C:/Users/Miguel/Documents/EQUITIES/stockDaily -apiKey <INSERT KEY> -function DAILY
			This will download files of daily data on S&P 500 tickers to the desired folder path.

		python download.py -tickerUniverse AAPL -folderPath C:/Users/Miguel/Documents/EQUITIES/stockIntraday1Min -apiKey <INSERT KEY> -function INTRADAY -interval 1min
			This will download files of daily data on S&P 500 tickers to the desired folder path.

		Inputs: implicit through command prompt
		Outputs: 0 if everything works
	"""
	prompts = sys.argv
	## Handles which symbol the user wants to download.
	cmdparser = CCmdParser(prompts)
	tickerverse, name = cmdparser.get_tickerverse()
	## Handles where the user wants to download their files. 
	# Default folder path is relevant to the author only. 
	folder_path = cmdparser.get_generic(query="-folderPath", default="/Users/openamiguel/Documents/EQUITIES/stockDaily", req=False)
	## Handles the user's API key. 
	api_key = cmdparser.get_generic(query="-apiKey")
	## Handles the desired time series function. 
	function = cmdparser.get_generic(query="-function")
	## Handles the special case: if INTRADAY selected. 
	interval = cmdparser.get_generic(query="-interval") if function == "INTRADAY" else ""
	## Handles user choice of forex or equity (not forex)
	if name == "FOREX":
		fx_format = "{}function=FX_{}&from_symbol={}&to_symbol={}&apikey={}&datatype={}&outputsize={}"
		downloader = CDownloader(folder_path, api_key, function, interval, url_format=fx_format)
		downloader.load_separate(tickerverse)
	else:
		downloader = CDownloader(folder_path, api_key, function, interval)
		downloader.load_separate(tickerverse)
	## Closing output
	logger.info("Download complete. Have a nice day!")

if __name__ == "__main__":
	"""
	md = CMacroDownloader()
	df = md.yield_curve_multi()
	df.to_csv("/Users/openamiguel/Desktop/UST_10year.txt", sep='\t')
	"""
	main()