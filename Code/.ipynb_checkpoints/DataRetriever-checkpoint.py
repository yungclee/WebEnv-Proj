#!/usr/bin/env python
# coding: utf-8



import json 
import requests
import sys
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm 
import base64 
import functools 
'''
Usage: 
    this class is implemented to use the http ackend service to retreive/scrap the data from the 
    web service from webenv.net. from the pre setted up deviceID and corresponding port in python dictionary 
    
WorkFlow: 
    1) the class is initialized and some information is required to established the necessary credentital for 
        the data retreival 
    2) after specifying the device and ports to be used (using update or add device function) default will set
        to be used for the WebEnv Server room's device ID. ""will need to be manually update for other cases""
    3) retreive and collect the .json file from the API request then unpack with json package 
    4) will map the device and map and reorganize into the closedt minute and output pandas DataFrame.
    
Variable:
    1) start_date: the starting date 
    2) end_date: the end date of the history period 
    3) acc: account information 
    4) pwd: this will be parsed and convert into 64 bit to send request 
    5) verbose: (default:True)
    
Attributes: 
    Some cass attributes will be initialized depending on the variables can be used
    1) lst_device: the list for device ID to retreive information 
    2) lst_port: the list of port that corresponds to the deviceID from the previous attribubtes
    3) device_ID: dictionary that will map 1) and 2) 
    4) DateIndex: current hard coded into "1min" time period.
    5) dict_dev2name: the device and port variable encoding mapping 
'''
    
class DataRetriever:
    
    ## initialize
    def __init__(self, start_date:str, end_date:str, acct:str, pwd:str, verbose = True):
        self.API = 'http://www.webenv.net/noviows/novio.svc/GetDataLog'
        self.start_date = start_date
        self.end_date = end_date
        self.auth = base64.b64encode(bytes(acct + ':' + pwd, 'utf-8')).decode('ascii')
        self.verbose = verbose
        self.lst_device = ['324', '113', '363', '364', '288', '287', '285', '197', '279', '278', '277']
        self.lst_port = [['3', '5'],
                         ['1', '2', '3', '4', '11', '12'],
                         ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
                         ['1'],
                         ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
                         ['1'],
                         ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
                         ['1'],
                         ['1'],
                         ['1'],
                         ['1']]
        self.deviceID = dict(zip(self.lst_device, self.lst_port))
        self.DateIndex = pd.date_range(start_date, str(int(end_date) + 1), freq='1min').to_frame(index = False, name = 'LogTime')
        self.merged_df = None
        self.dict_dev2name = {'LogTime': 'LogTime',
                             '287_ch1': 'Dev1_CPU_loading',
                             '288_ch1': 'Dev0_ambient_temp',
                             '288_ch2': 'Dev0_CPU_temp',
                             '288_ch3': 'Dev0_power_temp',
                             '288_ch4': 'Dev0_fan0_speed',
                             '288_ch5': 'Dev0_fan1_speed',
                             '288_ch6': 'Dev0_fan2_speed',
                             '288_ch7': 'Dev0_fan3_speed',
                             '288_ch8': 'Dev0_fan4_speed',
                             '288_ch9': 'Dev0_fan5_speed',
                             '288_ch10': 'Dev0_power_reading',
                             '363_ch1': 'Dev1_ambient_temp',
                             '363_ch2': 'Dev1_CPU_temp',
                             '363_ch3': 'Dev1_power_temp',
                             '363_ch4': 'Dev1_fan0_speed',
                             '363_ch5': 'Dev1_fan1_speed',
                             '363_ch6': 'Dev1_fan2_speed',
                             '363_ch7': 'Dev1_fan3_speed',
                             '363_ch8': 'Dev1_fan4_speed',
                             '363_ch9': 'Dev1_fan5_speed',
                             '363_ch10': 'Dev1_power_reading',
                             '197_ch1': 'Dev2_CPU_loading',
                             '285_ch1': 'Dev2_ambient_temp',
                             '285_ch2': 'Dev2_CPU_temp',
                             '285_ch3': 'Dev2_power_temp',
                             '285_ch4': 'Dev2_fan0_speed',
                             '285_ch5': 'Dev2_fan1_speed',
                             '285_ch6': 'Dev2_fan2_speed',
                             '285_ch7': 'Dev2_fan3_speed',
                             '285_ch8': 'Dev2_fan4_speed',
                             '285_ch9': 'Dev2_fan5_speed',
                             '285_ch10': 'Dev2_power_reading',
                             '279_ch1': 'FileServer_CPU_loading',
                             '324_ch3': 'DC_cold_aisle',
                             '324_ch5': 'DC_hot_aisle',
                             '278_ch1': 'ERP_CPU_loading',
                             '113_ch1': 'AC_avg_usage',
                             '113_ch2': 'IT_avg_usage',
                             '113_ch3': 'DC_avg_usage',
                             '113_ch4': 'PUE',
                             '113_ch11': 'AC0_current',
                             '113_ch12': 'AC1_current',
                             '364_ch1': 'Dev0_CPU_loading',
                             '277_ch1': 'WebEnv_CPU_loading'}
        
        ## check input validity
        if len(self.start_date) != 8 or len(self.end_date)!= 8: 
            print("Input must in format YYYYMMDD!")
            return;
        # check if start date and end date sequence validity
        if self.start_date > self.end_date: 
            print("Please check the order of start date and end date")
            return;
    
    def get_deviceID(self):
        for pos, k in enumerate(self.deviceID.keys()):
            print("#%2s Port %3s :" %(pos,k), end = '')
            print(",".join(self.deviceID[k]))
        
    def add_device(self, deviceID:dict):
        for elmt in deviceID.keys(): 
            if elmt in self.deviceID:
                if self.verbose: print ('Device [%s] existed' %elmt)
            else: 
                if not all(isinstance(item, str) for item in deviceID[elmt]):
                    print("InstanceError: portID should be a string, failed to add device.")
                    return 
                self.deviceID[elmt] = deviceID[elmt]
                if self.verbose: print ('Adding new device [%s]...' %elmt)
    
    def convert_closet_minute(self, input_str:str, verbose = False):
        date = datetime.strptime(input_str,'%Y-%m-%d %H:%M:%S')
        if verbose: print('from:' ,date)
        if date.second >= 30: 
            date += timedelta(minutes = 1)
        date -= timedelta(seconds = date.second)
        if verbose: print('to:' ,date)
        return date
    
    def retrieve(self):
        '''
        Variables: None
        Usage: 
            The method will retrive all datalog with the deviceID and specified port number. 
            the requested data from the web service return is in .json format will be dumped into python panda 
            for further preprocessing
        Return: 
            A pandas DataFrame, merged
        '''
        if self.verbose: print("Begin retrieving data ...")
        df_out = self.DateIndex
        for pos, deviceID in tqdm(enumerate(self.deviceID.keys())):
            print('file: (' , str(pos + 1), '/ ' + str(len(self.deviceID.keys())) + ')')
            print('---------------------------------------')
            print('Getting Device #%s Data ....' %deviceID)
            df = self.get_datalog(deviceID, self.deviceID[deviceID])
            ## columns to get
            var_lst = ['LogTime'] + ['ch' + str(x)for x in self.deviceID[deviceID]]
            df = df[var_lst]
            df['LogTime'] = df['LogTime'].apply(lambda x :self.convert_closet_minute(x))
            df.columns = ['LogTime'] + [deviceID + '_ch' + str(x)for x in self.deviceID[deviceID]]
            print(df.columns)
            df_out = df_out.merge(df,on=['LogTime'],how='left', copy = False)
        #self.df_out += place_hold
        if self.verbose: 
            print("Data Retrieval complete")
        #df = functools.reduce(lambda left,right: pd.merge(left,right,on=['LogTime'],how='outer'), place_hold)
        df_out.columns = [self.dict_dev2name[x] for x in df_out.columns]
        if self.verbose: print("Merging completed.")
        return df_out.drop_duplicates()
    
    def get_datalog(self, deviceID, PortInfo):
        '''
        Variables: 
            1. start_date: the start date of the data log requesting
            2. end_date: the last date of the data log requesting
            3. API: https service address
            4. authorization_code: required and will be parsed for API

        Usage: 
            The program will attempt to request the information based on the user input and return pandas data frame.
            Success: 
                return pandas data frame including the information from the request
            Failure: 
                1. if any of the arguments is invalid will print error message and return; 
                2. if the request is not successful from the server will parse the message and return
        '''
        
        url = "/".join([self.API, self.auth, deviceID, self.start_date, self.end_date])

        if self.verbose: print(url)
        ## get data, code refer and modified from "https://realpython.com/python-requests/"
        try:
            response = requests.get(url)

            # If the response was successful, no Exception will be raised
            response.raise_for_status()
        except HTTPError as http_err:
            print(f'HTTP error occurred: {http_err}')  
        except Exception as err:
            print(f'Other error occurred: {err}')  

        ## extract information from json
        ## first convert json into dictionary can check if the status is correct
        obj = json.loads(response.text)

        if obj['status'] != 'success':
            print('Failed! ' + obj['status'])
            return
        
        ## if it is correctly downloaded then use the port information to select the information to keep
        return pd.DataFrame(obj['datalog'], dtype=float)

