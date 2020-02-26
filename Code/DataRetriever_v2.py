#!/usr/bin/env python
# coding: utf-8



import json 
import requests
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
        self.lst_device = [] ## initialize as empty list
        self.lst_port = [] ## initialize as empty list
        self.deviceID = dict(zip(self.lst_device, self.lst_port))
        self.DateIndex = pd.date_range(start_date, str(int(end_date) + 1), freq='1min').to_frame(index = False, name = 'LogTime')
        self.merged_df = None
        self.dict_dev2name = {} ## initialize as empty dictionary 
        
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
        
    def add_device(self, deviceID):
        
        ## check input type and convert to python dictionary, expected json convert to dict
        if not isinstance(deviceID, dict):
            json2dict = json.loads(deviceID)
            lst_devices = []
            lst_ports =[]
            len_check = []
            for i, (key, val) in enumerate(json2dict.items()):
                #print(key, val)
                len_check.append(len(val))
                if i == 0: 
                    lst_devices = val
                    self.lst_device.append(val)
                elif i == 1:
                    lst_ports = val
                    self.lst_port.append(val)
            ## check the length matches 
            if len(len_check) == 2:
                if all(l == len_check[0] for l in len_check):
                    ## refresh `self.deviceID`
                    self.deviceID = dict(zip(self.lst_device, self.lst_port))
                    sys.stdout.write("Number of Devices entered: #%3s \n" %len_check[0])
                    for key, val in zip(lst_devices, lst_ports):
                        sys.stdout.write("Device [#%3s] ports:" %key)
                        for elmt in val: 
                            sys.stdout.write("#%2s " %elmt)
                        sys.stdout.write('\n')
                else:
                    sys.stderr.write("Number of Devices and Ports unmatched... Devices:#%3s, Ports:#%3s" %(len_check[0],len_check[1]))
            else: 
                sys.stderr.write("Unexpected or Extra variables entered.")
    
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

