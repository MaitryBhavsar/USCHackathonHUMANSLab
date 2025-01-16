import os
import re
import csv
import json
from datetime import datetime, timezone

import pandas as pd
import dask.dataframe as dd
from dask.diagnostics import ProgressBar

"""
dtypes = {
    'id': 'object', 
    'text': 'str', 
    'retweetedTweet': 'str', 
    'retweetedTweetID': 'float64', 
    'retweetedUserID': 'float64', 
    'id_str': 'object', 
    'replyCount': 'int64', 
    'retweetCount': 'int64', 
    'likeCount': 'int64', 
    'quoteCount': 'int64', 
    'conversationIdStr': 'str', 
    'hashtags': 'object',  # This could be a list or a string
    'viewCount': 'object', 
    'quotedTweet': 'object',  # This could be a string or dictionary
    'in_reply_to_status_id_str': 'str', 
    'in_reply_to_user_id_str': 'str', 
    'user': 'object'  # This can be a dictionary or JSON-like object
}
"""

columns_to_keep = [
    'id', 'text', 'retweetedTweet', 'retweetedTweetID', 'retweetedUserID', 
    'id_str', 'replyCount', 'retweetCount', 'likeCount', 'quoteCount', 
    'conversationIdStr', 'hashtags', 'viewCount', 'quotedTweet', 
    'in_reply_to_status_id_str', 'in_reply_to_user_id_str', 'user'
]

"""
def custom_serializer(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()  # Convert datetime to ISO 8601 string
    if isinstance(obj, timezone):
        return str(obj)  # Convert timezone to string
    raise TypeError(f"Type {type(obj)} not serializable")
"""

def replace_datetime(match):
    year, month, day, hour, minute, second = match.groups()
    # If second is not provided (None), default it to 0
    if second is None:
        second = 0
    else:
        second = int(second) 
    # Return the formatted datetime string
    return f'"{int(year):04d}-{int(month):02d}-{int(day):02d}T{int(hour):02d}:{int(minute):02d}:{second:02d}Z"'

"""
def replace_outer_quotes(json_str):
    # Replace outer single quotes with double quotes but preserve the inner ones
    json_str = re.sub(r"(?<!\\)\'", "\"", json_str)
    return json_str
"""

# Convert to JSON string
def extract_user_info(row):    
    x=row["user"].replace("\"", "'")
    x=x.replace("{'id':", "{\"id\":")
    x=x.replace("'id_str': '", "\"id_str\": \"")
    x=x.replace("', 'url': '", "\", \"url\": \"")
    x=x.replace("', 'username': '", "\", \"username\": \"")
    x=x.replace("', 'rawDescription': '", "\", \"rawDescription\": \"")
    x=x.replace("', 'created':", "\", \"created\":")
    x=x.replace(", 'followersCount': ", ", \"followersCount\": ")
    x=x.replace("'friendsCount': ", "\"friendsCount\": ")
    x=x.replace(" 'statusesCount': ", " \"statusesCount\": ")
    x=x.replace(" 'favouritesCount': ", " \"favouritesCount\": ")
    x=x.replace(" 'listedCount': ", " \"listedCount\": ")
    x=x.replace(" 'mediaCount': ", " \"mediaCount\": ")
    x=x.replace(" 'location': '", " \"location\": \"")
    x=x.replace("', 'profileImageUrl': '", "\", \"profileImageUrl\": \"")
    x=x.replace("', 'profileBannerUrl': '", "\", \"profileBannerUrl\": \"")
    x=x.replace("', 'protected': '", "\", \"protected\": \"")
    x=x.replace("', 'verified':", "\", \"verified\":")
    x=x.replace(", 'blue': ", ", \"blue\": ")
    x=x.replace(", 'blueType': ", ", \"blueType\": ")
    x=x.replace(", 'descriptionLinks': [", ", \"descriptionLinks\": \"[")
    x=x.replace("], '_type': '", "]\", \"_type\": \"")
    x=x.replace("'}", "\"}")
   
    pattern = r"datetime\.datetime\((\d{4}),\s*(\d{1,2}),\s*(\d{1,2}),\s*(\d{1,2}),\s*(\d{1,2}),\s*(\d{1,2})?(?:,|\s*)\s*tzinfo=datetime\.timezone\.utc\)"

    updated_text = re.sub(pattern, replace_datetime, x)

    updated_json_string = re.sub(r': \bFalse\b', ': "False"', updated_text)
    updated_json_string = re.sub(r': \bTrue\b', ': "True"', updated_json_string)
    updated_json_string = re.sub(r': \bNone\b', ': "None"', updated_json_string)
    updated_json_string = re.sub(r'\\.', '', updated_json_string)
    
    try:
        user_info = json.loads(updated_json_string)
    except: 
        user_info={}
    
    return pd.Series({
        'user': user_info.get('id_str', None),
        'user_followersCount': user_info.get('followersCount', None),
        'user_friendsCount': user_info.get('friendsCount', None),
        'created': user_info.get('created', None),
        'user_statusesCount': user_info.get('statusesCount', None),
        'user_favouritesCount': user_info.get('favouritesCount', None),
        'user_listedCount': user_info.get('listedCount', None),
        'user_mediaCount': user_info.get('mediaCount', None),
        'location': user_info.get('location', None)
    })


def extract_view_count(row):
    try:
       viewcount=row["viewCount"].replace("'","\"")
        # Convert string to a dictionary and extract 'count'
       viewcount=json.loads(viewcount)
       return pd.Series({
        'viewCount': viewcount.get("count", None)
       })
    except:
        return pd.Series({
            'viewCount': 0  # Default in case of any error
        })

"""
def check_dtypes(row):
    for col, expected_dtype in dtypes.items():
        if not isinstance(row[col], pd.api.types.pandas_dtype(expected_dtype)):
            print(f"Row does not match expected dtype for {col}:")
            print(row)


def clean_ascii(df):
    return df.applymap(lambda x: ''.join([i for i in str(x) if ord(i) < 128]) if isinstance(x, str) else x)
"""

def load_csvdata(file_path):
    df = pd.read_csv(file_path)
    return df


def preprocess_df(df):
    df_dask = dd.from_pandas(df, npartitions=4) 
    df_en = df_dask[df_dask['lang'] == 'en']
    df_en = df_en[columns_to_keep]

    # Use Dask's map_partitions with Dask DataFrame
    df_en[['user', 'user_followersCount', 'user_friendsCount','created','user_statusesCount','user_favouritesCount','user_listedCount','user_mediaCount', 'location']] = df_en.map_partitions(
        lambda df: df.apply(extract_user_info, axis=1),
        meta={'user': 'str', 'user_followersCount': 'int64', 'user_friendsCount': 'int64', 'created': 'str', 'user_statusesCount': 'str', 'user_favouritesCount': 'str', 'user_listedCount': 'str', 'user_mediaCount': 'str', 'location': 'str'}
    )

    # Apply the viewCount extraction
    df_en[['viewCount']] = df_en.map_partitions(
        lambda df: df.apply(extract_view_count, axis=1),
        meta={'viewCount': 'int64'}
    )
    
    return df_en



def datapre_block(listfolders):
    df_list=[]
    for i in listfolders: 
        print(i)
        folder_path="/media/Seagate_exos14tb_1/maitry/usc-x-24-us-election-mainre/part_"+str(i)

        # Use os.walk() to traverse the folder and subfolders
        for root, dirs, files in os.walk(folder_path):
            # Filter the list of files to only include CSV files
            csv_files = [f for f in files if f.endswith('.csv')]
            print ("**",csv_files)

            # Loop through the list of CSV files in the current folder
            for file in csv_files:
                file_path = os.path.join(root, file)
                result=load_csvdata(file_path)
                df=preprocess_df(result)    
                df_list.append(df)  # Add the DataFrame to the list
                print(file, "completed\n")

    # Concatenate all DataFrames into one DataFrame
    all_data_df = dd.concat(df_list, ignore_index=True)

    with ProgressBar():
        all_data_df.compute().to_pickle('dask_dataframe_1to22.pkl')
    
    print("You can now work with the cleaned dataframe!")


if __name__ == "__main__":
    # List of folder paths you want to search for CSV files    
    listfolders=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
    datapre_block(listfolders)