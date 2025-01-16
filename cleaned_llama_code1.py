import sys
import argparse

import json
import numpy as np
import pandas as pd

import ollama

# Function to split a DataFrame into smaller chunks of a specified length
def data_slicing(df:pd.DataFrame, slice_length:int):
    row_count = len(df.index)
    count=0
    df_list=[]

    while count<row_count-1:
        start=count
        end=min(start+slice_length,row_count-1)
        df_list.append(df[start:end])
        count+=slice_length
    
    return(df_list)

# Function to adjust JSON strings for consistent formatting
def json_adjustments(string:str):
    if (string)[-1] != "}":
        string+="}"
    
    string=(string).replace("\"subject_type\" \"", "\"subject_type\": \"")

    return string

# Function to interact with the Ollama API to get responses based on user and system messages
def ollama_response(model_name:str, system_message:str, user_message:str):
    response=ollama.chat(
    model=model_name,
    messages=[ {"role":"system", "content": system_message}, {"role": "user", "content": user_message},],)
    response=response["message"]["content"]
    return response

# Helper function to convert various data types into strings
def convert_to_string(variable):
    if isinstance(variable, list):
        return ', '.join(map(str, variable))
    else:
        return str(variable)

# Predefined examples to guide the system message for analysis
output_examples = '''

example 1:
{
  "sentiment": "positive",
  "emotion": "resentment",
  "subject_info":
  [ {"subject_type" : "entity", "subject": "Kamala Harris", "subject_stance": "nonsupportive"} ,
   {"subject_type" : "event", "subject": "Trump giving a speech about Kamala's past", "subject_stance": "negative"} ]
}

example 2:
{
  "sentiment": "NA",
  "emotion": "NA",
  "subject_info":
  [ {"subject_type" : "event", "subject": "news on donald trump affairs", "subject_stance": "negative"} ]
}

example 3:
{
  "sentiment": "negative",
  "emotion": "distrust",
  "subject_info":
  [ {"subject_type" : "entity", "subject": "Donald Trump", "subject_stance": "nonsupportive"} ,
   {"subject_type" : "topic", "subject": "elligations on donald trump", "subject_stance": "supportive"} ]
}

example 4:
{
  "sentiment": "positive",
  "emotion": "excitement",
  "subject_info":
  [ {"subject_type" : "entity", "subject": "Kamla Harris", "subject_stance": "supportive"} ,
   {"subject_type" : "event", "subject": "Jo Biden resigning from candidacy", "subject_stance": "nonsupportive"} ]
}

example 5:
{
  "sentiment": "positive",
  "emotion": "anger, distrust, concern",
  "subject_info":
  [ {"subject_type" : "entity", "subject": "Biden", "subject_stance": "nonsupportive"} ,
   {"subject_type" : "topic", "subject": "Russian nuclear sub in Florida", "subject_stance": "negative"},
   {"subject_type" : "entity", "subject": "Donald Trump", "subject_stance": "supportive"} ]
}

example 6:
{
  "sentiment": "neutral",
  "emotion": "NA",
  "subject_info": [ ]
}


'''

# System message template to guide the LLM in analyzing tweets
system_message=f"""
You are a political tweet text expert analyst. You also analyze hate text in tweets. You will be given 1 tweet and their hashtags related to 2024 US presidential elections. There are two main presidential candidates tweets are about :
 
candidate 1 - Donald Trump (Republican Party)
candidate 2 - Kamala Harris (Democratic Party)

Extract below things from the tweet only if they are explicitly mentioned or directly implied. Avoid assumptions or unrelated entities. If some information is not there, write "NA" in that field.

1. Overall sentiment expressed in the tweet (options: positive, negative, neutral).
2. Overall emotion expressed in the tweet (options: all emotions in the Plutchik's wheel of emotion, NA).
3. Extract and dentify all politics or presidential election related subjects(entity/topic/event) for which stance/sentiment is EXPLICITELY mentioned in the tweet. Entity representation is entity name and event/topic representation is its complete and concise description.
4. Analyze tweet to extract the stance/sentiment of the tweet writer towards extracted subject(entity/topic/event). If its topic, sentiment can be positive or negative. If its event/entity stance can be supportive or nonsupportive.
5. List above all identified entity/topic/event and respective stance/sentiment towards them in appropriate JSON format (subject_type, subject(representation), subject_sentiment). (options: [ subject_type : entity, event, topic], [subject : entity name, event description, topic description], subject_stance: [positive, negative, supportive, nonsupportive, neutral])

Output format:
Below are different example outputs. Provide the output only in JSON format as described in BELOW examples without any extra information or examples. DO NOT output any extra information than JSON.
{output_examples}
"""

# Function to refresh output structure as a dictionary
def refresh_output():
    output_data={
    "tweet_id" : [],
    "tweet_text" : [],
    "sentiment" : [],
    "emotion" : [],
    "subject_type" : [],
    "subject" : [],
    "subject_stance" : []
    }
    return (output_data)


# Main function to process tweets in batches and analyze them using the LLM
def llama_event_detection(startt: int, endd: int, tweets_list: list, model_name: str):
    count=0
    total_count=0
    outputs={}
    current_tweets=""
    bad_count=0

    for df_id in range(startt,endd):
        total_count=df_id+1
        tweet_df=tweets_list[df_id]
        output_data=refresh_output()
        print ("\n***now processing ", str(total_count*1000), "***\n")

        for index,row in tweet_df.iterrows():
            user_message="\ntweet:n"+row["text"]+"\n"
            response=ollama_response(model_name, system_message, user_message)
            response= json_adjustments(response)

            try:
                response_json=json.loads(response)
                if len(response_json["subject_info"])>0:
                    for i in range(0,len(response_json["subject_info"])):
                        output_data["tweet_id"].append(row["id_str"])
                        output_data["tweet_text"].append(row["text"])
                        output_data["sentiment"].append(convert_to_string(response_json.get("sentiment",None)))
                        output_data["emotion"].append(convert_to_string(response_json.get("emotion", None)))
                        output_data["subject_type"].append(response_json["subject_info"][i].get("subject_type",None))
                        output_data["subject"].append(convert_to_string(response_json["subject_info"][i].get("subject",None)))
                        output_data["subject_stance"].append(response_json["subject_info"][i].get("subject_stance",None))
                else:
                    print("*****", row["text"])
                    bad_count+=1
            except:
                print(user_message)
                print("?????" , response)
                bad_count+=1

        print("#### bad_count:", bad_count)
        df_tweet_sentiment_op = pd.DataFrame.from_dict(output_data)
        out_file_name=str(total_count)+"_output.csv"
        df_tweet_sentiment_op.to_csv(out_file_name)


if __name__ == "__main__":
    
    likedquotedretweet = pd.read_pickle('likedquotedretweets1to22.pickle')
    tweets_list=data_slicing(likedquotedretweet,1000)

    # Initialize argparse
    parser = argparse.ArgumentParser(description="Process variables for the script.")
    
    # Add arguments for start and end indices
    parser.add_argument('--startt', type=int, help="Start index for slicing.")
    parser.add_argument('--endd', type=int, help="End index for slicing.")
    parser.add_argument('--model_name', type=str, help="LLM model name",
                        default="llama3.2", choices=["llama3.2"])

    # Parse the arguments
    args = parser.parse_args()

    
        
    assert args.startt and args.endd, "startt and/or endd arguments not provided!"

    llama_event_detection(args.startt, args.endd, tweets_list, args.model_name)