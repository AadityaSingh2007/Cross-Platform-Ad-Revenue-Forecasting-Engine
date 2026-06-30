import pandas as pd
def load_google(path):
    google = pd.read_csv(path)
    google["metrics_cost_micros"] /= 1000000
    google = google.rename(columns={
        "metrics_cost_micros":"Spend",
        "metrics_conversions":"Conversions",
        "metrics_clicks":"Clicks",
        "metrics_impressions":"Impressions",
        "segments_date":"Date",
        "metrics_conversions_value":"Revenue",
        "campaign_advertising_channel_type":"CampaignType",
        "campaign_budget_amount":"campaign_budget",
        "campaign_name":"CampaignName"
    })

    google = google.drop(columns=[
        "metrics_video_views",
        "Unnamed: 0"
    ])

    google["CampaignType"] = google["CampaignType"].str.upper()

    google["Source"] = "Google"

    return google

def load_bing(path):
    bing = pd.read_csv(path)
    bing = bing.rename(columns={
        "CampaignId":"campaign_id",
        "TimePeriod":"Date",
        "DailyBudget":"campaign_budget"
    })
    bing["CampaignType"] = bing["CampaignType"].str.upper()
    bing["Source"] = "Bing"
    bing = bing.drop(columns=["Unnamed: 0"])

    return bing

def merge_data(google,bing):

    columns=[
        "Date",
        "CampaignName",
        "CampaignType",
        "campaign_budget",
        "Spend",
        "Impressions",
        "Clicks",
        "Conversions",
        "Revenue",
        "Source"
    ]

    google=google[columns]

    bing=bing[columns]

    df=pd.concat([google,bing],ignore_index=True)

    return df

def clean_data(df):
    df=df.dropna()

    df["CampaignType"]=df["CampaignType"].replace({
        "PERFORMANCE_MAX":"PERFORMANCEMAX"
    })

    return df

def preprocess_data(google_path,bing_path):

    google=load_google(google_path)

    bing=load_bing(bing_path)

    df=merge_data(google,bing)

    df=clean_data(df)

    return df
