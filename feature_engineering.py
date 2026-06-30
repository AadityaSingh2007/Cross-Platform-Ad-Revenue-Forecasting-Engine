import pandas as pd

def create_features(df):

    df["Date"]=pd.to_datetime(df["Date"])

    df["Year"]=df["Date"].dt.year

    df["Month"]=df["Date"].dt.month

    df["Day"]=df["Date"].dt.day

    df["DayOfWeek"]=df["Date"].dt.dayofweek

    df["Week"]=df["Date"].dt.isocalendar().week.astype(int)

    df=df.drop(columns=["Date"])

    df=df.drop(columns=[

        "CampaignName",

        "Conversions"

    ])

    df=pd.get_dummies(df,columns=["CampaignType","Source"],
        drop_first=True,
        dtype=int
    )
    return df
