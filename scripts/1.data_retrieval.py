from psaw import PushshiftAPI
import pandas as pd
import numpy as np


class SubmissionData(object):
    def __init__(self, api_obj, subreddit_name):
        self.api = api_obj
        self.subreddit_name = subreddit_name
        self.submission_results = None
        self.submission_df = None

    def retrieve_submissions(self, limit=None):
        if limit is not None:
            submission_results = list(self.api.search_submissions(subreddit=self.subreddit_name,
                                                                  filter=['title', 'selftext', 'permalink', 'created_utc'],
                                                                  limit=limit))
        else:
            submission_results = list(self.api.search_submissions(subreddit=self.subreddit_name,
                                                                 filter=['title', 'selftext', 'permalink', 'created_utc']))
        self.submission_results = submission_results
        return None

    def submissions_to_dataframe(self):
        sub_dict = {'title': [],
                    'text': [],
                    'link': [],
                    'created_utc': []}

        for post in self.submission_results:
            sub_dict['title'].append(post.title)
            sub_dict['text'].append(post.selftext)
            sub_dict['link'].append(post.permalink)
            sub_dict['created_utc'].append(post.created_utc)

        self.submission_df = pd.DataFrame(sub_dict)
        return None

    def clean_sub_data(self):
        self.submission_df.loc[self.submission_df['text'] == '[removed]', 'text'] = ''
        self.submission_df['text'] = self.submission_df['text'].str.replace('\n', ' ')
        self.submission_df['text'] = self.submission_df['title'] + ' ' + self.submission_df['text']
        self.submission_df['type'] = 'submission'
        self.submission_df['created_utc'] = pd.to_datetime(self.submission_df.created_utc, unit='s')
        self.submission_df['created_mst'] = self.submission_df.created_utc.dt.tz_localize(tz='UTC').dt.tz_convert(tz='US/Mountain')
        self.submission_df = self.submission_df.drop(columns=['title', 'created_utc'])
        return None




api = PushshiftAPI()

endo_sub = SubmissionData(api, 'Endo')
endo_sub.retrieve_submissions(limit=100)
endo_sub.submissions_to_dataframe()
endo_sub.clean_sub_data()
print(endo_sub.submission_df.head())
