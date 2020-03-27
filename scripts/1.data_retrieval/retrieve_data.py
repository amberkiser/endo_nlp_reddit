from psaw import PushshiftAPI
import pandas as pd
import numpy as np
import datetime


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
            try:
                title = post.title
                text = post.selftext
                link = post.permalink
                created = post.created_utc
            except AttributeError:
                continue
            else:
                sub_dict['title'].append(title)
                sub_dict['text'].append(text)
                sub_dict['link'].append(link)
                sub_dict['created_utc'].append(created)

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

    def save_sub_data(self):
        self.submission_df.to_csv('../../data/%s_sub_data.csv' % self.subreddit_name, encoding='utf-8-sig')
        return None


class CommentData(object):
    def __init__(self, api_obj, subreddit_name):
        self.api = api_obj
        self.subreddit_name = subreddit_name
        self.comment_results = None
        self.comment_df = None

    def retrieve_comments(self, limit=None):
        if limit is not None:
            comment_results = list(self.api.search_comments(subreddit=self.subreddit_name,
                                                            filter=['body', 'permalink', 'created_utc'],
                                                            limit=limit))
        else:
            comment_results = list(self.api.search_comments(subreddit=self.subreddit_name,
                                                            filter=['body', 'permalink', 'created_utc']))
        self.comment_results = comment_results
        return None

    def comments_to_dataframe(self):
        comm_dict = {'text': [],
                     'link': [],
                     'created_utc': []}

        for post in self.comment_results:
            try:
                text = post.body
                link = post.permalink
                created = post.created_utc
            except AttributeError:
                continue
            else:
                comm_dict['text'].append(text)
                comm_dict['link'].append(link)
                comm_dict['created_utc'].append(created)

        self.comment_df = pd.DataFrame(comm_dict)
        return None

    def clean_comment_data(self):
        self.comment_df.loc[self.comment_df['text'] == '[removed]', 'text'] = ''
        self.comment_df['text'] = self.comment_df['text'].str.replace('\n', ' ')
        self.comment_df['type'] = 'comment'
        self.comment_df['created_utc'] = pd.to_datetime(self.comment_df.created_utc, unit='s')
        self.comment_df['created_mst'] = self.comment_df.created_utc.dt.tz_localize(tz='UTC').dt.tz_convert(tz='US/Mountain')
        self.comment_df = self.comment_df.drop(columns=['created_utc'])
        return None

    def save_comment_data(self):
        self.comment_df.to_csv('../../data/comment_data/%s_comment_data.csv' % self.subreddit_name, encoding='utf-8-sig')
        return None
