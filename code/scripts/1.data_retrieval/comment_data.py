from psaw import PushshiftAPI
import pandas as pd
import numpy as np
from retrieve_data import CommentData


api = PushshiftAPI()

endo_sub = CommentData(api, 'Endo')
endo_sub.retrieve_comments(limit=10000)
endo_sub.comments_to_dataframe()
endo_sub.clean_comment_data()
endo_sub.save_comment_data()

endometriosis_sub = CommentData(api, 'endometriosis')
endometriosis_sub.retrieve_comments(limit=10000)
endometriosis_sub.comments_to_dataframe()
endometriosis_sub.clean_comment_data()
endometriosis_sub.save_comment_data()

PCOS_sub = CommentData(api, 'PCOS')
PCOS_sub.retrieve_comments(limit=10000)
PCOS_sub.comments_to_dataframe()
PCOS_sub.clean_comment_data()
PCOS_sub.save_comment_data()
