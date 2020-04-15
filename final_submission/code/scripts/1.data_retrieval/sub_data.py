from psaw import PushshiftAPI
import pandas as pd
import numpy as np
from retrieve_data import SubmissionData


api = PushshiftAPI()

endo_sub = SubmissionData(api, 'Endo')
endo_sub.retrieve_submissions()
endo_sub.submissions_to_dataframe()
endo_sub.clean_sub_data()
endo_sub.save_sub_data()

endometriosis_sub = SubmissionData(api, 'endometriosis')
endometriosis_sub.retrieve_submissions()
endometriosis_sub.submissions_to_dataframe()
endometriosis_sub.clean_sub_data()
endometriosis_sub.save_sub_data()

PCOS_sub = SubmissionData(api, 'PCOS')
PCOS_sub.retrieve_submissions()
PCOS_sub.submissions_to_dataframe()
PCOS_sub.clean_sub_data()
PCOS_sub.save_sub_data()
