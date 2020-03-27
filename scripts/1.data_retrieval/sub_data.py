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

twox_sub = SubmissionData(api, 'TwoXChromosomes')
twox_sub.retrieve_submissions(limit=100000)
twox_sub.submissions_to_dataframe()
twox_sub.clean_sub_data()
twox_sub.save_sub_data()

chronicpain_sub = SubmissionData(api, 'ChronicPain')
chronicpain_sub.retrieve_submissions()
chronicpain_sub.submissions_to_dataframe()
chronicpain_sub.clean_sub_data()
chronicpain_sub.save_sub_data()

infertility_sub = SubmissionData(api, 'infertility')
infertility_sub.retrieve_submissions()
infertility_sub.submissions_to_dataframe()
infertility_sub.clean_sub_data()
infertility_sub.save_sub_data()

trolling_sub = SubmissionData(api, 'trollingforababy')
trolling_sub.retrieve_submissions()
trolling_sub.submissions_to_dataframe()
trolling_sub.clean_sub_data()
trolling_sub.save_sub_data()

PCOS_sub = SubmissionData(api, 'PCOS')
PCOS_sub.retrieve_submissions()
PCOS_sub.submissions_to_dataframe()
PCOS_sub.clean_sub_data()
PCOS_sub.save_sub_data()

TTCPCOS_sub = SubmissionData(api, 'TTC_PCOS')
TTCPCOS_sub.retrieve_submissions()
TTCPCOS_sub.submissions_to_dataframe()
TTCPCOS_sub.clean_sub_data()
TTCPCOS_sub.save_sub_data()

askwomen_sub = SubmissionData(api, 'AskWomen')
askwomen_sub.retrieve_submissions(limit=30000)
askwomen_sub.submissions_to_dataframe()
askwomen_sub.clean_sub_data()
askwomen_sub.save_sub_data()

womenhealth_sub = SubmissionData(api, 'WomensHealth')
womenhealth_sub.retrieve_submissions()
womenhealth_sub.submissions_to_dataframe()
womenhealth_sub.clean_sub_data()
womenhealth_sub.save_sub_data()

obgyn_sub = SubmissionData(api, 'obgyn')
obgyn_sub.retrieve_submissions()
obgyn_sub.submissions_to_dataframe()
obgyn_sub.clean_sub_data()
obgyn_sub.save_sub_data()
