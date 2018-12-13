---
title: EDA
notebook: LC_EDA.ipynb
nav_include: 3
---

## Contents
{:.no_toc}
*  
{: toc}




```python
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from plotly import tools
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

import seaborn as sns
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 500)

import warnings
warnings.filterwarnings('ignore')

% matplotlib inline
```



<script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script><script>requirejs.config({paths: { 'plotly': ['https://cdn.plot.ly/plotly-latest.min']},});if(!window._Plotly) {require(['plotly'],function(plotly) {window._Plotly=plotly;});}</script>


## Data Pre-Processing

Merge the yearly and quarterly data downloaded from lending club website. 



```python
loanstats_input = ['2007_2011', '2012_2013', '2014', '2015', 
                   '2016Q1', '2016Q2', '2016Q3', '2016Q4', 
                   '2017Q1', '2017Q2', '2017Q3', '2017Q4', 
                   '2018Q1', '2018Q2', '2018Q3']

for i in loanstats_input: 
    if i == '2007_2011': 
        df_raw = pd.read_csv("data/LoanStats/LoanStats_securev1_%s.csv" % i, header=1)
        df_raw = df_raw[:-2]
    else: 
        temp = pd.read_csv("data/LoanStats/LoanStats_securev1_%s.csv" % i, header=1)
        temp = temp[:-2]
        df_raw = df_raw.append(temp)
        
df_raw = df_raw.reset_index(drop=True)
```


### Feature Selection
There are totally 151 columns in the raw dataset. In the initial feature selection stage, we applied several very rigorous  selection criteria: 
- The full sample feature coverage should be larger than 60%, otherwise, there are two many missing values. 
    - Around 5% of the loans are joint applications in full sample, which means 95% of the feature columns regarding the second applicatant will be missing. Thus, all the columns regarding the second applicant are dropped, for example FICO scores (sec_app_fico_range_low, sec_app_fico_range_high), earliest credit line at time (sec_app_earliest_cr_line), etc for the second applicant.
    - Some other columns have coverage less than 60% as well. For example, column number of months since the borrower's last delinquency (mths_since_last_major_derog) has only 25% coverage. For the missing data, it's impossible for us to know exactly the reason behind it, either because of unavailability by nature or unwillingness of applicants providing the information. Thus, we decided to drop these types of columns as well.
- The features with look-ahead bias are dropped. 
    - Column post charge off gross recovery (recoveries) will directly indicate the loan has been in charge-off status. However, as an investor, we want to predict if loan is going to end up as a good loan or bad loan at the initiation stage. The information of recoveries is not what we know about beforehand. Thus, we have to drop it from the predictors. 
    - There are some other columns have to be dropped as well, for example late fees received to date (total_rec_late_fee), payments received to date for total amount funded, etc. All the information that is not known at the beginning of the application should not be included as predictors. 
- The redundant features are dropped. 
    - Credit grades and credit sub grades contain the same information, but just in different granularity. We kept grades and dropped sub grades. 
    - Zip codes and States also contain similar information, and we kept states as predictor, partly because there are too many zip codes.   
    - From fixed income formula, we can mathematically calculate the monthly installment amount given annual interest rate, term and loan amount. Thus, installment does not contain any new information, and thus is dropped.   



```python
df_description = pd.read_excel('data/LCDataDictionary_select.xlsx', sheet_name='LoanStats')
df_description.style.set_properties(subset=['Description'], **{'width': '1000px'})
```





<style  type="text/css" >
    #T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow0_col1 {
            width:  1000px;
        }    #T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow1_col1 {
            width:  1000px;
        }    #T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow2_col1 {
            width:  1000px;
        }    #T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow3_col1 {
            width:  1000px;
        }    #T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow4_col1 {
            width:  1000px;
        }    #T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow5_col1 {
            width:  1000px;
        }    #T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow6_col1 {
            width:  1000px;
        }    #T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow7_col1 {
            width:  1000px;
        }    #T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow8_col1 {
            width:  1000px;
        }    #T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow9_col1 {
            width:  1000px;
        }    #T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow10_col1 {
            width:  1000px;
        }    #T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow11_col1 {
            width:  1000px;
        }    #T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow12_col1 {
            width:  1000px;
        }    #T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow13_col1 {
            width:  1000px;
        }    #T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow14_col1 {
            width:  1000px;
        }    #T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow15_col1 {
            width:  1000px;
        }    #T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow16_col1 {
            width:  1000px;
        }    #T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow17_col1 {
            width:  1000px;
        }    #T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow18_col1 {
            width:  1000px;
        }    #T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow19_col1 {
            width:  1000px;
        }    #T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow20_col1 {
            width:  1000px;
        }    #T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow21_col1 {
            width:  1000px;
        }    #T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow22_col1 {
            width:  1000px;
        }    #T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow23_col1 {
            width:  1000px;
        }    #T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow24_col1 {
            width:  1000px;
        }</style>  
<table id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011d" > 
<thead>    <tr> 
        <th class="blank level0" ></th> 
        <th class="col_heading level0 col0" >Feature</th> 
        <th class="col_heading level0 col1" >Description</th> 
    </tr></thead> 
<tbody>    <tr> 
        <th id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011dlevel0_row0" class="row_heading level0 row0" >0</th> 
        <td id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow0_col0" class="data row0 col0" >loan_amnt</td> 
        <td id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow0_col1" class="data row0 col1" >The listed amount of the loan applied for by the borrower.</td> 
    </tr>    <tr> 
        <th id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011dlevel0_row1" class="row_heading level0 row1" >1</th> 
        <td id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow1_col0" class="data row1 col0" >term</td> 
        <td id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow1_col1" class="data row1 col1" >The number of payments on the loan. Values are in months and can be either 36 or 60.</td> 
    </tr>    <tr> 
        <th id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011dlevel0_row2" class="row_heading level0 row2" >2</th> 
        <td id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow2_col0" class="data row2 col0" >int_rate</td> 
        <td id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow2_col1" class="data row2 col1" >Interest Rate on the loan</td> 
    </tr>    <tr> 
        <th id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011dlevel0_row3" class="row_heading level0 row3" >3</th> 
        <td id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow3_col0" class="data row3 col0" >grade</td> 
        <td id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow3_col1" class="data row3 col1" >LC assigned loan grade</td> 
    </tr>    <tr> 
        <th id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011dlevel0_row4" class="row_heading level0 row4" >4</th> 
        <td id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow4_col0" class="data row4 col0" >emp_length</td> 
        <td id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow4_col1" class="data row4 col1" >Employment length in years. </td> 
    </tr>    <tr> 
        <th id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011dlevel0_row5" class="row_heading level0 row5" >5</th> 
        <td id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow5_col0" class="data row5 col0" >home_ownership</td> 
        <td id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow5_col1" class="data row5 col1" >The home ownership status provided by the borrower during registration or obtained from the credit report.</td> 
    </tr>    <tr> 
        <th id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011dlevel0_row6" class="row_heading level0 row6" >6</th> 
        <td id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow6_col0" class="data row6 col0" >annual_inc</td> 
        <td id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow6_col1" class="data row6 col1" >The self-reported annual income provided by the borrower during registration.</td> 
    </tr>    <tr> 
        <th id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011dlevel0_row7" class="row_heading level0 row7" >7</th> 
        <td id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow7_col0" class="data row7 col0" >verification_status</td> 
        <td id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow7_col1" class="data row7 col1" >Indicates if income was verified by LC, not verified, or if the income source was verified</td> 
    </tr>    <tr> 
        <th id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011dlevel0_row8" class="row_heading level0 row8" >8</th> 
        <td id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow8_col0" class="data row8 col0" >issue_d</td> 
        <td id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow8_col1" class="data row8 col1" >The month which the loan was funded</td> 
    </tr>    <tr> 
        <th id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011dlevel0_row9" class="row_heading level0 row9" >9</th> 
        <td id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow9_col0" class="data row9 col0" >loan_status</td> 
        <td id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow9_col1" class="data row9 col1" >Current status of the loan</td> 
    </tr>    <tr> 
        <th id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011dlevel0_row10" class="row_heading level0 row10" >10</th> 
        <td id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow10_col0" class="data row10 col0" >purpose</td> 
        <td id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow10_col1" class="data row10 col1" >A category provided by the borrower for the loan request. </td> 
    </tr>    <tr> 
        <th id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011dlevel0_row11" class="row_heading level0 row11" >11</th> 
        <td id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow11_col0" class="data row11 col0" >addr_state</td> 
        <td id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow11_col1" class="data row11 col1" >The state provided by the borrower in the loan application</td> 
    </tr>    <tr> 
        <th id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011dlevel0_row12" class="row_heading level0 row12" >12</th> 
        <td id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow12_col0" class="data row12 col0" >dti</td> 
        <td id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow12_col1" class="data row12 col1" >A ratio calculated using the borrower’s total monthly debt payments on the total debt obligations</td> 
    </tr>    <tr> 
        <th id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011dlevel0_row13" class="row_heading level0 row13" >13</th> 
        <td id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow13_col0" class="data row13 col0" >delinq_2yrs</td> 
        <td id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow13_col1" class="data row13 col1" >The number of 30+ days past-due incidences of delinquency in the borrower's credit file for the past 2 years</td> 
    </tr>    <tr> 
        <th id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011dlevel0_row14" class="row_heading level0 row14" >14</th> 
        <td id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow14_col0" class="data row14 col0" >earliest_cr_line</td> 
        <td id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow14_col1" class="data row14 col1" >The month the borrower's earliest reported credit line was opened</td> 
    </tr>    <tr> 
        <th id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011dlevel0_row15" class="row_heading level0 row15" >15</th> 
        <td id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow15_col0" class="data row15 col0" >fico_range_high</td> 
        <td id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow15_col1" class="data row15 col1" >The upper boundary range the borrower’s FICO at loan origination belongs to.</td> 
    </tr>    <tr> 
        <th id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011dlevel0_row16" class="row_heading level0 row16" >16</th> 
        <td id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow16_col0" class="data row16 col0" >fico_range_low</td> 
        <td id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow16_col1" class="data row16 col1" >The lower boundary range the borrower’s FICO at loan origination belongs to.</td> 
    </tr>    <tr> 
        <th id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011dlevel0_row17" class="row_heading level0 row17" >17</th> 
        <td id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow17_col0" class="data row17 col0" >inq_last_6mths</td> 
        <td id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow17_col1" class="data row17 col1" >The number of inquiries in past 6 months (excluding auto and mortgage inquiries)</td> 
    </tr>    <tr> 
        <th id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011dlevel0_row18" class="row_heading level0 row18" >18</th> 
        <td id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow18_col0" class="data row18 col0" >open_acc</td> 
        <td id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow18_col1" class="data row18 col1" >The number of open credit lines in the borrower's credit file.</td> 
    </tr>    <tr> 
        <th id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011dlevel0_row19" class="row_heading level0 row19" >19</th> 
        <td id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow19_col0" class="data row19 col0" >pub_rec</td> 
        <td id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow19_col1" class="data row19 col1" >Number of derogatory public records</td> 
    </tr>    <tr> 
        <th id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011dlevel0_row20" class="row_heading level0 row20" >20</th> 
        <td id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow20_col0" class="data row20 col0" >revol_util</td> 
        <td id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow20_col1" class="data row20 col1" >Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.</td> 
    </tr>    <tr> 
        <th id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011dlevel0_row21" class="row_heading level0 row21" >21</th> 
        <td id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow21_col0" class="data row21 col0" >application_type</td> 
        <td id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow21_col1" class="data row21 col1" >Indicates whether the loan is an individual application or a joint application with two co-borrowers</td> 
    </tr>    <tr> 
        <th id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011dlevel0_row22" class="row_heading level0 row22" >22</th> 
        <td id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow22_col0" class="data row22 col0" >acc_now_delinq</td> 
        <td id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow22_col1" class="data row22 col1" >The number of accounts on which the borrower is now delinquent.</td> 
    </tr>    <tr> 
        <th id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011dlevel0_row23" class="row_heading level0 row23" >23</th> 
        <td id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow23_col0" class="data row23 col0" >tot_coll_amt</td> 
        <td id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow23_col1" class="data row23 col1" >Total collection amounts ever owed</td> 
    </tr>    <tr> 
        <th id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011dlevel0_row24" class="row_heading level0 row24" >24</th> 
        <td id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow24_col0" class="data row24 col0" >tot_cur_bal</td> 
        <td id="T_5afab87a_fe8a_11e8_a9d1_28b2bdea011drow24_col1" class="data row24 col1" >Total current balance of all accounts</td> 
    </tr></tbody> 
</table> 





```python
df = df_raw[["loan_amnt"
             , "term"
             , "int_rate"
             , "grade"
             , "emp_length"
             , "home_ownership"
             , "annual_inc"
             , "verification_status"
             , "issue_d"
             , "loan_status"
             , "purpose"
             , "addr_state"
             , "dti"
             , "delinq_2yrs"
             , "earliest_cr_line"
             , "fico_range_low"
             , "fico_range_high"
             , "inq_last_6mths"
             , "open_acc"
             , "pub_rec"
             , "revol_util"
             , "application_type"
             , "acc_now_delinq"
             , "tot_coll_amt"
             , "tot_cur_bal"]]
```


### Data Cleaning
- Drop rows that are all NA
- Change some columns of string types to numeric types 
- Special treatment for certain columns:
    - There are less than 0.01% missing data in variable revol_util, which is the percentage amount of credit the borrower is using relative to all available revolving credit. We treat these missing observations as bad data, and thus, droppred from the dataset. 



```python
df.dropna(how='all', inplace=True)

df['int_rate'] = df['int_rate'].apply(lambda x: float(x.strip().replace('%', '')))
df['term'] = df['term'].apply(lambda x: int(x.strip().replace('months', '')))

df['temp'] = df['revol_util'].apply(lambda x: 1 if isinstance(x, float) else 0)
df = df[df['temp']==0]
df.drop(['temp'], axis=1, inplace=True)
df['revol_util'] = df['revol_util'].apply(lambda x: float(x.strip().replace('%', '')))
```


## EDA

### Distribution of Loans

<h4> Summary: </h4>
<li> Lending Club business is doing well in term of the incremental <b>mean loan amount</b>. We can see that borrowers are relying more on the platform to finance over the years. </li>
<li> Majority of the <b>loan amount</b> is ranging from 5,000 to 20,000 USD. </li> 




```python
df['issue_d'] = pd.to_datetime(df['issue_d'])
df['year'] = df['issue_d'].apply(lambda x: int(x.year))
```




```python
import warnings
warnings.filterwarnings('ignore')

f, ax = plt.subplots(1, 2, figsize=(20,7))

g1 = sns.barplot(df['year'], df['loan_amnt'], data=df, ax=ax[0])
g1.set_title('Mean Loan Amount Issued', fontsize=16)
g1.set_xlabel('Year', fontsize=16)
g1.set_ylabel('Loan Amount Issued', fontsize=16)
g1.tick_params(labelsize=14)

g2 = sns.distplot(df['loan_amnt'], ax=ax[1])
g2.set_xlabel('Loan Amount', fontsize=16)
g2.set_ylabel('Frequency', fontsize=16)
g2.set_title('Distribution of Loan Amount', fontsize=16)
g2.tick_params(labelsize=14)
```



![png](LC_EDA_files/LC_EDA_13_0.png)


### Good Loans vs Bad Loans

According the lending club website, here is the loan status definition: 
- <b>Current</b>: Loan is up to date on all outstanding payments. 
- <b>In Grace Period</b>: Loan is past due but within the 15-day grace period. 
- <b>Late (16-30)</b>: Loan has not been current for 16 to 30 days. 
- <b>Late (31-120)</b>: Loan has not been current for 31 to 120 days. 
- <b>Fully paid</b>: Loan has been fully repaid, either at the expiration of the 3- or 5-year year term or as a result of a prepayment.
- <b>Default</b>: Loan has not been current for an extended period of time. 
- <b>Charged Off</b>: Loan for which there is no longer a reasonable expectation of further payments. Upon Charge Off, the remaining principal balance of the Note is deducted from the account balance. 

<h4> Summary: </h4>
<ul>
<li> Bad loans consist only <b>12.66%</b> of the total loans in full sample.</li>
<li> The number of bad loans typically tends to move together with the number of good loans over the years, however, that co-movement starts to diverge in recent three years. The reason is because almost of of the loans less than 3-years old are new loans, with most of them in the <b>status of current</b>. In order to reduce this <b>sample bias</b>, we will drop loans in recent years in feature engineering stage.</li>
</ul>



```python
bad_loan = ["Charged Off", 
            "Default", 
            "Does not meet the credit policy. Status:Charged Off", 
            "In Grace Period", 
            "Late (16-30 days)", 
            "Late (31-120 days)"]

df['response'] = df['loan_status'].apply(lambda x: 'Bad Loan' if x in bad_loan else 'Good Loan')
```




```python
f, ax = plt.subplots(1, 2, figsize=(16,8))

colors = ["#3791D7", "#D72626"]
labels = "Good Loans", "Bad Loans"

plt.suptitle('Good Loans vs Bad Loans', fontsize=20)

df["response"].value_counts().plot.pie(
    explode=[0,0.25], 
    autopct='%1.2f%%',
    ax=ax[0], 
    shadow=True, 
    colors=colors, 
    labels=labels, 
    fontsize=16, 
    startangle=70)

ax[0].set_ylabel('% of Condition of Loans', fontsize=14)
ax[0].tick_params(labelsize=14)

palette = ["#3791D7", "#E01E1B"]

g = sns.barplot(x="year", 
                y="loan_amnt", 
                hue="response", 
                data=df, 
                palette=palette, 
                estimator=lambda x: len(x) / len(df) * 100)
ax[1].set_xlabel('Year', fontsize=16)
ax[1].set_ylabel('% Number of Loans', fontsize=16)
ax[1].legend(fontsize=16)
ax[1].tick_params(labelsize=12)
```



![png](LC_EDA_files/LC_EDA_16_0.png)


### Loans by Grade 

<h4> Summary: </h4>
<ul>
<li> Most of the loan are with grades beween B and D. </li>
<li> <b>Generally, the higher the grade, the higher probabilities of bad loans.</b> </li>
</ul>



```python
by_grade = df.groupby(['grade', 'response']).size().unstack()

by_grade_norm = by_grade.copy()
by_grade_norm['sum'] = by_grade_norm['Bad Loan'] + by_grade_norm['Good Loan']
by_grade_norm['Bad Loan'] = by_grade_norm.apply(lambda x: x['Bad Loan'] / x['sum'] * 100, axis=1)
by_grade_norm['Good Loan'] = by_grade_norm.apply(lambda x: x['Good Loan'] / x['sum'] * 100, axis=1)
by_grade_norm.drop(['sum'], inplace=True, axis=1)
```




```python
f, ax = plt.subplots(1, 2, figsize=(20,7))

cmap = plt.cm.coolwarm_r

by_grade.plot(kind='bar', stacked=True, colormap=cmap, ax=ax[0], grid=False)
ax[0].set_title('Number of Loans by Grade', fontsize=16)
ax[0].set_xlabel('Grade', fontsize=16)
ax[0].set_ylabel('Number of Loans', fontsize=16)
ax[0].tick_params(labelsize=16)
ax[0].legend(fontsize=16)

by_grade_norm.plot(kind='bar', stacked=True, ax=ax[1], colormap=cmap)
ax[1].set_title('Percentage Number of Loans by Sub-Grade', fontsize=16)
ax[1].set_xlabel('Grade', fontsize=16)
ax[1].set_ylabel('% of Loans', fontsize=16)
ax[1].tick_params(labelsize=16)
ax[1].legend(fontsize=16)
plt.show()
```



![png](LC_EDA_files/LC_EDA_19_0.png)


### Loans by Purpose 

<h4> Findings Summary: </h4>
<ul>
<li> <b>Debt consolidation</b> is the biggest purpose for the loans from the borrowers. </li>
<li> Even though <b>education</b> as a purpose of loans has the smallest percentage, the default rate is the <b>highest</b> among all purposes, followed by <b>small business</b>.</li>
</ul>



```python
by_purpose = df.groupby(['response', 'purpose']).size().unstack().T
by_purpose['sum'] = by_purpose['Bad Loan'] + by_purpose['Good Loan']
by_purpose.sort_values(['sum'], inplace=True)
by_purpose.drop(['sum'], axis=1, inplace=True)

by_purpose_norm = df.groupby(['response', 'purpose']).size().unstack().apply(lambda x: x/x.sum() * 100).T
by_purpose_norm.sort_values(['Bad Loan'], inplace=True)
```




```python
f, ax = plt.subplots(1, 2, figsize=(20,7))

cmap = plt.cm.coolwarm_r

by_purpose.plot(kind='bar', stacked=True, colormap=cmap, ax=ax[0], grid=False)
ax[0].set_title('Number of Loans by Purpose', fontsize=16)
ax[0].set_xlabel('Purpose', fontsize=16)
ax[0].set_ylabel('Number of Loans', fontsize=16)
ax[0].tick_params(labelsize=16)
ax[0].legend(fontsize=16)

by_purpose_norm.plot(kind='bar', stacked=True, ax=ax[1], colormap=cmap)
ax[1].set_title('Percentage Number of Loans by Purpose', fontsize=16)
ax[1].set_xlabel('Purpose', fontsize=16)
ax[1].set_ylabel('% of Loans', fontsize=16)
ax[1].tick_params(labelsize=16)
ax[1].legend(fontsize=16)
plt.show()
```



![png](LC_EDA_files/LC_EDA_22_0.png)


### Loans by State 

<h4> Summary: </h4>
<ul>
<li> The loan defulat rates are very marginally different among US states, and could add more noise than information.</li>
<li> IOWA has the highest default rate among all states, but further investigation shows that there are only 14 loans in Iowa in full history, so we'd better not think too much into it. </li>
</ul>



```python
by_state = df.groupby(['response', 'addr_state']).size().unstack().T
by_state['bad_loan_ptg'] = by_state.apply(lambda x: x['Bad Loan'] / (x['Bad Loan'] + x['Good Loan']), axis=1)
by_state.reset_index(inplace=True)
```




```python
for col in by_state.columns:
    by_state[col] = by_state[col].astype(str)
    
scl = [[0.0, 'rgb(202, 202, 202)'],[0.2, 'rgb(253, 205, 200)'],[0.4, 'rgb(252, 169, 161)'],\
            [0.6, 'rgb(247, 121, 108  )'],[0.8, 'rgb(232, 70, 54)'],[1.0, 'rgb(212, 31, 13)']]

by_state['text'] = by_state['addr_state'] 

data = [dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = by_state['addr_state'],
        z = by_state['bad_loan_ptg'], 
        locationmode = 'USA-states',
        text = by_state['text'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "%")
        ) ]


layout = dict(
    title = 'Default Rates by States',
    geo = dict(
        scope = 'usa',
        projection=dict(type='albers usa'),
        showlakes = True,
        lakecolor = 'rgb(255, 255, 255)')
)

fig = dict(data=data, layout=layout)
iplot(fig, filename='d3-cloropleth-map')
```



<div id="9361eec1-85b7-4116-8290-dfdbe07abead" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";Plotly.newPlot("9361eec1-85b7-4116-8290-dfdbe07abead", [{"autocolorscale": false, "colorbar": {"title": "%"}, "colorscale": [[0.0, "rgb(202, 202, 202)"], [0.2, "rgb(253, 205, 200)"], [0.4, "rgb(252, 169, 161)"], [0.6, "rgb(247, 121, 108  )"], [0.8, "rgb(232, 70, 54)"], [1.0, "rgb(212, 31, 13)"]], "locationmode": "USA-states", "locations": ["AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "HI", "IA", "ID", "IL", "IN", "KS", "KY", "LA", "MA", "MD", "ME", "MI", "MN", "MO", "MS", "MT", "NC", "ND", "NE", "NH", "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VA", "VT", "WA", "WI", "WV", "WY"], "marker": {"line": {"color": "rgb(255,255,255)", "width": 2}}, "text": ["AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "HI", "IA", "ID", "IL", "IN", "KS", "KY", "LA", "MA", "MD", "ME", "MI", "MN", "MO", "MS", "MT", "NC", "ND", "NE", "NH", "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VA", "VT", "WA", "WI", "WV", "WY"], "z": ["0.12780448717948717", "0.15077982894074848", "0.14926209602092286", "0.12631662154417722", "0.12967110811266655", "0.10214804512440118", "0.10656782802075612", "0.09625246548323471", "0.12728470278280915", "0.13437976547299785", "0.11658628043108013", "0.13675298804780878", "0.21428571428571427", "0.07914804102024717", "0.10920760936430006", "0.1308897562079303", "0.10384934921074494", "0.13107668398815017", "0.14938899368490116", "0.12277867528271405", "0.134125469675211", "0.06524110844425049", "0.1287712215694049", "0.12666417152328505", "0.13198133419423003", "0.14224210704035117", "0.10823529411764705", "0.13403594108904202", "0.09761976498945465", "0.12048528796541626", "0.09141168620722524", "0.1326765021185359", "0.1372670258811059", "0.14309500489715965", "0.13940006472896166", "0.126660374966546", "0.14582586886420637", "0.0950491907331006", "0.13032598682853724", "0.11281559603707254", "0.10367523968954497", "0.13709866416686198", "0.13073208769456857", "0.12447214159694953", "0.12026883622214361", "0.12913356733789028", "0.08566508824795523", "0.10454463399223757", "0.113722004976893", "0.10306588388780169", "0.11064207953788047"], "type": "choropleth", "uid": "1139a101-f418-4dcf-9c38-f1a2042cad29"}], {"geo": {"lakecolor": "rgb(255, 255, 255)", "projection": {"type": "albers usa"}, "scope": "usa", "showlakes": true}, "title": "Default Rates by States"}, {"showLink": true, "linkText": "Export to plot.ly", "plotlyServerURL": "https://plot.ly"})});</script><script type="text/javascript">window.addEventListener("resize", function(){window._Plotly.Plots.resize(document.getElementById("9361eec1-85b7-4116-8290-dfdbe07abead"));});</script>


### Loans by Debt-to-Income Ratio 

<h4> Summary: </h4>
<ul>
<li> <b>Bad loans have higher debt-to-income ratios. </b></li>
<li> By visual inspection, good loans have DTI ratio around 17, while bad loans around 20. </li>
</ul>



```python
dti_good = list((df.dropna(subset=['dti']))[df['response']=='Good Loan']['dti'])
dti_bad = list((df.dropna(subset=['dti']))[df['response']=='Bad Loan']['dti'])

dti_good_low, dti_good_high = np.percentile(dti_good, 0), np.percentile(dti_good, 99)
dti_bad_low, dti_bad_high = np.percentile(dti_bad, 0), np.percentile(dti_bad, 99)

dti_good_trim = [x for x in dti_good if x > dti_good_low and x < dti_good_high]
dti_bad_trim = [x for x in dti_bad if x > dti_bad_low and x < dti_bad_high]
```




```python
f, ax = plt.subplots(figsize=(20,6))

colors = ["#3791D7", "#D72626"]

sns.distplot(dti_good_trim, ax=ax, color=colors[0], label='Good Loan')
sns.distplot(dti_bad_trim, ax=ax, color=colors[1], label='Bad Loan')
plt.title("Loans by Debt-to-Income Ratio", fontsize=16)
plt.xlabel('Debt-to-Income Ratio', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
plt.legend(fontsize=16)
plt.tick_params(labelsize=16)
plt.show()
```



![png](LC_EDA_files/LC_EDA_28_0.png)


### Loans by FICO score 

<h4> Summary: </h4>
<ul>
<li> The distributions of FICO scores of good and bad loans are similar overall, with majority of scores between 670 to 720. However, <b>good loans have higher FICO scores on the right tail, above 740. </b> This makes economic sense, because good loan applicants with high FICO scores will tend to pay the monthly installment on time.</li>
</ul>



```python
df['fico'] = (df['fico_range_low'] + df['fico_range_high']) / 2
```




```python
fico_good = list(df[df['response']=='Good Loan']['fico'])
fico_bad = list(df[df['response']=='Bad Loan']['fico'])

fico_good_low, fico_good_high = np.percentile(fico_good, 0.5), np.percentile(fico_good, 99)
fico_bad_low, fico_bad_high = np.percentile(fico_bad, 0.5), np.percentile(fico_bad, 99)

fico_good_trim = [x for x in fico_good if x > fico_good_low and x < fico_good_high]
fico_bad_trim = [x for x in fico_bad if x > fico_bad_low and x < fico_bad_high]
```




```python
f, ax = plt.subplots(figsize=(20,6))

colors = ["#3791D7", "#D72626"]

sns.distplot(fico_good_trim, hist=False, bins=50, ax=ax, color=colors[0], label='Good Loan')
sns.distplot(fico_bad_trim, hist=False, bins=50, ax=ax, color=colors[1], label='Bad Loan')
plt.title("Loans by FICO score", fontsize=16)
plt.xlabel('FICO score', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
plt.legend(fontsize=16)
plt.tick_params(labelsize=16)
plt.show()
```



![png](LC_EDA_files/LC_EDA_32_0.png)




```python
df.to_csv("data/output_eda.csv", index=False)
```

