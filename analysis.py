import pandas as pd 
from sklearn.preprocessing import LabelEncoder
import os
import pathlib
from matplotlib import pyplot as plt
import seaborn as sns

if __name__ == "__main__":
     parent_path = pathlib.Path(__file__).parent.absolute()
     local_path='/HW_F24_MLT/loan_prediction/loan_prediction.csv'
     df = pd.read_csv(os.path.join(parent_path,'HW_F24_MLT/loan_prediction/loan_prediction.csv'))



     df=df.drop(['Loan_ID'],axis=1 )  # delete the cloumn load_id 

     print(df.head())

     #  Gender Married Dependents     Education Self_Employed  ApplicantIncome  CoapplicantIncome  LoanAmount  Loan_Amount_Term  Credit_History Property_Area Loan_Status
     # 0   Male      No          0      Graduate            No             5849                0.0         NaN             360.0             1.0         Urban           Y
     # 1   Male     Yes          1      Graduate            No             4583             1508.0       128.0             360.0             1.0         Rural           N
     # 2   Male     Yes          0      Graduate           Yes             3000                0.0        66.0             360.0             1.0         Urban           Y
     # 3   Male     Yes          0  Not Graduate            No             2583             2358.0       120.0             360.0             1.0         Urban           Y
     # 4   Male      No          0      Graduate            No             6000                0.0       141.0             360.0             1.0         Urban           Y

     df.info()

     # <class 'pandas.core.frame.DataFrame'>
     # Index: 614 entries, 0 to 613
     # Data columns (total 12 columns):
     #  #   Column             Non-Null Count  Dtype  
     # ---  ------             --------------  -----  
     #  0   Gender             601 non-null    object      => missing values
     #  1   Married            611 non-null    object      => missing values
     #  2   Dependents         599 non-null    object      => missing values
     #  3   Education          614 non-null    object
     #  4   Self_Employed      582 non-null    object      => missing values
     #  5   ApplicantIncome    614 non-null    int64
     #  6   CoapplicantIncome  614 non-null    float64
     #  7   LoanAmount         592 non-null    float64     => missing values
     #  8   Loan_Amount_Term   600 non-null    float64     => missing values
     #  9   Credit_History     564 non-null    float64     => missing values
     #  10  Property_Area      614 non-null    object
     #  11  Loan_Status        614 non-null    object
     # dtypes: float64(4), int64(1), object(7)
     # memory usage: 62.4+ KB

     ############################################################################################################################
     #المعالجة الأولية للبيانات

     df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
     df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
     df['Dependents']= df['Dependents'].fillna(df['Dependents'].mode()[0])
     df['Self_Employed']= df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
     df['LoanAmount']= df['LoanAmount'].fillna(df['LoanAmount'].median())
     df['Loan_Amount_Term']= df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean())
     df['Credit_History']= df['Credit_History'].fillna(df['Credit_History'].mode()[0])


     print(df.info())
     ### data after proccessing the messing value 
     # Data columns (total 12 columns):
     #  #   Column             Non-Null Count  Dtype
     # ---  ------             --------------  -----
     #  0   Gender             614 non-null    object       
     #  1   Married            614 non-null    object
     #  2   Dependents         614 non-null    object
     #  3   Education          614 non-null    object
     #  4   Self_Employed      614 non-null    object
     #  5   ApplicantIncome    614 non-null    int64
     #  6   CoapplicantIncome  614 non-null    float64
     #  7   LoanAmount         614 non-null    float64
     #  8   Loan_Amount_Term   614 non-null    float64
     #  9   Credit_History     614 non-null    float64
     #  10  Property_Area      614 non-null    object
     #  11  Loan_Status        614 non-null    object
     # dtypes: float64(4), int64(1), object(7)
     # memory usage: 62.4+ KB

     #######
     ###   clomuns(Gender,Married,Dependents, Education,Self_Employed,Credit_History,Property_Area,Loan_Status )  => categrory values
     ###   columns(ApplicantIncome,CoapplicantIncome,LoanAmount, Loan_Amount_Term )  => numerical values
     #######

     print(df.describe())

     ## ###########################statistical describe##############################################
     #        ApplicantIncome  CoapplicantIncome  LoanAmount  Loan_Amount_Term  Credit_History
     # count       614.000000         614.000000  614.000000        614.000000      614.000000
     # mean       5403.459283        1621.245798  145.752443        342.000000        0.855049
     # std        6109.041673        2926.248369   84.107233         64.372489        0.352339
     # min         150.000000           0.000000    9.000000         12.000000        0.000000
     # 25%        2877.500000           0.000000  100.250000        360.000000        1.000000
     # 50%        3812.500000        1188.500000  128.000000        360.000000        1.000000
     # 75%        5795.000000        2297.250000  164.750000        360.000000        1.000000
     # max       81000.000000       41667.000000  700.000000        480.000000        1.000000

     ############################################################################################################################

     ############################################################################################################################

     #استبدال القيم النصية بقيم رقمية
     le_gender = LabelEncoder()
     le_gender.fit(df['Gender'])

     le_married = LabelEncoder()
     le_married.fit(df['Married'])

     le_dependents = LabelEncoder()
     le_dependents.fit(df['Dependents'])

     le_education = LabelEncoder()
     le_education.fit(df['Education'])


     le_self_employed = LabelEncoder()
     le_self_employed.fit(df['Self_Employed'])

     le_property_area = LabelEncoder()
     le_property_area.fit(df['Property_Area'])

     le_loan_status = LabelEncoder()
     le_loan_status.fit(df['Loan_Status'])

     df['Gender']= le_gender.transform(df['Gender'])
     df['Married']=le_married.transform(df['Married'])
     df['Dependents']= le_dependents.transform(df['Dependents'])
     df['Education']=le_education.transform(df['Education'])
     df['Self_Employed']=le_self_employed.transform(df['Self_Employed'])
     df['Property_Area']=le_property_area.transform(df['Property_Area'])
     df['Loan_Status']=le_loan_status.transform(df['Loan_Status'])

     print(df.info())

     #  #   Column             Non-Null Count  Dtype
     # ---  ------             --------------  -----
     #  0   Gender             614 non-null    int64
     #  1   Married            614 non-null    int64
     #  2   Dependents         614 non-null    int64
     #  3   Education          614 non-null    int64
     #  4   Self_Employed      614 non-null    int64
     #  5   ApplicantIncome    614 non-null    int64
     #  6   CoapplicantIncome  614 non-null    float64
     #  7   LoanAmount         614 non-null    float64
     #  8   Loan_Amount_Term   614 non-null    float64
     #  9   Credit_History     614 non-null    float64
     #  10  Property_Area      614 non-null    int64
     #  11  Loan_Status        614 non-null    int64

     print(df.head(10))

     ########################################################################################################################
     # حساب الترابط بين الأعمد ة
     dfcorrelations = df.corr(method='pearson')
     # طباعة عمود التصنيف
     print("####  correlation ####")
     print(dfcorrelations['Loan_Status'])
     plt.figure(figsize=(8, 6)) 
     sns.heatmap(dfcorrelations, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
     plt.title('Correlation Matrix')
     plt.show()

     ####  correlation with loan_status ####
     # Gender               0.017987
     # Married              0.091478
     # Dependents           0.010118
     # Education           -0.085884
     # Self_Employed       -0.003700
     # ApplicantIncome     -0.004710
     # CoapplicantIncome   -0.059187
     # LoanAmount          -0.033214
     # Loan_Amount_Term    -0.020974
     # Credit_History       0.540556
     # Property_Area        0.032112
     # Loan_Status          1.000000


     # ApplicantIncome is highly overall correlated with LoanAmount
     # Credit_History is highly overall correlated with Loan_Status

     ###########################################################################################################################

     ####  عرض توزع قيم الأعمدة   
     for label in df.columns:
          sns.boxplot(df[label])
          plt.show()


     ###############################################################################################################

     #رسم مخططات التكرار والكثافة للأعمدة حسب قيمة التصنيف(حالة القرض)
     for label in df.columns:
          plt.subplot(1,2,1)
          plt.hist(df[df['Loan_Status']==0][label], color='red', label='No', alpha=0.5)
          plt.hist(df[df['Loan_Status']==1][label], color='blue', label='Yes', alpha=0.5)
          plt.title(label)
          plt.ylabel('Count')
          plt.xlabel(label)
          plt.legend()

          plt.subplot(1,2,2)
          plt.hist(df[df['Loan_Status']==0][label], color='red', label='No', alpha=0.5, density=True)
          plt.hist(df[df['Loan_Status']==1][label], color='blue', label='Yes', alpha=0.5, density=True)
          plt.title(label)
          plt.ylabel('Probability')
          plt.xlabel(label)
          plt.legend()
          plt.show()


     ############################################

     ## رسم مخططات العلاقة بين مختلف الأعمدة 
     for i in range(len(df.columns)-1):
          for j in range(i+1, len(df.columns)-1):
               x_label = df.columns[i]
               y_label = df.columns[j]
               sns.scatterplot(x=x_label, y=y_label, data=df, hue='Loan_Status')
               plt.show()
               

     ############################################################################################################################

     ##profile report

     # profile = ProfileReport(df, title='loan_prediction', explorative= True)
     # profile.to_file('loan_prediction.html')   #if it don't work run -> pip install setuptools


     ############################################################################################################################



def proccess_data(df):
     if not isinstance(df, pd.DataFrame):    
         df = pd.DataFrame(df)     
     if 'Loan_ID' in df.columns:
         df=df.drop(['Loan_ID'],axis=1 )
     if len(df.columns) == 11:
          if 'Gender' not in df.columns:
               df.columns=['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area']
     elif len(df.columns) == 12:  
          if 'Gender' not in df.columns: 
               df.columns=['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area','Loan_Status']  
     else:
          raise Exception("The Dataframe shape must be (:, 11) or (: , 12)") 
     
         
     #المعالجة الأولية للبيانات
     df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
     df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
     df['Dependents']= df['Dependents'].fillna(df['Dependents'].mode()[0])
     df['Self_Employed']= df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
     df['LoanAmount']= df['LoanAmount'].fillna(df['LoanAmount'].median())
     df['Loan_Amount_Term']= df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean())
     df['Credit_History']= df['Credit_History'].fillna(df['Credit_History'].mode()[0])
     #استبدال القيم النصية بقيم رقمية
     le_gender = LabelEncoder()
     le_gender.fit(df['Gender'])
     le_married = LabelEncoder()
     le_married.fit(df['Married'])
     le_dependents = LabelEncoder()
     le_dependents.fit(df['Dependents'])
     le_education = LabelEncoder()
     le_education.fit(df['Education'])
     le_self_employed = LabelEncoder()
     le_self_employed.fit(df['Self_Employed'])
     le_property_area = LabelEncoder()
     le_property_area.fit(df['Property_Area'])
     df['Gender']= le_gender.transform(df['Gender'])
     df['Married']=le_married.transform(df['Married'])
     df['Dependents']= le_dependents.transform(df['Dependents'])
     df['Education']=le_education.transform(df['Education'])
     df['Self_Employed']=le_self_employed.transform(df['Self_Employed'])
     df['Property_Area']=le_property_area.transform(df['Property_Area'])

     if 'Loan_Status' in df.columns:
       le_loan_status = LabelEncoder()
       le_loan_status.fit(df['Loan_Status'])
       df['Loan_Status']=le_loan_status.transform(df['Loan_Status'])

     return df

###########################################################################################################################

def plot_analysis(df):      
     df= proccess_data(df)
     os.makedirs(os.path.join(pathlib.Path(__file__).parent.absolute(),'analysis'), exist_ok=True)
     ##الترابط
     dfcorrelations = df.corr(method='pearson')
     plt.figure(figsize=(8, 6)) 
     sns.heatmap(dfcorrelations, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
     plt.title('Correlation Matrix')
     plt.savefig(os.path.join(pathlib.Path(__file__).parent.absolute(),'analysis' ,"correlation matrix.png"))
     plt.close()
     ####  عرض توزع قيم الأعمدة   
     for label in df.columns:
          plt.figure(figsize=(8, 6)) 
          sns.boxplot(df[label])
          plt.savefig(os.path.join(pathlib.Path(__file__).parent.absolute(),'analysis' ,f"{label+'distribution'}.png"))
          plt.close()


     #رسم مخططات التكرار والكثافة للأعمدة حسب قيمة التصنيف(حالة القرض)
     for label in df.columns:
          plt.subplot(1,2,1)
          plt.hist(df[df['Loan_Status']==0][label], color='red', label='No', alpha=0.5)
          plt.hist(df[df['Loan_Status']==1][label], color='blue', label='Yes', alpha=0.5)
          plt.title(label)
          plt.ylabel('Count')
          plt.xlabel(label)
          plt.legend()

          plt.subplot(1,2,2)
          plt.hist(df[df['Loan_Status']==0][label], color='red', label='No', alpha=0.5, density=True)
          plt.hist(df[df['Loan_Status']==1][label], color='blue', label='Yes', alpha=0.5, density=True)
          plt.title(label)
          plt.ylabel('Probability')
          plt.xlabel(label)
          plt.legend()
          plt.savefig(os.path.join(pathlib.Path(__file__).parent.absolute(),'analysis' ,f"{label}.png"))
          plt.close()


     ## رسم مخططات العلاقة بين مختلف الأعمدة 
     for i in range(len(df.columns)-1):
          for j in range(i+1, len(df.columns)-1):
               x_label = df.columns[i]
               y_label = df.columns[j]
               sns.scatterplot(x=x_label, y=y_label, data=df, hue='Loan_Status')
               plt.savefig(os.path.join(pathlib.Path(__file__).parent.absolute(),'analysis' ,f"{df.columns[i]+' vs '+df.columns[j]}.png"))
               plt.close()
     
     # رسم توزع الأصناف
     count_values=df['Loan_Status'].value_counts() 
     labels = count_values.index.to_list() 
     plt.title('Original classes distribution')
     plt.pie(x = count_values, labels = labels, autopct ='%1.1f%%' )
     plt.savefig(os.path.join(pathlib.Path(__file__).parent.absolute(),'analysis' ,f"Original classes distribution.png"))
     plt.close()
                    
###########################################################################################################################


def generate_data_profile(df):
      from ydata_profiling import ProfileReport      
      profile = ProfileReport(df, title='loan_prediction', explorative= True)
      profile.to_file(os.path.join(pathlib.Path(__file__).parent.absolute(),'loan_prediction.html'))   #if it don't work run -> pip install setuptools


###########################################################################################################################

def analysis_info(df):
     df = proccess_data(df)
     return {'correlation':df.corr(method='pearson'), 'sample': df.head(), 'info': df.info, 'statistical': df.describe()}

###########################################################################################################################

def live_analysis(df):
     print("##data befor proccessing##")
     print(df.head())
     print(df.info())
     df = proccess_data(df)
     print("##data befor proccessing##")
     print(df.head())
     print(df.info())
     # حساب الترابط بين الأعمد ة
     dfcorrelations = df.corr(method='pearson')
     # طباعة عمود التصنيف
     print("####  correlation ####")
     print(dfcorrelations['Loan_Status'])
     plt.figure(figsize=(8, 6)) 
     sns.heatmap(dfcorrelations, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
     plt.title('Correlation Matrix')
     plt.show()     
     ####  عرض توزع قيم الأعمدة   
     for label in df.columns:
          sns.boxplot(df[label])
          plt.show()
        #رسم مخططات التكرار والكثافة للأعمدة حسب قيمة التصنيف(حالة القرض)
     for label in df.columns:
          plt.subplot(1,2,1)
          plt.hist(df[df['Loan_Status']==0][label], color='red', label='No', alpha=0.5)
          plt.hist(df[df['Loan_Status']==1][label], color='blue', label='Yes', alpha=0.5)
          plt.title(label)
          plt.ylabel('Count')
          plt.xlabel(label)
          plt.legend()
          plt.subplot(1,2,2)
          plt.hist(df[df['Loan_Status']==0][label], color='red', label='No', alpha=0.5, density=True)
          plt.hist(df[df['Loan_Status']==1][label], color='blue', label='Yes', alpha=0.5, density=True)
          plt.title(label)
          plt.ylabel('Probability')
          plt.xlabel(label)
          plt.legend()
          plt.show()
     
     ## رسم مخططات العلاقة بين مختلف الأعمدة 
     for i in range(len(df.columns)-1):
          for j in range(i+1, len(df.columns)-1):
               x_label = df.columns[i]
               y_label = df.columns[j]
               sns.scatterplot(x=x_label, y=y_label, data=df, hue='Loan_Status')
               plt.show()



