import pandas as pd 
from sklearn.model_selection import train_test_split 
from imblearn.over_sampling import SMOTE 
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report,confusion_matrix   
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import CategoricalNB,GaussianNB
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
import pickle
from analysis import proccess_data
import os
import pathlib

def live_model_run(df):
     
    df= proccess_data(df)
    
    ############################################
    #تحديد الدخل والخرج

    X= df[['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area']]
    y= df['Loan_Status']


    # ############################################################################################################################
    # scale dataset

    scaler= StandardScaler()
    X[['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']]= scaler.fit_transform(X[['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']])

    ##################################################################################################################################
    ############################################################################################################################

    #تقسيم البيانات إلى بيانات تدريب وبيانات اختبار
    # التقسيم إلى تدريب واختبار
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    ############################################################################################################################
    # موازنة الأصناف
    # معاينة نسب توزع الأصناف

     # حساب عدد القيم من كل صنف
    count_values=y_train.value_counts() 
    # قائمة الفهارس
    labels = count_values.index.to_list() 
    # عنوان الرسم
    plt.title('Original classes distribution')
     # رسم الفطيرة
    plt.pie(x = count_values, labels = labels, autopct ='%1.1f%%' ) # إظهار الرس م
    plt.show()

    #اجراء موازنة للأصناف
    
    oversample = SMOTE() 
    X_train, y_train = oversample.fit_resample(X_train, y_train) # حساب عد د القيم من كل صنف
    count_values=y_train.value_counts() # قائمة الفهار س
    labels = count_values.index.to_list() # عنوان الرسم
    plt.title('New classes distribution') # رسم الفطيرة
    plt.pie(x = count_values, labels = labels, autopct = '%1.1f%%' ) # إظهار الرسم
    plt.show()

    ##################################################################################################################################
    # # # التنبؤ باستخدام الجار الأقرب
    knn_model = KNeighborsClassifier(5)

    knn_model.fit(X_train, y_train)

    y_pred = knn_model.predict(X_test)

    #مقاييس تقييم النموذج
    # حساب مصفوفة الارتباك
    cf_matrix_knn = confusion_matrix(y_test, y_pred)
    print ('Confusion Matrix for knn model:')
    print (cf_matrix_knn) 

    # حساب مقاييس الأدا ء
    accuracy_knn=accuracy_score(y_test, y_pred)
    print ('Accuracy Score knn model :{:.2f}'.format(accuracy_knn*100))

    # micro
    micro_Precision_Score_knn=precision_score(y_test, y_pred, average='micro')
    micro_Recall_Score_knn=recall_score(y_test, y_pred, average='micro')
    micro_F1_Score_knn=f1_score(y_test, y_pred, average='micro')
    print ('micro Precision Score knn model:{:.2f}'.format(micro_Precision_Score_knn*100)) 
    print ('micro Recall Score knn model:{:.2f}'.format(micro_Recall_Score_knn*100))
    print ('micro F1 Score knn model :{:.2f}'.format(micro_F1_Score_knn*100))

    # macro
    macro_Precision_Score_knn=precision_score(y_test, y_pred, average='macro')
    macro_Recall_Score_knn=recall_score(y_test, y_pred, average='macro')
    macro_F1_Score_knn=f1_score(y_test, y_pred, average='macro')
    print ('macro Precision Score knn model:{:.2f}'.format(macro_Precision_Score_knn*100)) 
    print ('macro Recall Score knn model :{:.2f}'.format(macro_Recall_Score_knn*100))
    print ('macro F1 Score  knn model:{:.2f}'.format(macro_F1_Score_knn*100))

    # weighted
    weighted_Precision_Score_knn=precision_score(y_test, y_pred, average='weighted')
    weighted_Recall_Score_knn=recall_score(y_test, y_pred, average='weighted')
    weighted_F1_Score_knn=f1_score(y_test, y_pred, average='weighted')
    print ('weighted Precision Score knn model:{:.2f}'.format(weighted_Precision_Score_knn*100)) 
    print ('weighted Recall Score knn model :{:.2f}'.format(weighted_Recall_Score_knn*100))
    print ('weighted F1 Score knn model:{:.2f}'.format(weighted_F1_Score_knn*100))

    ## classfication report

    knn_report=classification_report(y_test,y_pred)
    print("############### knn_report #######################")
    print(knn_report)

####################################################################################################################

    #التصنيف باستخدام الانحدار اللوجستي


    logistic_model =LogisticRegression()
    logistic_model.fit(X_train , y_train)
    # التنبؤ مع بيانات الاختبار
    ypred= logistic_model.predict(X_test)
    # حساب مصفوفة الارتباك
    cf_matrix_logistic = confusion_matrix(y_test, ypred)
    print ('Confusion Matrix for tree model:')
    print (cf_matrix_logistic ) 

    # حساب مقاييس الأدا ء
    accuracy_logistic=accuracy_score(y_test, ypred)
    print ('Accuracy Score logistic model :{:.2f}'.format(accuracy_logistic*100))
    # micro
    micro_Precision_Score_logistic=precision_score(y_test, ypred, average='micro')
    micro_Recall_Score_logistic_model=recall_score(y_test, ypred, average='micro')
    micro_F1_Score_logistic_model=f1_score(y_test, ypred, average='micro')
    print ('micro Precision Score logistic model:{:.2f}'.format(micro_Precision_Score_logistic*100)) 
    print ('micro Recall Score logistic model:{:.2f}'.format(micro_Recall_Score_logistic_model*100))
    print ('micro F1 Score logistic model :{:.2f}'.format(micro_F1_Score_logistic_model*100))
    # macro
    macro_Precision_Score_logistic_model=precision_score(y_test, ypred, average='macro')
    macro_Recall_Score_logistic_model=recall_score(y_test, ypred, average='macro')
    macro_F1_Score_logistic_model=f1_score(y_test, ypred, average='macro')
    print ('macro Precision Score logistic model:{:.2f}'.format(macro_Precision_Score_logistic_model*100))
    print ('macro F1 Score  logistic model:{:.2f}'.format(macro_F1_Score_logistic_model*100))
    # weighted
    weighted_Precision_Score_logistic_model=precision_score(y_test, ypred, average='weighted')
    weighted_Recall_Score_logistic_model=recall_score(y_test, ypred, average='weighted')
    weighted_F1_Score_logistic_model=f1_score(y_test, ypred, average='weighted')
    print ('weighted Precision Score logistic model:{:.2f}'.format(weighted_Precision_Score_logistic_model*100)) 
    print ('weighted Recall Score logistic model :{:.2f}'.format(weighted_Recall_Score_logistic_model*100))
    print ('weighted F1 Score logistic model:{:.2f}'.format(weighted_F1_Score_logistic_model*100))


    logistic_report=classification_report(y_test, ypred)
    print("############### logistic_model_report #######################")
    print(logistic_report)

    ############################################################################################################################
    #التصنيف باستخدام Support Vector Machines

    svm_model = SVC()
    svm_model.fit(X_train,y_train)

    y_pred=svm_model.predict(X_test)

    # حساب مصفوفة الارتباك
    cf_matrix_svm = confusion_matrix(y_test, y_pred)
    print ('Confusion Matrix for svm:')
    print (cf_matrix_svm) 


    # حساب مقاييس الأدا ء
    accuracy_svm=accuracy_score(y_test, y_pred)
    print ('Accuracy Score SVM model :{:.2f}'.format(accuracy_svm*100))
    # micro
    micro_Precision_Score_svm=precision_score(y_test, y_pred, average='micro')
    micro_Recall_Score_svm_model=recall_score(y_test, y_pred, average='micro')
    micro_F1_Score_svm_model=f1_score(y_test, y_pred, average='micro')
    print ('micro Precision Score svm model:{:.2f}'.format(micro_Precision_Score_svm*100)) 
    print ('micro Recall Score svm model:{:.2f}'.format(micro_Recall_Score_svm_model*100))
    print ('micro F1 Score svm model :{:.2f}'.format(micro_F1_Score_svm_model*100))
    # macro
    macro_Precision_Score_svm_model=precision_score(y_test, y_pred, average='macro')
    macro_Recall_Score_svm_model=recall_score(y_test, y_pred, average='macro')
    macro_F1_Score_svm_model=f1_score(y_test, y_pred, average='macro')
    print ('macro Precision Score svm model:{:.2f}'.format(macro_Precision_Score_svm_model*100))
    print ('macro F1 Score  svm model:{:.2f}'.format(macro_F1_Score_svm_model*100))
    # weighted
    weighted_Precision_Score_svm_model=precision_score(y_test, y_pred, average='weighted')
    weighted_Recall_Score_svm_model=recall_score(y_test, y_pred, average='weighted')
    weighted_F1_Score_svm_model=f1_score(y_test, y_pred, average='weighted')
    print ('weighted Precision Score svm model:{:.2f}'.format(weighted_Precision_Score_svm_model*100)) 
    print ('weighted Recall Score svm model :{:.2f}'.format(weighted_Recall_Score_svm_model*100))
    print ('weighted F1 Score svm model:{:.2f}'.format(weighted_F1_Score_svm_model*100))


    svm_report=classification_report(y_test, y_pred)
    print("############### svm_model_report #######################")
    print(svm_report)

    ###############################################################################################################################
    #التصنيف باستخدام شجرة القرار

    tree_model = DecisionTreeClassifier(criterion="entropy")
    tree_model.fit(X_train, y_train)

    # #تمثيل الشجرة
    # text_representation = tree.export_text(tree_model)
    # print(text_representation)

    # تنبؤ النموذ ج
    ypred=tree_model.predict(X_test) 


    #مقاييس تقييم النموذج
    # حساب مصفوفة الارتباك
    cf_matrix_tree = confusion_matrix(y_test, ypred)
    print ('Confusion Matrix for tree model:')
    print (cf_matrix_tree) 

    # حساب مقاييس الأدا ء
    accuracy_tree=accuracy_score(y_test, ypred)
    print ('Accuracy Score tree model :{:.2f}'.format(accuracy_tree*100))

    # micro
    micro_Precision_Score_tree=precision_score(y_test, ypred, average='micro')
    micro_Recall_Score_tree_model=recall_score(y_test, ypred, average='micro')
    micro_F1_Score_tree_model=f1_score(y_test, ypred, average='micro')
    print ('micro Precision Score tree model:{:.2f}'.format(micro_Precision_Score_tree*100)) 
    print ('micro Recall Score tree model:{:.2f}'.format(micro_Recall_Score_tree_model*100))
    print ('micro F1 Score tree model :{:.2f}'.format(micro_F1_Score_tree_model*100))

    # macro
    macro_Precision_Score_tree_model=precision_score(y_test, ypred, average='macro')
    macro_Recall_Score_tree_model=recall_score(y_test, ypred, average='macro')
    macro_F1_Score_tree_model=f1_score(y_test, ypred, average='macro')
    print ('macro Precision Score tree model:{:.2f}'.format(macro_Precision_Score_tree_model*100)) 
    print ('macro Recall Score tree model :{:.2f}'.format(macro_Recall_Score_tree_model*100))
    print ('macro F1 Score  tree model:{:.2f}'.format(macro_F1_Score_tree_model*100))

    # weighted
    weighted_Precision_Score_tree_model=precision_score(y_test, ypred, average='weighted')
    weighted_Recall_Score_tree_model=recall_score(y_test, ypred, average='weighted')
    weighted_F1_Score_tree_model=f1_score(y_test, ypred, average='weighted')
    print ('weighted Precision Score tree model:{:.2f}'.format(weighted_Precision_Score_tree_model*100)) 
    print ('weighted Recall Score tree model :{:.2f}'.format(weighted_Recall_Score_tree_model*100))
    print ('weighted F1 Score tree model:{:.2f}'.format(weighted_F1_Score_tree_model*100))


    tree_report=classification_report(y_test, ypred)
    print("############### svm_model_report #######################")
    print(tree_report)
    ############################################################################################################################



    model_rf= RandomForestClassifier(n_estimators=150, max_depth=50)

    model_rf.fit(X_train,y_train)
 
    ypred = model_rf.predict(X_test)

    # حساب مقاييس الأدا ء
    accuracy_rf=accuracy_score(y_test, ypred)
    print ('Accuracy Score Randomforest model :{:.2f}'.format(accuracy_rf*100))

    # micro
    micro_Precision_Score_rf=precision_score(y_test, ypred, average='micro')
    micro_Recall_Score_rf=recall_score(y_test, ypred, average='micro')
    micro_F1_Score_rf=f1_score(y_test, ypred, average='micro')
    print ('micro Precision Randomforest model:{:.2f}'.format(micro_Precision_Score_rf*100)) 
    print ('micro Recall Score Randomforest  model:{:.2f}'.format(micro_Recall_Score_rf*100))
    print ('micro F1 Score Randomforest model :{:.2f}'.format(micro_F1_Score_rf*100))

    # macro
    macro_Precision_Score_rf=precision_score(y_test, ypred, average='macro')
    macro_Recall_Score_rf=recall_score(y_test, ypred, average='macro')
    macro_F1_Score_rf=f1_score(y_test, ypred, average='macro')
    print ('macro Precision Score Randomforest model:{:.2f}'.format(macro_Precision_Score_rf*100)) 
    print ('macro Recall Score Randomforest model :{:.2f}'.format(macro_Recall_Score_rf*100))
    print ('macro F1 Score  Randomforest model:{:.2f}'.format(macro_F1_Score_rf*100))

    # weighted
    weighted_Precision_Score_rf=precision_score(y_test, ypred, average='weighted')
    weighted_Recall_Score_rf=recall_score(y_test, ypred, average='weighted')
    weighted_F1_Score_rf=f1_score(y_test, ypred, average='weighted')
    print ('weighted Precision Score Randomforest model:{:.2f}'.format(weighted_Precision_Score_rf*100)) 
    print ('weighted Recall Score Randomforest model :{:.2f}'.format(weighted_Recall_Score_rf*100))
    print ('weighted F1 Score Randomforest model:{:.2f}'.format(weighted_F1_Score_rf*100))

    
    rf_report=classification_report(y_test, ypred)
    print("############### Random Forest_report #######################")
    print(rf_report)

    ###############################################################################################################################
    #التصنيف باستخدام بايز



    #تقسيم البيانات إلى بيانات رقمية وفئوية
    X_cat=df[['Gender','Married','Dependents','Education','Self_Employed','Credit_History','Property_Area']]
    X_num=df[['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']]


    y=df['Loan_Status']


    X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(X_cat, y, test_size=0.2)
    X_train_num, X_test_num, y_train_num, y_test_num = train_test_split(X_num, y, test_size=0.2)


    #اجراء موازنة للأصناف
    oversample_cat = SMOTE() 
    X_train_cat, y_train_cat = oversample_cat.fit_resample(X_train_cat, y_train_cat) 

    oversample_num = SMOTE() 
    X_train_num, y_train_num = oversample_num.fit_resample(X_train_num, y_train_num) 

    # الملائمة مع البيانات

    # المصنف الفئوي
    model_cat_bayes = CategoricalNB(alpha=0)

     # المصنف الرقمي
    model_num_bayes = GaussianNB()
     # ملائمة المصنف الفئو ي
    model_cat_bayes.fit(X_train_cat, y_train_cat) 
    # ملائمة المصنف الرقمي
    model_num_bayes.fit(X_train_num,  y_train_num)


    #   تنبؤ النموذ ج الفئوي
    ypred_cat=model_cat_bayes.predict(X_test_cat) 
    # حساب مصفوفة الارتباك
    cf_matrix_bayes = confusion_matrix(y_test_cat, ypred_cat)
    print ('Confusion Matrix for bayes_model:')
    print (cf_matrix_bayes) 

    accuracy_bayes= accuracy_score(y_test_cat, ypred_cat)
    print ('Accuracy Score bayes model :{:.2f}'.format(accuracy_bayes*100))

    # micro
    micro_Precision_Score_bayes=precision_score(y_test_cat, ypred_cat, average='micro')
    micro_Recall_Score_bayes_model=recall_score(y_test_cat, ypred_cat, average='micro')
    micro_F1_Score_bayes_model=f1_score(y_test_cat, ypred_cat, average='micro')
    print ('micro Precision Score bayes model:{:.2f}'.format(micro_Precision_Score_bayes*100)) 
    print ('micro Recall Score bayes model:{:.2f}'.format(micro_Recall_Score_bayes_model*100))
    print ('micro F1 Score bayes model :{:.2f}'.format(micro_F1_Score_bayes_model*100))

    # macro
    macro_Precision_Score_bayes_model=precision_score(y_test_cat, ypred_cat, average='macro')
    macro_Recall_Score_bayes_model=recall_score(y_test_cat, ypred_cat, average='macro')
    macro_F1_Score_bayes_model=f1_score(y_test_cat, ypred_cat, average='macro')
    print ('macro Precision Score bayes model:{:.2f}'.format(macro_Precision_Score_bayes_model*100)) 
    print ('macro Recall Score bayes model :{:.2f}'.format(macro_Recall_Score_bayes_model*100))
    print ('macro F1 Score  bayes model:{:.2f}'.format(macro_F1_Score_bayes_model*100))

    # weighted
    weighted_Precision_Score_bayes_model=precision_score(y_test_cat, ypred_cat, average='weighted')
    weighted_Recall_Score_bayes_model=recall_score(y_test_cat, ypred_cat, average='weighted')
    weighted_F1_Score_bayes_model=f1_score(y_test_cat, ypred_cat, average='weighted')
    print ('weighted Precision Score bayes model:{:.2f}'.format(weighted_Precision_Score_bayes_model*100)) 
    print ('weighted Recall Score bayes model :{:.2f}'.format(weighted_Recall_Score_bayes_model*100))
    print ('weighted F1 Score bayes model:{:.2f}'.format(weighted_F1_Score_bayes_model*100))

    ############################################################################################################################
    #   تنبؤ النموذ ج الرقمي
    ypred_num=model_num_bayes.predict(X_test_num) 
    # حساب مصفوفة الارتباك
    cf_matrix_bayes = confusion_matrix(y_test_num, ypred_num)
    print ('Confusion Matrix for bayes_model:')
    print (cf_matrix_bayes) # تسميات قيم مصفوفة الارتباك


    accuracy_num_bayes= accuracy_score(y_test_num, ypred_num)
    print ('Accuracy Score bayes num model :{:.2f}'.format(accuracy_num_bayes*100))

    # micro
    micro_Precision_Score_num_bayes=precision_score(y_test_num, ypred_num, average='micro')
    micro_Recall_Score_bayes_num_model=recall_score(y_test_num, ypred_num, average='micro')
    micro_F1_Score_bayes_num_model=f1_score(y_test_num, ypred_num, average='micro')
    print ('micro Precision Score bayes model:{:.2f}'.format(micro_Precision_Score_num_bayes*100)) 
    print ('micro Recall Score bayes model:{:.2f}'.format(micro_Recall_Score_bayes_num_model*100))
    print ('micro F1 Score bayes model :{:.2f}'.format(micro_F1_Score_bayes_num_model*100))

    # macro
    macro_Precision_Score_num_bayes_model=precision_score(y_test_num, ypred_num, average='macro')
    macro_Recall_Score_num_bayes_model=recall_score(y_test_num, ypred_num, average='macro')
    macro_F1_Score_num_bayes_model=f1_score(y_test_num, ypred_num, average='macro')
    print ('macro Precision Score bayes model:{:.2f}'.format(macro_Precision_Score_num_bayes_model*100)) 
    print ('macro Recall Score bayes model :{:.2f}'.format(macro_Recall_Score_num_bayes_model*100))
    print ('macro F1 Score  bayes model:{:.2f}'.format(macro_F1_Score_num_bayes_model*100))

    # weighted
    weighted_Precision_Score_num_bayes_model=precision_score(y_test_num, ypred_num, average='weighted')
    weighted_Recall_Score_num_bayes_model=recall_score(y_test_num, ypred_num, average='weighted')
    weighted_F1_Score_num_bayes_model=f1_score(y_test_num, ypred_num, average='weighted')
    print ('weighted Precision Score bayes model:{:.2f}'.format(weighted_Precision_Score_num_bayes_model*100)) 
    print ('weighted Recall Score bayes model :{:.2f}'.format(weighted_Recall_Score_num_bayes_model*100))
    print ('weighted F1 Score bayes model:{:.2f}'.format(weighted_F1_Score_num_bayes_model*100))


    ## classfication report

    nums_bayes_report=classification_report(y_test_num,ypred_num)
    print("############### nums_bayes_report #######################")
    print(nums_bayes_report)

    cat_bayes_report=classification_report(y_test_cat, ypred_cat)
    print("############### cat_bayes_report #######################")
    print(cat_bayes_report)

    ###########################################################################################################################

        
    ############################################################################################################################
    #انشاء جدول لمقارنة مقاييس الأداء للمصنفات 
    result = pd.DataFrame(columns=['M','knn','logistic','SVM','tree','Randomforest','bayes_cat','bayes_num'])

    row = {'M':'accuracy','knn': accuracy_knn,'logistic':accuracy_logistic,'SVM':accuracy_svm,'tree':accuracy_tree,'Randomforest': accuracy_rf,'bayes_cat':accuracy_bayes,'bayes_num':accuracy_num_bayes}
    result= result._append(row,ignore_index=True)
    row = {'M':'micro_Precision','knn': micro_Precision_Score_knn,'logistic':micro_Precision_Score_logistic,'SVM':micro_Precision_Score_svm,'tree':micro_Precision_Score_tree,'Randomforest':micro_Precision_Score_rf ,'bayes_cat':micro_Precision_Score_bayes,'bayes_num':micro_Precision_Score_num_bayes}
    result= result._append(row,ignore_index=True)
    row = {'M':'micro_Recall','knn':micro_Recall_Score_knn ,'logistic':micro_Recall_Score_logistic_model,'SVM':micro_Recall_Score_svm_model,'tree':micro_Recall_Score_tree_model,'Randomforest':micro_Recall_Score_rf ,'bayes_cat':micro_Recall_Score_bayes_model,'bayes_num':micro_Recall_Score_bayes_num_model}
    result= result._append(row,ignore_index=True)
    row = {'M':'micro_F1_Score','knn':micro_F1_Score_knn ,'logistic':micro_F1_Score_logistic_model,'SVM':micro_F1_Score_svm_model,'tree':micro_F1_Score_tree_model,'Randomforest': micro_F1_Score_rf,'bayes_cat':micro_F1_Score_bayes_model,'bayes_num':micro_F1_Score_bayes_num_model}
    result= result._append(row,ignore_index=True)
    row = {'M':':macro_Precision','knn':macro_Precision_Score_knn ,'logistic':macro_Precision_Score_logistic_model,'SVM':macro_Precision_Score_svm_model,'tree':macro_Precision_Score_tree_model,'Randomforest':macro_Precision_Score_rf,'bayes_cat':macro_Precision_Score_bayes_model,'bayes_num':macro_Precision_Score_num_bayes_model}
    result= result._append(row,ignore_index=True)
    row = {'M':'macro_Recall','knn':macro_Recall_Score_knn ,'logistic':macro_Recall_Score_logistic_model,'SVM':macro_Recall_Score_svm_model,'tree':macro_Recall_Score_tree_model,'Randomforest':macro_Recall_Score_rf ,'bayes_cat':macro_Recall_Score_bayes_model,'bayes_num':macro_Recall_Score_num_bayes_model}
    result= result._append(row,ignore_index=True)
    row = {'M':':macro_F1_Score','knn':macro_F1_Score_knn ,'logistic':macro_F1_Score_logistic_model,'SVM':macro_F1_Score_svm_model,'tree':macro_F1_Score_tree_model,'Randomforest':macro_F1_Score_rf ,'bayes_cat':macro_F1_Score_bayes_model,'bayes_num':macro_F1_Score_num_bayes_model}
    result= result._append(row,ignore_index=True)
    row = {'M':'weighted_Precision_Score','knn':weighted_Precision_Score_knn ,'logistic':weighted_Precision_Score_logistic_model,'SVM':weighted_Precision_Score_svm_model,'tree':weighted_Precision_Score_tree_model,'Randomforest': weighted_Precision_Score_rf,'bayes_cat':weighted_Precision_Score_bayes_model,'bayes_num':weighted_Precision_Score_num_bayes_model}
    result= result._append(row,ignore_index=True)
    row = {'M':'weighted_Recall_Score','knn':weighted_Recall_Score_knn ,'logistic':weighted_Recall_Score_logistic_model,'SVM':weighted_Recall_Score_svm_model,'tree':weighted_Recall_Score_tree_model,'Randomforest': weighted_Recall_Score_rf,'bayes_cat':weighted_Recall_Score_bayes_model,'bayes_num':weighted_Recall_Score_num_bayes_model}
    result= result._append(row,ignore_index=True)
    row = {'M':'weighted_F1_Score','knn':weighted_F1_Score_knn ,'logistic':weighted_F1_Score_logistic_model,'SVM':weighted_F1_Score_svm_model,'tree':weighted_F1_Score_tree_model,'Randomforest': weighted_F1_Score_rf,'bayes_cat':weighted_F1_Score_bayes_model,'bayes_num':weighted_F1_Score_num_bayes_model}
    result= result._append(row,ignore_index=True)
    print(result)

    # ############################################################################################################################
        #                           M       knn  logistic       SVM      tree  Randomforest  bayes_cat  bayes_num
        # 0                  accuracy  0.658537  0.813008  0.788618  0.601626      0.674797   0.780488   0.504065
        # 1           micro_Precision  0.658537  0.813008  0.788618  0.601626      0.674797   0.780488   0.504065
        # 2              micro_Recall  0.658537  0.813008  0.788618  0.601626      0.674797   0.780488   0.504065
        # 3            micro_F1_Score  0.658537  0.813008  0.788618  0.601626      0.674797   0.780488   0.504065
        # 4          :macro_Precision  0.608430  0.796987  0.754402  0.572589      0.630769   0.735624   0.451465
        # 5              macro_Recall  0.617222  0.735387  0.717945  0.584224      0.644249   0.714559   0.455110
        # 6           :macro_F1_Score  0.610994  0.754619  0.730808  0.568668      0.634146   0.723088   0.451735
        # 7  weighted_Precision_Score  0.674787  0.808044  0.780196  0.647473      0.696143   0.773133   0.487701
        # 8     weighted_Recall_Score  0.658537  0.813008  0.788618  0.601626      0.674797   0.780488   0.504065
        # 9         weighted_F1_Score  0.665170  0.802303  0.780504  0.616166      0.682729   0.775363   0.494425

    #   the best model is logistic_model





def try_different_model(df):
    df= proccess_data(df)
    X= df[['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area']]
    y= df['Loan_Status']
    scaler= StandardScaler()
    X[['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']]= scaler.fit_transform(X[['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    oversample = SMOTE() 
    X_train, y_train = oversample.fit_resample(X_train, y_train) 
    
    # # # التنبؤ باستخدام الجار الأقرب
    knn_model = KNeighborsClassifier(5)
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    cf_matrix_knn = confusion_matrix(y_test, y_pred)
    accuracy_knn=accuracy_score(y_test, y_pred)
    micro_Precision_Score_knn=precision_score(y_test, y_pred, average='micro')
    micro_Recall_Score_knn=recall_score(y_test, y_pred, average='micro')
    micro_F1_Score_knn=f1_score(y_test, y_pred, average='micro')
    macro_Precision_Score_knn=precision_score(y_test, y_pred, average='macro')
    macro_Recall_Score_knn=recall_score(y_test, y_pred, average='macro')
    macro_F1_Score_knn=f1_score(y_test, y_pred, average='macro')
    weighted_Precision_Score_knn=precision_score(y_test, y_pred, average='weighted')
    weighted_Recall_Score_knn=recall_score(y_test, y_pred, average='weighted')
    weighted_F1_Score_knn=f1_score(y_test, y_pred, average='weighted')
    knn_report=classification_report(y_test,y_pred)
####################################################################################################################
    #التصنيف باستخدام الانحدار اللوجستي
    logistic_model =LogisticRegression()
    logistic_model.fit(X_train , y_train)
    # التنبؤ مع بيانات الاختبار
    ypred= logistic_model.predict(X_test)
    # حساب مصفوفة الارتباك
    cf_matrix_logistic = confusion_matrix(y_test, ypred)
    accuracy_logistic=accuracy_score(y_test, ypred)
    # micro
    micro_Precision_Score_logistic=precision_score(y_test, ypred, average='micro')
    micro_Recall_Score_logistic_model=recall_score(y_test, ypred, average='micro')
    micro_F1_Score_logistic_model=f1_score(y_test, ypred, average='micro')
    # macro
    macro_Precision_Score_logistic_model=precision_score(y_test, ypred, average='macro')
    macro_Recall_Score_logistic_model=recall_score(y_test, ypred, average='macro')
    macro_F1_Score_logistic_model=f1_score(y_test, ypred, average='macro')
    # weighted
    weighted_Precision_Score_logistic_model=precision_score(y_test, ypred, average='weighted')
    weighted_Recall_Score_logistic_model=recall_score(y_test, ypred, average='weighted')
    weighted_F1_Score_logistic_model=f1_score(y_test, ypred, average='weighted')
    logistic_report=classification_report(y_test, ypred)
    ############################################################################################################################
    #التصنيف باستخدام Support Vector Machines
    from sklearn.svm import SVC
    svm_model = SVC()
    svm_model.fit(X_train,y_train)
    y_pred=svm_model.predict(X_test)
    cf_matrix_svm = confusion_matrix(y_test, y_pred)
    # حساب مقاييس الأدا ء
    accuracy_svm=accuracy_score(y_test, y_pred)
    # micro
    micro_Precision_Score_svm=precision_score(y_test, y_pred, average='micro')
    micro_Recall_Score_svm_model=recall_score(y_test, y_pred, average='micro')
    micro_F1_Score_svm_model=f1_score(y_test, y_pred, average='micro')
    # macro
    macro_Precision_Score_svm_model=precision_score(y_test, y_pred, average='macro')
    macro_Recall_Score_svm_model=recall_score(y_test, y_pred, average='macro')
    macro_F1_Score_svm_model=f1_score(y_test, y_pred, average='macro')
    # weighted
    weighted_Precision_Score_svm_model=precision_score(y_test, y_pred, average='weighted')
    weighted_Recall_Score_svm_model=recall_score(y_test, y_pred, average='weighted')
    weighted_F1_Score_svm_model=f1_score(y_test, y_pred, average='weighted')
    svm_report=classification_report(y_test, y_pred)
    ###############################################################################################################################
    #التصنيف باستخدام شجرة القرار
    from sklearn import tree
    from sklearn.tree import DecisionTreeClassifier
    tree_model = DecisionTreeClassifier(criterion="entropy")
    tree_model.fit(X_train, y_train)
    ypred=tree_model.predict(X_test) 
    #مقاييس تقييم النموذج
    # حساب مصفوفة الارتباك
    cf_matrix_tree = confusion_matrix(y_test, ypred)
    # حساب مقاييس الأدا ء
    accuracy_tree=accuracy_score(y_test, ypred)
    # micro
    micro_Precision_Score_tree=precision_score(y_test, ypred, average='micro')
    micro_Recall_Score_tree_model=recall_score(y_test, ypred, average='micro')
    micro_F1_Score_tree_model=f1_score(y_test, ypred, average='micro')
    # macro
    macro_Precision_Score_tree_model=precision_score(y_test, ypred, average='macro')
    macro_Recall_Score_tree_model=recall_score(y_test, ypred, average='macro')
    macro_F1_Score_tree_model=f1_score(y_test, ypred, average='macro')
    # weighted
    weighted_Precision_Score_tree_model=precision_score(y_test, ypred, average='weighted')
    weighted_Recall_Score_tree_model=recall_score(y_test, ypred, average='weighted')
    weighted_F1_Score_tree_model=f1_score(y_test, ypred, average='weighted')
    tree_report=classification_report(y_test, ypred)
    ############################################################################################################################
    from sklearn.ensemble import RandomForestClassifier
    model_rf= RandomForestClassifier(n_estimators=150, max_depth=50)
    model_rf.fit(X_train,y_train)
    ypred = model_rf.predict(X_test)
    # حساب مقاييس الأدا ء
    cf_matrix_rf = confusion_matrix(y_test, ypred)
    accuracy_rf=accuracy_score(y_test, ypred)
    micro_Precision_Score_rf=precision_score(y_test, ypred, average='micro')
    micro_Recall_Score_rf=recall_score(y_test, ypred, average='micro')
    micro_F1_Score_rf=f1_score(y_test, ypred, average='micro')
    # macro
    macro_Precision_Score_rf=precision_score(y_test, ypred, average='macro')
    macro_Recall_Score_rf=recall_score(y_test, ypred, average='macro')
    macro_F1_Score_rf=f1_score(y_test, ypred, average='macro')
    # weighted
    weighted_Precision_Score_rf=precision_score(y_test, ypred, average='weighted')
    weighted_Recall_Score_rf=recall_score(y_test, ypred, average='weighted')
    weighted_F1_Score_rf=f1_score(y_test, ypred, average='weighted')
    rf_report=classification_report(y_test, ypred)
    ###############################################################################################################################
    #التصنيف باستخدام بايز
    from sklearn.naive_bayes import CategoricalNB,GaussianNB
    #تقسيم البيانات إلى بيانات رقمية وفئوية
    X_cat=df[['Gender','Married','Dependents','Education','Self_Employed','Credit_History','Property_Area']]
    X_num=df[['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']]
    y=df['Loan_Status']
    X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(X_cat, y, test_size=0.2)
    X_train_num, X_test_num, y_train_num, y_test_num = train_test_split(X_num, y, test_size=0.2)
    #اجراء موازنة للأصناف
    oversample_cat = SMOTE() 
    X_train_cat, y_train_cat = oversample_cat.fit_resample(X_train_cat, y_train_cat) 
    oversample_num = SMOTE() 
    X_train_num, y_train_num = oversample_num.fit_resample(X_train_num, y_train_num) 
    # المصنف الفئوي
    model_cat_bayes = CategoricalNB(alpha=0)
     # المصنف الرقمي
    model_num_bayes = GaussianNB()
    model_cat_bayes.fit(X_train_cat, y_train_cat) 
    model_num_bayes.fit(X_train_num,  y_train_num)
    ypred_cat=model_cat_bayes.predict(X_test_cat) 
    cf_matrix_cat_bayes = confusion_matrix(y_test_cat, ypred_cat)
    accuracy_bayes= accuracy_score(y_test_cat, ypred_cat)
    # micro
    micro_Precision_Score_bayes=precision_score(y_test_cat, ypred_cat, average='micro')
    micro_Recall_Score_bayes_model=recall_score(y_test_cat, ypred_cat, average='micro')
    micro_F1_Score_bayes_model=f1_score(y_test_cat, ypred_cat, average='micro')
    # macro
    macro_Precision_Score_bayes_model=precision_score(y_test_cat, ypred_cat, average='macro')
    macro_Recall_Score_bayes_model=recall_score(y_test_cat, ypred_cat, average='macro')
    macro_F1_Score_bayes_model=f1_score(y_test_cat, ypred_cat, average='macro')
    # weighted
    weighted_Precision_Score_bayes_model=precision_score(y_test_cat, ypred_cat, average='weighted')
    weighted_Recall_Score_bayes_model=recall_score(y_test_cat, ypred_cat, average='weighted')
    weighted_F1_Score_bayes_model=f1_score(y_test_cat, ypred_cat, average='weighted')
    ############################################################################################################################
    #   تنبؤ النموذ ج الرقمي
    ypred_num=model_num_bayes.predict(X_test_num) 
    # حساب مصفوفة الارتباك
    cf_matrix_num_bayes = confusion_matrix(y_test_num, ypred_num)
    accuracy_num_bayes= accuracy_score(y_test_num, ypred_num)
    # micro
    micro_Precision_Score_num_bayes=precision_score(y_test_num, ypred_num, average='micro')
    micro_Recall_Score_bayes_num_model=recall_score(y_test_num, ypred_num, average='micro')
    micro_F1_Score_bayes_num_model=f1_score(y_test_num, ypred_num, average='micro')
    # macro
    macro_Precision_Score_num_bayes_model=precision_score(y_test_num, ypred_num, average='macro')
    macro_Recall_Score_num_bayes_model=recall_score(y_test_num, ypred_num, average='macro')
    macro_F1_Score_num_bayes_model=f1_score(y_test_num, ypred_num, average='macro')
    # weighted
    weighted_Precision_Score_num_bayes_model=precision_score(y_test_num, ypred_num, average='weighted')
    weighted_Recall_Score_num_bayes_model=recall_score(y_test_num, ypred_num, average='weighted')
    weighted_F1_Score_num_bayes_model=f1_score(y_test_num, ypred_num, average='weighted')
    ## classfication report
    nums_bayes_report=classification_report(y_test_num,ypred_num)
    cat_bayes_report=classification_report(y_test_cat, ypred_cat)
    ###########################################################################################################################
    # انشاء قاموس لمقارنة مقاييس الأداء للمصنفات
    results= {
         'knn': {
              'accuracy': accuracy_knn,
              'micro_Precision': micro_Precision_Score_knn,
              'micro_Recall': micro_Recall_Score_knn,
              'micro_F1_Score': micro_F1_Score_knn,
              'macro_Precision': macro_Precision_Score_knn,
              'macro_Recall': macro_Recall_Score_knn,
              'macro_F1_Score': macro_F1_Score_knn,
              'weighted_Precision_Score': weighted_Precision_Score_knn,
              'weighted_Recall_Score': weighted_Recall_Score_knn,
              'weighted_F1_Score': weighted_F1_Score_knn,
              'cf_matrix': cf_matrix_knn,
              'report': knn_report
         },
         'logistic':{
              'accuracy': accuracy_logistic,
              'micro_Precision': micro_Precision_Score_logistic,
              'micro_Recall': micro_Recall_Score_logistic_model,
              'micro_F1_Score': micro_F1_Score_logistic_model,
              'macro_Precision': macro_Precision_Score_logistic_model,
              'macro_Recall': macro_Recall_Score_logistic_model,
              'macro_F1_Score': macro_F1_Score_logistic_model,
              'weighted_Precision_Score': weighted_Precision_Score_logistic_model,
              'weighted_Recall_Score': weighted_Recall_Score_logistic_model,
              'weighted_F1_Score':weighted_F1_Score_logistic_model,
              'cf_matrix': cf_matrix_logistic,
              'report': logistic_report
         },
         'SVM':{
              'accuracy': accuracy_svm,
              'micro_Precision': micro_Precision_Score_svm,
              'micro_Recall':micro_Recall_Score_svm_model,
              'micro_F1_Score':micro_F1_Score_svm_model,
              'macro_Precision': macro_Precision_Score_svm_model,
              'macro_Recall': macro_Recall_Score_svm_model,
              'macro_F1_Score': macro_F1_Score_svm_model,
              'weighted_Precision_Score': weighted_Precision_Score_svm_model,
              'weighted_Recall_Score': weighted_Recall_Score_svm_model,
              'weighted_F1_Score': weighted_F1_Score_svm_model,
              'cf_matrix': cf_matrix_svm,
              'report': svm_report
         },
         'tree':{
              'accuracy': accuracy_tree,
              'micro_Precision': micro_Precision_Score_tree,
              'micro_Recall': micro_Recall_Score_tree_model,
              'micro_F1_Score': micro_F1_Score_tree_model,
              'macro_Precision': macro_Precision_Score_tree_model,
              'macro_Recall': macro_Recall_Score_tree_model,
              'macro_F1_Score': macro_F1_Score_tree_model,
              'weighted_Precision_Score': weighted_Precision_Score_tree_model,
              'weighted_Recall_Score': weighted_Recall_Score_tree_model,
              'weighted_F1_Score': weighted_F1_Score_tree_model,
              'cf_matrix': cf_matrix_tree,
              'report': tree_report
         },
         'Randomforest':{
              'accuracy': accuracy_rf,
              'micro_Precision': micro_Precision_Score_rf,
              'micro_Recall': micro_Recall_Score_rf,
              'micro_F1_Score': micro_F1_Score_rf,
              'macro_Precision': macro_Precision_Score_rf,
              'macro_Recall': macro_Recall_Score_rf,
              'macro_F1_Score': macro_F1_Score_rf,
              'weighted_Precision_Score': weighted_Precision_Score_rf,
              'weighted_Recall_Score': weighted_Recall_Score_rf,
              'weighted_F1_Score': weighted_F1_Score_rf,
              'cf_matrix': cf_matrix_rf,
              'report': rf_report
         },
         'bayes_cat':{
              'accuracy': accuracy_bayes,
              'micro_Precision': micro_Precision_Score_bayes,
              'micro_Recall': micro_Recall_Score_bayes_model,
              'micro_F1_Score': micro_F1_Score_bayes_model,
              'macro_Precision': macro_Precision_Score_bayes_model,
              'macro_Recall': macro_Recall_Score_bayes_model,
              'macro_F1_Score': macro_F1_Score_bayes_model,
              'weighted_Precision_Score': weighted_Precision_Score_bayes_model,
              'weighted_Recall_Score': weighted_Recall_Score_bayes_model,
              'weighted_F1_Score': weighted_F1_Score_bayes_model,
              'cf_matrix': cf_matrix_cat_bayes,
              'report': cat_bayes_report
         },
         'bayes_num':{
              'accuracy': accuracy_num_bayes,
              'micro_Precision': micro_Precision_Score_num_bayes,
              'micro_Recall': micro_Recall_Score_bayes_num_model,
              'micro_F1_Score': micro_F1_Score_bayes_num_model,
              'macro_Precision': macro_Precision_Score_num_bayes_model,
              'macro_Recall': macro_Recall_Score_num_bayes_model,
              'macro_F1_Score': macro_F1_Score_num_bayes_model,
              'weighted_Precision_Score': weighted_Precision_Score_num_bayes_model,
              'weighted_Recall_Score': weighted_Recall_Score_num_bayes_model,
              'weighted_F1_Score': weighted_F1_Score_num_bayes_model,
              'cf_matrix': cf_matrix_num_bayes,
              'report': nums_bayes_report
         },
    }
    return results



def train_and_build(df, model_path=""):
      df= proccess_data(df)
      X= df[['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area']]
      y= df['Loan_Status']
      oversample = SMOTE() 
      X, y = oversample.fit_resample(X, y)
      pipe = make_pipeline(StandardScaler(), LogisticRegression())
      endpoint_model = pipe.fit(X, y)
     
      parent_path = pathlib.Path(__file__).parent.absolute()
     #  with open(parent_path if model_path==None else os.path.join(parent_path,model_path), 'wb') as file:
     #        pickle.dump(endpoint_model, file)
      with open(os.path.join(parent_path,model_path,'model.pkl'), 'wb') as file:
            pickle.dump(endpoint_model, file)

def predict_value(row, model_path=""):
      row = proccess_data(row)  
      parent_path = pathlib.Path(__file__).parent.absolute()
      with open(os.path.join(parent_path,model_path,'model.pkl'), 'rb') as file:
           loaded_model= pickle.load(file)

      return loaded_model.predict(row)


if __name__ == "__main__":   
    parent_path = pathlib.Path(__file__).parent.absolute()
    local_path='data/loan_prediction.csv'
    df = pd.read_csv(os.path.join(parent_path,local_path)) 
    live_model_run(df)


