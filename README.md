# bank-loan-approval
## Data Analysis Part
## قسم تحليل البيانات
ملف تحليل البيانات analysis.py
### إدراج ملف تحليل البيانات في المشروع
```
import analysis
```
### تابع إجراء تحليل كامل للبيانات في بيئة بايثون الخاصة بك
```
live_analysis(dataFrame)    # dataframe as panada dataframe 
```
مثال
```
df = pd.read_csv("F:/MLT_Chapters_F23/hm/HW_F24_MLT/loan_prediction/loan_prediction.csv")
live_analysis(df)
```

### تابع أجراء معالجة وتنظيف وتهيئة البيانات 
```
cleaned_dataframe = proccess_data(df)
```


### تابع الإظهار ورسم مخططات العلاقات
```
plot_analysis(df, dir_path="")   # dir_path is the relative path where you want to store images
```
مثال
```
plot_analysis(df, "images\images")
```

### تابع توليد بروفايل البيانات باستخدام مكتبة ydata_profiling
```
generate_data_profile(df, dir_path="")  # dir_path is the relative path where you want to store html profile
```
مثال
```
generate_data_profile(df, "profile\profile")
```

### تابع إيجاد المعلومات الاحصائية الخاصة بالبيانات
```
info = analysis_info(df)
```
- info is dictionary like {'correlation':....., 'sample': ......, 'info': ....., 'statistical': .......}

مثال
```
inf =analysis_info(df)
print(inf['correlation'])
```

### لتشغيل الملف بشكل مباشر على جهازك 
يجب تخزين البيانات في المسار local_path='data/loan_prediction.csv'
أو تعديل المسار من الملف
