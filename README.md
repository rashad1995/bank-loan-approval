# bank-loan-approval
## model building  Part
## قسم تحليل وبناء النموذج
ملف تحليل وبناء النموذج building.py
### إدراج الملف في المشروع
يجب إدراج موديول تحليل البيانات بالاضافة لملف بناء الموديول
```
import analysis
import building
```
### تابع إجراء التنبؤ باستخدام نماذج مختلفة وإرجاع مقاييس الأداء للنماذج المختلفة
```
results = try_different_model(df)
```
مثال
```
df = pd.read_csv("F:/MLT_Chapters_F23/hm/HW_F24_MLT/loan_prediction/loan_prediction.csv")
x= try_different_model(df)
print(x)
```

### تابع تدريب نموذج وحفظ النموذج المدرب 
```
train_and_build(df, model_path="")  # model_path is the relative path where you want to save the model
```
مثال
```
df = pd.read_csv("F:/MLT_Chapters_F23/hm/HW_F24_MLT/loan_prediction/loan_prediction.csv")
train_and_build(df)
```

### تابع التنبؤ بقيمة محددة حسب النموذج المحفوظ
```
predict_value(row, model_path="")  # model_path is the path for saved model
```
مثال
```
p = predict_value([[1,0,0,0,0,5849,0.0,128.0,360.0,1.0,2]], 'model')
```
أو لمجموعة قيم 
```
p = predict_value(dataFrame, 'model')
```

### لتشغيل الملف بشكل مباشر على جهازك ومعاينة نتائج تحليل نماذج مختلفة 
يجب تخزين البيانات في المسار local_path='data/loan_prediction.csv'
أو تعديل المسار من الملف
