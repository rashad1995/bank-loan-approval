# bank-loan-approval

# bank-loan-approval

# تطبيق ويب للموافقة على القروض باستخدام تعلم الآلة

هذا المشروع عبارة عن تطبيق ويب يساعد البنوك في اتخاذ قرارات تلقائية حول الموافقة أو رفض طلبات القروض. يعتمد على نموذج تعلم آلة مدرب على بيانات حقيقية بهدف بناء نظام متكامل يشمل تحليل البيانات، تدريب النموذج، تقييم الأداء، وواجهة استخدام تفاعلية

---

## كيفية تشغيل المشروع على جهازك (Locally)

1. **نسخ المشروع من GitHub:**

```bash
gitclone
https://github.com/rashad1995/bank-loan-approval.git
cd bank-loan-approval
```
3. **تثبيت المكتبات المطلوبة:**
```bash
pip install -r requirements.txt
```
---

## تشغيل التطبيق: باستخدام Streamlit أو Flask

يمكن تطوير وتشغيل المشروع باستخدام **Streamlit** أو **Flask** حسب الطريقة المفضلة:

### الخيار الأول: Streamlit (مُفضل لتطبيقات تعلم الآلة)

Streamlit هو مكتبة بسيطة وسريعة لبناء تطبيقات تفاعلية باستخدام بايثون، ومثالية لمشاريع تحليل البيانات والنماذج.

**لتشغيل التطبيق:**

```bash
streamlit run app.py
```

- سيفتح التطبيق تلقائيًا في المتصفح على: `http://localhost:8501`

### الخيار الثاني: Flask (مخصص أكثر)

Flask هو إطار عمل خفيف يسمح ببناء واجهات ويب مخصصة بشكل كامل. مناسب إذا كنت تريد تحكمًا أكبر في تصميم الواجهة.

**لتشغيل التطبيق:**

```bash
python app.py
```

- سيفتح التطبيق على: `http://127.0.0.1:5000`

---

| الإطار      | الأفضل لـ                         | السهولة      | التخصيص        |
|-------------|----------------------------------|--------------|----------------|
| Streamlit   | النماذج وتحليل البيانات السريع     | عالية         | محدود           |
| Flask       | تطبيقات الويب الكاملة              | متوسطة        | عالي            |

---

## المكتبات المطلوبة (requirements.txt)

تأكد من تثبيت المكتبات الموجودة في ملف `requirements.txt`.

### محتويات مقترحة لملف `requirements.txt`:

```
pandas
numpy
scikit-learn
matplotlib
seaborn
plotly
streamlit
joblib
```

### وصف المكتبات:

| المكتبة         | الاستخدام                                       |
|------------------|------------------------------------------------|
| `pandas`         | معالجة وتحليل البيانات                         |
| `numpy`          | العمليات الرياضية والتعامل مع المصفوفات        |
| `scikit-learn`   | بناء وتقييم نماذج تعلم الآلة                   |
| `matplotlib`     | الرسوم البيانية الأساسية                        |
| `seaborn`        | رسوم بيانية متقدمة                             |
| `plotly`         | رسوم تفاعلية (اختياري)                         |
| `streamlit`      | بناء واجهة المستخدم                             |
| `joblib`         | حفظ واسترجاع النموذج المدرب                     |

--
## أعضاء الفريق

- رشاد خرما – `rashad_306889`  
- علي عباس – `ali_241692`  
- ريم أحمد – `reem_278333`
- منصور دويري - `mansour_265936`
- جلال الحمود   - `jalal_266688`


-------------------------------------------------------------------------------------------------------------



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



---------------------------------------------------------------------------


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
