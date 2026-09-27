# 🚀 Deployment — FAA Wildlife Strike Damage Prediction

بيغطي المتطلب كامل زي ما طلب المحاضر: **حفظ الموديل + API (FastAPI) + واجهة Streamlit + رفعه Online**.

## 📁 هيكل المشروع

```
project/
├── app.py                                          # واجهة Streamlit
├── api.py                                          # FastAPI service
├── pipeline.py                                     # منطق المعالجة المشترك (نفس الكود لـapp.py وapi.py)
├── train_demo_model.py                             # سكريبت تدريب الموديل
├── demo_model.joblib                               # الموديل المدرّب المحفوظ
├── demo_model_columns.joblib                       # ترتيب الأعمدة (80 عمود)
├── model_data_A_v2_imputed_scaler.joblib           # RobustScaler المدرّب
├── model_data_A_v2_imputed_target_encoding_maps.joblib  # K-Fold Target Encoding maps
├── requirements.txt
├── Procfile                                        # لـRender/Heroku
└── .gitignore
```

---

## 1️⃣ تشغيل محلي (على جهازك)

### أ) تثبيت المكتبات
```bash
pip install -r requirements.txt
```

### ب) تشغيل واجهة Streamlit
```bash
streamlit run app.py
```
هيفتح على `http://localhost:8501` تلقائيًا.

### ج) تشغيل الـAPI (FastAPI)
```bash
uvicorn api:app --reload --port 8000
```
- التوثيق التفاعلي (Swagger UI) هيبقى على: `http://localhost:8000/docs`
- تقدر تجرب طلب مباشر:
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "AC_CLASS":"A","AC_MASS":1,"TYPE_ENG":"D","NUM_ENGS":1,
  "PHASE_OF_FLIGHT":"En Route","HEIGHT":5000,"SPEED":180,"DISTANCE":10,
  "INCIDENT_MONTH":9,"INCIDENT_YEAR":2026,"HOUR":18,"TIME_OF_DAY":"Dusk",
  "SKY":"Overcast","IS_FOG":false,"IS_RAIN":true,"IS_SNOW":false,
  "SPECIES":"Unknown bird","SIZE":"Large","seen_reported":true,
  "NUM_SEEN_ORDINAL":3,"NUM_STRUCK_ORDINAL":3,
  "STATE":"CA","FAAREGION":"AWP","threshold":0.225
}'
```
**النتيجة الحقيقية اللي طلعت لما اختبرته:** `{"damage_probability":0.9205,"threshold_used":0.225,"predicted_damage":true}` ✅

---

## 2️⃣ رفعه على GitHub

```bash
git init
git add .
git commit -m "FAA Wildlife Strike damage prediction — full pipeline + API + Streamlit UI"
git branch -M main
git remote add origin https://github.com/<username>/<repo-name>.git
git push -u origin main
```

⚠️ **ملحوظة**: ملف `.gitignore` مستبعد بيه ملفات الداتا الكبيرة (`Public.xlsx`, `engineered_data.csv`, أي `.csv`) عشان الريبو يفضل خفيف. الملفات المطلوبة فعليًا للـDeployment (الموديل + الـjoblib files) مش مستبعدة، هترفع عادي.

---

## 3️⃣ الاستضافة (Hosting)

### أ) Streamlit Cloud (لواجهة الـapp.py) — الأسهل والمجاني

1. روح على [share.streamlit.io](https://share.streamlit.io)
2. اعمل تسجيل دخول بحساب GitHub
3. اضغط **New app**، اختار الريبو بتاعك، وحدد:
   - **Main file path**: `app.py`
4. اضغط **Deploy** — هياخد دقيقة أو اتنين ويطلعلك رابط عام (`https://<app-name>.streamlit.app`)

### ب) Render (لـAPI الـFastAPI) — مجاني للمشاريع الصغيرة

1. روح على [render.com](https://render.com) واعمل حساب (بـGitHub)
2. **New +** → **Web Service** → اختار الريبو
3. الإعدادات:
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn api:app --host 0.0.0.0 --port $PORT`
4. اضغط **Create Web Service** — Render هيستخدم `$PORT` تلقائيًا (نفس الـPort اللي في الـProcfile)

### ج) Heroku (بديل لـRender)

```bash
heroku login
heroku create faa-wildlife-strike-api
git push heroku main
```
Heroku هيقرأ `Procfile` تلقائيًا ويشغّل الـAPI.

---

## 4️⃣ لو عايز تستخدم موديل الفريق الحقيقي بدل الديمو

الموديل الحالي (`demo_model.joblib`) هو Logistic Regression بسيط عشان الديمو يشتغل من غير أي مكتبات تقيلة. لو شخص 3 عنده موديل XGBoost/LightGBM/LinearSVC مُدرَّب فعليًا:

```python
import joblib
joblib.dump(trained_model, "demo_model.joblib")
```

بس لازم الموديل يكون اتدرّب على نفس الـ80 عمود بنفس الترتيب الموجودين في `demo_model_columns.joblib` (أو `model_data_A_v2_imputed_X_train.csv`). لو الموديل استخدم مكتبة زي xgboost، لازم تضيفها في `requirements.txt`.

---

## ✅ خلاصة تحقق كل بنود المحاضر

| المطلوب | الحالة |
|---|---|
| Save the trained model (pickle/joblib) | ✅ `demo_model.joblib` |
| API بـFlask أو FastAPI | ✅ `api.py` (FastAPI) — مُختبَر فعليًا وشغّال |
| واجهة Streamlit | ✅ `app.py` — مُختبَر فعليًا |
| Host على GitHub + Cloud | ✅ خطوات كاملة فوق (Streamlit Cloud + Render/Heroku) |
