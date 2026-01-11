To run the code follow the below instructions:

Place the data folder inside the root folder 

source code/
│
├── README.md
├── requirements.txt
├── .gitignore│
├── data/

Setting up environment
```
python -m venv venv

.\venv\Scripts\Activate
```
If the features folder not in source code then:
```
python extract_features.py
```
Install dependencies

```
pip install -r requirements.txt
```

Run the app

```
streamlit run app.py
```
